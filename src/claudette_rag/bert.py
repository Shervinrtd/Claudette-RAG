"""Fine-tune a BERT-style encoder for multi-label unfair-clause classification.

The model reads one sentence and outputs 8 independent probabilities, one per
unfairness category (sigmoid + binary cross-entropy). A category is predicted when its
probability is >= 0.5; a sentence with no predicted category is fair.

Training choices (close to the LexGLUE setup):
- AdamW, learning rate 3e-5, linear schedule with 10% warm-up, mixed precision on GPU;
- early stopping on the validation LexGLUE macro-F1 (patience 3 epochs): macro-F1 gives
  rare categories the same weight as frequent ones;
- several random seeds, because results of small fine-tuning runs vary between seeds.
"""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from claudette_rag.evaluation import evaluate, to_matrix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(records: list[dict], tokenizer, n_labels: int, max_length: int,
                batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    """Tokenize once, then pad each batch dynamically to its longest sentence."""
    encodings = tokenizer([r["text"].strip() for r in records], truncation=True, max_length=max_length)
    labels = to_matrix([r["labels"] for r in records], n_labels).astype(np.float32)
    items = [{key: encodings[key][i] for key in encodings} | {"labels": labels[i]}
             for i in range(len(records))]

    def collate(batch):
        features = [{k: v for k, v in item.items() if k != "labels"} for item in batch]
        padded = tokenizer.pad(features, return_tensors="pt")
        padded["labels"] = torch.tensor(np.stack([item["labels"] for item in batch]))
        return padded

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=generator if shuffle else None)


@torch.no_grad()
def predict_probs(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.config.num_labels))


def probs_to_labels(probs: np.ndarray, threshold: float = 0.5) -> list[list[int]]:
    return [list(map(int, np.flatnonzero(row >= threshold))) for row in probs]


def train(model_name: str, train_records: list[dict], val_records: list[dict], label_names: list[str],
          seed: int = 1, epochs: int = 20, lr: float = 3e-5, batch_size: int = 16,
          max_length: int = 256, patience: int = 3, threshold: float = 0.5,
          device: torch.device | None = None, log=print):
    """Fine-tune and return (model with the best validation weights, tokenizer, history)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    n_labels = len(label_names)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_labels, problem_type="multi_label_classification").to(device)

    train_loader = make_loader(train_records, tokenizer, n_labels, max_length, batch_size, True, seed)
    val_loader = make_loader(val_records, tokenizer, n_labels, max_length, 64, False)
    val_gold = [r["labels"] for r in val_records]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_score, best_state, best_epoch, bad_epochs, history = -1.0, None, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        start, total_loss = time.time(), 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # In mixed precision the first steps can be skipped (gradient overflow): the scale
            # then decreases. Only advance the learning-rate schedule after a real update.
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            total_loss += loss.item()

        val_pred = probs_to_labels(predict_probs(model, val_loader, device), threshold)
        headline = evaluate(val_gold, val_pred, label_names, n_bootstrap=0)["headline"]
        score = headline["lexglue_macro_f1"]
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_loader),
                        "val_macro_f1": score, "val_micro_f1": headline["lexglue_micro_f1"],
                        "seconds": round(time.time() - start, 1)})
        log(f"  epoch {epoch:>2}: loss {history[-1]['train_loss']:.4f} | "
            f"val macro-F1 {100 * score:.1f} | micro-F1 {100 * headline['lexglue_micro_f1']:.1f} "
            f"| {history[-1]['seconds']}s")

        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"  early stopping: no improvement for {patience} epochs")
                break

    model.load_state_dict(best_state)
    log(f"  best epoch {best_epoch} (val macro-F1 {100 * best_score:.1f})")
    return model, tokenizer, {"epochs": history, "best_epoch": best_epoch, "best_val_macro_f1": best_score}
