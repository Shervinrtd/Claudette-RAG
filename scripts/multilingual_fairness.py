"""Does the system protect consumers equally in every language? Group fairness by language.

The groups are the languages of the multilingual evaluation set: the Terms of Service of the same
10 companies in English, German, Italian and Polish, annotated by legal experts. English, the
language of the training data and of the knowledge base, is the reference group. For each system
and language the script reports:

- recall: unfair clauses caught (equality of opportunity asks for the same recall in every group);
- false-alarm rate: fair sentences flagged (equal false positive rates; with recall: equalized odds);
- precision: flags that are right (1 - precision: the conditional use error of the flags);
- flagged: share of sentences flagged (statistical parity; the share of unfair sentences is
  about the same in every language, so large differences here signal a bias);
- LexGLUE macro-F1.

Each language is compared with English through the ratio of the rates (the AEQUITAS convention:
a ratio between 0.8 and 1.25 counts as parity, the four-fifths rule) and the difference, with a
95% bootstrap interval (the two languages are resampled independently). For systems with several
seeds (Legal-BERT) the rates are averaged over the seeds.

Usage (from the project root):
    python scripts/multilingual_fairness.py --model Qwen/Qwen3.6-27B-FP8
    python scripts/multilingual_fairness.py --model gpt-4.1-mini-2025-04-14
Systems whose predictions are missing are skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from claudette_rag.evaluation import align, bootstrap_gap, group_rates, load_gold, read_jsonl, to_matrix
from claudette_rag.llm import file_safe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
LANGUAGES = ["en", "de", "it", "pl"]
RATES = ["recall", "false_alarm_rate", "precision", "flagged", "macro_f1"]
PARITY = (0.8, 1.25)


def systems_for(model: str) -> dict[str, list[str]]:
    """System name -> prediction file name patterns ({lang} is filled in), one per seed."""
    m = file_safe(model)
    systems = {f"Plain LLM ({model})": [f"llm_zeroshot_{m}_multilingual_{{lang}}"],
               f"RAG ({model})": [f"rag_{m}_e8r5p6_multilingual_{{lang}}"]}
    choice = RESULTS / f"gate_choice_{m}_e8r5p6.json"
    if choice.exists():
        c = json.loads(choice.read_text(encoding="utf-8"))
        systems[f"RAG + gate ({model})"] = [f"gated_k{c['k']}m{c['m']}_rag_{m}_e8r5p6_multilingual_{{lang}}"]
    systems["kNN vote (bge-m3)"] = ["knn_bge-m3_multilingual_{lang}"]
    systems["TF-IDF + SVM (English)"] = ["tfidf_svm_multilingual_{lang}"]
    systems["Legal-BERT (English), mean of seeds"] = [
        f"bert_legal-bert-base-uncased_hpc_seed{s}_multilingual_{{lang}}" for s in (1, 2, 3)]
    return systems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8", help="LLM whose runs are analysed")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    gold = {lang: load_gold(PROJECT_ROOT, "multilingual", language=lang) for lang in LANGUAGES}
    y_true = {lang: to_matrix([g["labels"] for g in gold[lang]], 8) for lang in LANGUAGES}
    report = {"model": args.model, "reference": "en", "parity_band": PARITY, "systems": {}}

    for system, patterns in systems_for(args.model).items():
        files = {lang: [PREDICTIONS / f"{p.format(lang=lang)}.jsonl" for p in patterns] for lang in LANGUAGES}
        files = {lang: [f for f in fs if f.exists()] for lang, fs in files.items()}
        if not all(files.values()):
            missing = [lang for lang, fs in files.items() if not fs]
            print(f"-- {system}: skipped (no predictions for {', '.join(missing)})")
            continue
        preds = {lang: [to_matrix(align(gold[lang], read_jsonl(f))[1], 8) for f in fs]
                 for lang, fs in files.items()}
        rates = {}
        for lang in LANGUAGES:
            per_seed = [group_rates(y_true[lang], p) for p in preds[lang]]
            rates[lang] = {k: sum(r[k] for r in per_seed) / len(per_seed) for k in RATES}
        entry = {"seeds": len(patterns), "languages": {}}
        print(f"\n=== {system}" + (f" ({len(files['en'])} seeds)" if len(patterns) > 1 else ""))
        print(f"{'':4s} {'n':>5s} {'unfair':>6s} {'recall':>7s} {'FA rate':>8s} {'precision':>9s} {'flagged':>8s} "
              f"{'macro-F1':>8s} | {'recall ratio':>12s} {'FA ratio':>8s} | {'recall diff [95% CI]':>22s} "
              f"{'macro-F1 diff [95% CI]':>24s}")
        for lang in LANGUAGES:
            r = rates[lang]
            n, n_unfair = len(gold[lang]), int(y_true[lang].any(axis=1).sum())
            row = {"n": n, "unfair": n_unfair, **r}
            line = (f"{lang:4s} {n:5d} {n_unfair:6d} {100 * r['recall']:7.1f} {100 * r['false_alarm_rate']:8.1f} "
                    f"{100 * r['precision']:9.1f} {100 * r['flagged']:8.1f} {100 * r['macro_f1']:8.1f} | ")
            if lang == "en":
                print(line + "reference")
            else:
                ref = rates["en"]
                ratios = {k: (r[k] / ref[k] if ref[k] else None) for k in RATES}
                gap = bootstrap_gap(y_true["en"], preds["en"], y_true[lang], preds[lang], args.n_bootstrap)
                row.update({"ratio_to_en": ratios, "difference_to_en": gap,
                            "recall_parity": ratios["recall"] is not None and PARITY[0] <= ratios["recall"] <= PARITY[1]})

                def fmt_ratio(v):
                    return "   -" if v is None else f"{v:.2f}" + ("" if PARITY[0] <= v <= PARITY[1] else "!")

                def fmt_gap(g):
                    star = "*" if g["p_value"] < 0.05 else " "
                    return f"{100 * g['difference']:+5.1f} [{100 * g['low']:+5.1f},{100 * g['high']:+5.1f}]{star}"
                print(line + f"{fmt_ratio(ratios['recall']):>12s} {fmt_ratio(ratios['false_alarm_rate']):>8s} | "
                      f"{fmt_gap(gap['recall']):>22s} {fmt_gap(gap['macro_f1']):>24s}")
            entry["languages"][lang] = row
        report["systems"][system] = entry

    print("\nFA = false alarm. Ratio language / English: '!' = outside the parity band 0.8-1.25 (four-fifths rule).")
    print("Difference language - English in points: '*' = significant at 5% (the interval does not contain 0).")
    out = RESULTS / f"multilingual_fairness_{file_safe(args.model)}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved -> {out.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
