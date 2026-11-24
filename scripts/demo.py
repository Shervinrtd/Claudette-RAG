"""Web demo (Phase 7): check one clause or a whole Terms of Service in the browser.

For every sentence the demo runs the evaluated system, RAG + evidence gate, and shows the verdict, the
model's explanation, the legal rationales and provisions it cited, the expert-labelled precedents and
each step of the decision. Without a language model it runs "Evidence only" (the kNN baseline).

Usage (from the project root, after scripts/build_knowledge_base.py and scripts/build_index.py):
    python scripts/demo.py                         # gpt-4.1-mini, the main model (OpenAI key in .env)
    python scripts/demo.py --evidence-only         # no language model: free and offline
    python scripts/demo.py --model qwen3:8b --base-url http://localhost:11434/v1          # local model (Ollama)
    python scripts/demo.py --model Qwen/Qwen3.6-27B-FP8 --base-url http://localhost:8000/v1 --no-thinking   # vLLM
Then open http://127.0.0.1:7860 (the browser opens by itself).

Options:
    --port 7861          another port
    --embed-device cpu   embeddings on the CPU (keeps the GPU free for a local model; slower for long documents)
    --share              temporary public link (anyone who has it can use the demo, and your API credits)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from claudette_rag.demo import build_app, launch
from claudette_rag.llm import DEFAULT_MODEL, OLLAMA_URL, make_client, server_options
from claudette_rag.pipeline import ClauseAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", help=f"OpenAI-compatible server of a local model, e.g. {OLLAMA_URL} (Ollama)")
    parser.add_argument("--no-thinking", action="store_true", help="local server (vLLM): no thinking phase")
    parser.add_argument("--extra-body", help="local server: extra request parameters as JSON text")
    parser.add_argument("--evidence-only", action="store_true", help="no language model (kNN over the precedents)")
    parser.add_argument("--workers", type=int, help="parallel model requests for documents (default 8, or 1 local)")
    parser.add_argument("--index", default="bge-m3", help="folder in data/processed/index/")
    parser.add_argument("--embed-device", choices=["cuda", "cpu"], help="device for the embeddings (default: GPU if any)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a temporary public link")
    args = parser.parse_args()

    model, client = None, None
    if not args.evidence_only:
        try:
            client = make_client(args.base_url, PROJECT_ROOT)
            model = args.model
        except SystemExit as missing_key:
            print(f"{missing_key}\nStarting with 'Evidence only' (no language model).")
    if args.base_url and "11434" in args.base_url:
        print("Ollama: start it with OLLAMA_CONTEXT_LENGTH=8192, otherwise the long RAG prompts are cut silently.")

    print("Loading the knowledge base, the index and the embedding model ...")
    analyzer = ClauseAnalyzer(PROJECT_ROOT, model=model, client=client,
                              extra_body=server_options(args.extra_body, args.no_thinking), index_name=args.index,
                              workers=args.workers or (1 if args.base_url else 8), embed_device=args.embed_device)
    print(f"Model: {model or 'none (evidence only)'} | evidence gate k={analyzer.gate['k']}, m={analyzer.gate['m']} "
          f"({analyzer.gate_source})")
    if args.share:
        print("WARNING: anyone with the public link can use the demo" + (" and your API credits." if model and
                                                                           not args.base_url else "."))
    launch(build_app(analyzer, PROJECT_ROOT), server_name="127.0.0.1", server_port=args.port, share=args.share,
           inbrowser=True)


if __name__ == "__main__":
    main()
