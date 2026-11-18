"""Check that the OpenAI API key works and list the models your account can use.

Listing models is free: no tokens are used.

Usage (from the project root, after creating the .env file):
    python scripts/check_openai.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not found. Copy .env.example to .env and paste your key into it.")

    client = OpenAI()
    try:
        models = sorted(m.id for m in client.models.list())
    except Exception as error:  # wrong key, no credit, network problem...
        sys.exit(f"The API key does not work: {type(error).__name__}: {error}")

    text_models = [m for m in models if m.startswith(("gpt-", "o1", "o3", "o4"))
                   and not any(x in m for x in ("audio", "realtime", "image", "tts", "transcribe", "search"))]
    print(f"The key works. {len(models)} models available, {len(text_models)} text models:")
    for m in text_models:
        print("  ", m)
    small = [m for m in text_models if "mini" in m or "nano" in m]
    if small:
        print("\nSmall, cheap models (candidates for this project):", ", ".join(small))


if __name__ == "__main__":
    main()
