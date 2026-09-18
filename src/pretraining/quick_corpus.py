"""
quick_corpus.py
-----------------
Builds a small text file from data you ALREADY have (no download, no
internet needed) purely so you can smoke-test the pretraining pipeline
(tokenizer -> packing -> training -> checkpoint) in minutes, before
spending hours/days on the real corpus.

This is NOT enough text for a usable model — it's a pipeline test.

Usage:
    python -m src.pretraining.quick_corpus
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.normalize import normalize_shahmukhi

OUT_PATH = Path("data/pretraining_raw/quick_corpus.txt")


def main() -> None:
    docs = []

    df1 = pd.read_excel("data/raw/punjabi_shahmukhi_master_dataset.xlsx")
    docs += [normalize_shahmukhi(t) for t in df1["Text"].dropna().tolist()]

    df2 = pd.read_excel("data/raw/master punjabi dataset.xlsx")
    docs += [normalize_shahmukhi(t) for t in df2["Article"].dropna().tolist()]

    docs = [d for d in docs if len(d) > 20]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(docs))

    n_tokens = sum(len(d.split()) for d in docs)
    print(f"Wrote {len(docs)} documents, ~{n_tokens:,} whitespace tokens -> {OUT_PATH}")
    print("This is a PIPELINE TEST corpus, not a real pretraining corpus. "
          "Move to build_pretraining_corpus.py for the real Tier 1 run.")


if __name__ == "__main__":
    main()
