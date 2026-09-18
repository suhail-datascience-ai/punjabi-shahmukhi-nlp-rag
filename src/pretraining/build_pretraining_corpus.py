"""
build_pretraining_corpus.py
-----------------------------
Step 1 of Phase 2 (from-scratch encoder pretraining): assemble a large raw
(unlabeled) Shahmukhi Punjabi text corpus. Your existing scraper.py pulled
~1,450 *labeled* articles from 5 category hubs — that's a fine-tuning-scale
dataset. Pretraining needs orders of magnitude more raw text, so this script
takes the highest-yield free source: the FULL Wikimedia dump for pnb.wikipedia
(every article, not just 5 categories), rather than more category-hub
crawling.

Usage (run on your own machine — needs real bandwidth, ~200MB-1GB download):

    pip install wikiextractor
    python -m src.pretraining.build_pretraining_corpus --download
    python -m src.pretraining.build_pretraining_corpus --extract
    python -m src.pretraining.build_pretraining_corpus --clean

Expect this dump alone to yield somewhere in the low tens of millions of
tokens (Shahmukhi/Western Punjabi Wikipedia is a small wiki). To reach the
Sindhi release's 327M-token scale you will likely also need:
  - OSCAR or CC-100/mC4, filtered to Perso-Arabic script + Punjabi-specific
    stopwords/vocabulary (neither ships a clean "pnb" split by default —
    you'll filter a larger "ur"/mixed-script dump yourself, see
    `filter_script()` below).
  - Any Shahmukhi-script news sites, books, or blogs you have the rights to
    use — check robots.txt / ToS per site before scraping.
Treat the Wikipedia dump as your first, cleanest 20-40M tokens, and layer
other sources on top once this pipeline is proven end-to-end.
"""

from __future__ import annotations

import argparse
import bz2
import re
import subprocess
import unicodedata
from pathlib import Path

import requests

DUMP_INDEX_URL = "https://dumps.wikimedia.org/pnbwiki/latest/"
DUMP_FILENAME = "pnbwiki-latest-pages-articles.xml.bz2"
RAW_DIR = Path("data/pretraining_raw")
EXTRACTED_DIR = Path("data/pretraining_raw/extracted")
CORPUS_OUT = Path("data/pretraining_raw/corpus.txt")

# Same allowed-character policy as src/preprocessing/normalize.py, so the
# pretraining corpus and the fine-tuning corpus are script-consistent.
_ALLOWED_RE = re.compile(r"[^\u0600-\u06FF\u0750-\u077F0-9۰-۹\s.,؟،!٪:؛\-()\"'’‘]")


def download_dump() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = DUMP_INDEX_URL + DUMP_FILENAME
    out_path = RAW_DIR / DUMP_FILENAME
    print(f"Downloading {url} ...")
    headers = {"User-Agent": "PunjabiShahmukhiNLPProject/1.0 (sohailakmal275@gmail.com) python-requests"}
    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"Saved -> {out_path}")
    return out_path


def extract_dump(dump_path: Path = RAW_DIR / DUMP_FILENAME) -> None:
    """Uses the `wikiextractor` package (pip install wikiextractor) to turn
    the raw XML dump into plain-text article files."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["python", "-m", "wikiextractor.WikiExtractor", str(dump_path), "-o", str(EXTRACTED_DIR), "--no-templates"],
        check=True,
    )
    print(f"Extracted -> {EXTRACTED_DIR}")


def filter_script(text: str, min_shahmukhi_ratio: float = 0.6) -> str | None:
    """Reject a document if it isn't majority Perso-Arabic script — useful
    when mining a mixed-language dump (e.g. filtering a larger ur/mixed
    corpus down to Shahmukhi-heavy documents)."""
    if not text.strip():
        return None
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    if arabic_chars / max(len(text), 1) < min_shahmukhi_ratio:
        return None
    return text


def clean_and_merge() -> None:
    lines_out = []
    seen_hashes: set[int] = set()
    for path in EXTRACTED_DIR.rglob("wiki_*"):
        with open(path, encoding="utf-8") as f:
            buffer = []
            for line in f:
                line = line.strip()
                if line.startswith("<doc") or line.startswith("</doc>") or not line:
                    if buffer:
                        doc = unicodedata.normalize("NFKC", " ".join(buffer))
                        doc = _ALLOWED_RE.sub(" ", doc)
                        doc = re.sub(r"\s+", " ", doc).strip()
                        if doc and filter_script(doc):
                            h = hash(doc[:200])  # cheap near-dup guard on the opening text
                            if h not in seen_hashes:
                                seen_hashes.add(h)
                                lines_out.append(doc)
                    buffer = []
                    continue
                buffer.append(line)

    CORPUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))

    total_tokens = sum(len(d.split()) for d in lines_out)
    print(f"Wrote {len(lines_out)} documents, ~{total_tokens:,} whitespace tokens -> {CORPUS_OUT}")
    print(
        "Compare this number honestly against the 327M-token Sindhi corpus in your "
        "model card before claiming comparable scale."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.download:
        download_dump()
    if args.extract:
        extract_dump()
    if args.clean:
        clean_and_merge()
    if not any([args.download, args.extract, args.clean]):
        parser.print_help()
