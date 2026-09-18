"""
normalize.py
------------
Text normalization utilities for Punjabi written in the Shahmukhi
(Perso-Arabic) script.

Why this exists
================
Shahmukhi text scraped from the web carries a lot of noise that hurts both
classification and NER models if left untouched:

  * Multiple Unicode encodings of the same visual character (e.g. Yeh vs.
    Farsi Yeh, Kaf vs. Keheh) which silently fragments the vocabulary.
  * Wikipedia markup remnants: reference tags, bracketed citation numbers,
    stray English tokens, URLs.
  * Kashida / tatweel elongation characters that add no linguistic
    information but inflate token counts.
  * Inconsistent whitespace (including Zero-Width Non-Joiner artifacts from
    copy-pasted web content).

This module centralises all of that so every part of the pipeline (EDA,
classifier training, NER training, RAG indexing) normalizes text in exactly
the same way. Without this, a document indexed one way and queried another
way will silently fail to match.
"""

from __future__ import annotations

import re
import unicodedata

# Perso-Arabic presentation-form / compatibility characters that should be
# folded onto their canonical counterparts before anything else happens.
_CHAR_MAP = {
    "ك": "ک",   # Arabic Kaf -> Shahmukhi Keheh
    "ي": "ی",   # Arabic Yeh -> Shahmukhi Yeh
    "ى": "ی",   # Alef Maksura -> Yeh
    "ۀ": "ہ",   # Heh with Yeh above -> Heh Goal
    "ھ": "ھ",   # (kept, doachashmee heh, distinct phoneme in Shahmukhi)
    "\u200c": " ",  # Zero-width non-joiner -> space (ZWNJ is meaningful in
                     # some Nastaliq fonts but breaks tokenizers reliably)
    "\u200b": "",   # Zero-width space -> drop
    "\ufeff": "",   # BOM -> drop
    "ـ": "",        # Tatweel/Kashida elongation -> drop
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_CITATION_RE = re.compile(r"\[\d+\]|\[cite[^\]]*\]", re.IGNORECASE)
_MULTI_WS_RE = re.compile(r"\s+")
# Keep Arabic/Shahmukhi block, Arabic presentation forms, digits (both
# Eastern-Arabic and Latin), and basic punctuation. Strip everything else
# (stray Latin words, emoji, HTML leftovers) which is standard practice for
# a monolingual corpus intended for a Punjabi-only model.
_ALLOWED_RE = re.compile(
    r"[^\u0600-\u06FF\u0750-\u077F0-9۰-۹\s.,؟،!٪:؛\-()\"'’‘]"
)


def normalize_shahmukhi(text: str) -> str:
    """Apply strict, deterministic normalization to a Shahmukhi string.

    This must be called on BOTH training text and inference-time text
    (classification, NER, RAG queries) — any drift between the two is a
    silent source of accuracy loss.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = unicodedata.normalize("NFKC", text)

    for bad, good in _CHAR_MAP.items():
        text = text.replace(bad, good)

    text = _URL_RE.sub(" ", text)
    text = _CITATION_RE.sub(" ", text)
    text = _ALLOWED_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()

    return text


def simple_word_tokenize(text: str) -> list[str]:
    """Whitespace + punctuation-aware tokenizer.

    Shahmukhi has no reliable off-the-shelf tokenizer in most NLP toolkits,
    so word-level splitting on normalized whitespace plus punctuation is the
    standard, defensible baseline used by this project. Sub-word tokenization
    for transformer models is handled separately by the model's own
    tokenizer (e.g. the XLM-R SentencePiece model).
    """
    text = normalize_shahmukhi(text)
    if not text:
        return []
    # split but keep punctuation as separate tokens
    return re.findall(r"[\u0600-\u06FF\u0750-\u077F0-9۰-۹]+|[.,؟،!٪:؛]", text)


if __name__ == "__main__":
    sample = "آزادی دا مطلب  اے[1]  ۔۔۔ https://pnb.wikipedia.org/wiki/x   کي"
    print(repr(normalize_shahmukhi(sample)))
    print(simple_word_tokenize(sample))
