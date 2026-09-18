"""
lexicon_bootstrap.py
---------------------
Distant-supervision BIO tagger for Shahmukhi Punjabi Named Entity
Recognition.

IMPORTANT — read this before quoting NER numbers anywhere public
==================================================================
The labels produced here are **silver-standard**, not gold-standard: every
tag comes from a fixed dictionary/gazetteer match, not from a human
annotator. This is a completely standard and legitimate way to *bootstrap*
a low-resource NER dataset (it is exactly how many early NER corpora for
under-resourced languages were built), but it has two known failure modes
that you must disclose in your README/model card if you publish this:

  1. Recall ceiling: any entity not in `NER_SEEDS` is silently tagged "O".
     The model will therefore inherit the lexicon's blind spots.
  2. No disambiguation: a surface form is tagged the same way everywhere,
     even where it is not actually an entity in context (e.g. "پنجاب" is
     always LOC even if used adjectivally).

Recommended path to a defensible NER claim (in increasing cost order):
  (a) Ship this as "weak/distant-supervision NER, seed-lexicon based" —
      honest and still useful for a portfolio project.
  (b) Manually review + correct a random sample (150-300 sentences) and
      report agreement between the auto-tags and your corrections — this
      is what `src/evaluation/evaluate.py::human_review_sample()` is for.
  (c) Best: have 2 people independently label a small gold set and report
      Cohen's Kappa (already scaffolded in the original notebook) as your
      real NER quality metric, and train a transformer against a mix of
      silver + gold data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from src.preprocessing.normalize import normalize_shahmukhi

# ---------------------------------------------------------------------------
# Seed gazetteer. Expanded from the project's original prototype with a
# broader set of high-frequency Punjabi/Pakistani/Subcontinental entities.
# Extend this list as you find false negatives during manual review.
# ---------------------------------------------------------------------------
NER_SEEDS: dict[str, list[str]] = {
    "PER": [
        "محمد", "سهیل", "بلھے شاہ", "وارث شاہ", "بابا فرید", "رنجیت سنگھ", "سکندر",
        "چنگیز خان", "اشوک", "عمران خان", "نواز شریف", "قائد اعظم", "جناح", "اقبال",
        "بھگت سنگھ", "گرو نانک", "شاہ حسین", "میاں محمد بخش", "ذوالفقار علی بھٹو",
        "بے نظیر بھٹو", "شہباز شریف", "آصف علی زرداری", "بابر اعظم", "شاہد آفریدی",
    ],
    "LOC": [
        "پنجاب", "پاکستان", "لاهور", "کراچی", "اسلام آباد", "ملتان", "فیصل آباد", "سیالکوٹ",
        "بھارت", "ہند", "سرگودھا", "راولپنڈی", "کشمیر", "امرتبھر", "دہلی", "لندن", "ایشیا",
        "گجرانوالہ", "بہاولپور", "پشاور", "کوئٹہ", "چناب", "راوی", "ستلج", "جہلم", "افغانستان",
    ],
    "ORG": [
        "حکومت", "سلطنت", "رومی سلطنت", "منگول سلطنت", "کمپنی", "ایسٹ انڈیا کمپنی",
        "فوج", "پی سی بی", "مجلس", "تحریک انصاف", "مسلم لیگ", "پیپلز پارٹی", "اقوام متحدہ",
        "آئی سی سی", "سپریم کورٹ", "پارلیمنٹ", "سینٹ",
    ],
    "MISC": [
        "کرکٹ ورلڈ کپ", "پی ایس ایل", "ورلڈ کپ", "جنگ", "معاهدہ", "اولمپکس", "آئی پی ایل",
        "مولا جٹ", "پنجابی کلتھر", "کلب", "فیسٹیول", "میلہ", "بیساکھی",
    ],
}

_TOKEN_STRIP_RE = re.compile(r"[^\w\s۔!?؟]", re.UNICODE)


def clean_and_tokenize(text: str) -> list[str]:
    """Whitespace tokenizer that preserves native sentence punctuation."""
    if not isinstance(text, str):
        return []
    text = _TOKEN_STRIP_RE.sub("", text)
    return text.split()


@dataclass
class Gazetteer:
    """Pre-sorted gazetteer for greedy longest-span matching."""

    seeds: dict[str, list[str]] = field(default_factory=lambda: NER_SEEDS)
    _sorted_entries: list[tuple[list[str], str]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        entries = []
        for entity_type, phrases in self.seeds.items():
            for phrase in phrases:
                words = phrase.split()
                entries.append((words, entity_type))
        # Longest spans first so multi-word entities win over single-word
        # substrings (e.g. "رومی سلطنت" over bare "سلطنت").
        self._sorted_entries = sorted(entries, key=lambda e: -len(e[0]))

    def tag(self, tokens: list[str]) -> list[str]:
        tags = ["O"] * len(tokens)
        n = len(tokens)
        for words, entity_type in self._sorted_entries:
            span = len(words)
            for i in range(n - span + 1):
                if tags[i] != "O":
                    continue
                if tokens[i : i + span] == words:
                    tags[i] = f"B-{entity_type}"
                    for j in range(1, span):
                        tags[i + j] = f"I-{entity_type}"
        return tags


def annotate_dataframe(df: pd.DataFrame, text_col: str = "Text") -> pd.DataFrame:
    """Return a copy of `df` with an `Annotated_Sequence` BIO column."""
    gaz = Gazetteer()
    out_rows = []
    sequences = []
    for _, row in df.iterrows():
        raw = row[text_col]
        tokens = clean_and_tokenize(normalize_shahmukhi(raw))
        if not tokens:
            sequences.append("")
            continue
        tags = gaz.tag(tokens)
        sequences.append(" ".join(f"{t}/{g}" for t, g in zip(tokens, tags)))
    df = df.copy()
    df["Annotated_Sequence"] = sequences
    return df[df["Annotated_Sequence"] != ""].reset_index(drop=True)


def entity_coverage_report(df: pd.DataFrame) -> dict:
    """Quick health-check: what fraction of tokens got a non-O tag?

    Low coverage (<5-10%) is a signal your gazetteer needs expanding before
    you trust the silver labels for training.
    """
    gaz = Gazetteer()
    total_tokens = 0
    tagged_tokens = 0
    per_type = {t: 0 for t in NER_SEEDS}
    for _, row in df.iterrows():
        tokens = clean_and_tokenize(normalize_shahmukhi(row["Text"]))
        if not tokens:
            continue
        tags = gaz.tag(tokens)
        total_tokens += len(tokens)
        for tag in tags:
            if tag != "O":
                tagged_tokens += 1
                per_type[tag.split("-")[1]] = per_type.get(tag.split("-")[1], 0) + 1
    return {
        "total_tokens": total_tokens,
        "tagged_tokens": tagged_tokens,
        "coverage_pct": round(100 * tagged_tokens / max(total_tokens, 1), 2),
        "per_entity_type_tag_count": per_type,
    }


if __name__ == "__main__":
    import sys

    src_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/punjabi_shahmukhi_master_dataset.xlsx"
    df = pd.read_excel(src_path)
    annotated = annotate_dataframe(df)
    report = entity_coverage_report(df)
    print("Coverage report:", report)
    annotated.to_csv("data/processed/punjabi_shahmukhi_unified_annotated.tsv", sep="\t", index=False, encoding="utf-8-sig")
    print(f"Wrote {len(annotated)} annotated rows.")
