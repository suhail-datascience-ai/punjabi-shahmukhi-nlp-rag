"""
agreement.py
------------
Inter-annotator agreement for CoNLL-style BIO NER files (one "token tag" per
line, blank line between sentences). Used to validate the gold benchmark.

Reports:
  * token-level Cohen's kappa over the full BIO tag set
  * entity-level exact-match precision / recall / F1 (annotator A as reference)
  * span-only F1 (boundaries agree, type ignored) and type agreement on those spans
  * per-type F1, and a list of disagreements for adjudication

    python -m src.evaluation.agreement data/gold/pilot_A.conll data/gold/pilot_B.conll \
        --out data/gold/pilot_agreement.json --disagreements data/gold/pilot_disagreements.tsv
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_conll(path: str | Path) -> list[list[tuple[str, str]]]:
    sents: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            if cur:
                sents.append(cur)
                cur = []
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"{path}:{ln}: expected 'token tag', got {line!r}")
        cur.append((parts[0], parts[-1]))
    if cur:
        sents.append(cur)
    return sents


def bio_to_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    """BIO tags -> [(start, end_exclusive, type)]. A stray I-X starts a span."""
    spans: list[tuple[int, int, str]] = []
    start, typ = None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t != "O" and not (t.startswith("B-") or t.startswith("I-")):
            raise ValueError(f"invalid tag {t!r}")
        continues = t.startswith("I-") and start is not None and t[2:] == typ
        if not continues:
            if start is not None:
                spans.append((start, i, typ))
                start, typ = None, None
            if t != "O":
                start, typ = i, t[2:]
    return spans


def cohen_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be non-empty and equal length")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def compare(sents_a, sents_b) -> tuple[dict, list[dict]]:
    if len(sents_a) != len(sents_b):
        raise ValueError(f"sentence count differs: {len(sents_a)} vs {len(sents_b)}")
    tags_a, tags_b = [], []
    exact = Counter()          # tp/fp/fn overall
    per_type = defaultdict(Counter)
    span_tp = type_agree = 0
    n_a = n_b = 0
    diffs: list[dict] = []
    for si, (sa, sb) in enumerate(zip(sents_a, sents_b)):
        toks_a, toks_b = [t for t, _ in sa], [t for t, _ in sb]
        if toks_a != toks_b:
            raise ValueError(f"sentence {si}: tokens differ between annotators")
        ta, tb = [g for _, g in sa], [g for _, g in sb]
        tags_a += ta
        tags_b += tb
        spa, spb = bio_to_spans(ta), bio_to_spans(tb)
        set_a, set_b = set(spa), set(spb)
        n_a += len(set_a)
        n_b += len(set_b)
        for s in set_a & set_b:
            exact["tp"] += 1
            per_type[s[2]]["tp"] += 1
        for s in set_a - set_b:
            exact["fn"] += 1
            per_type[s[2]]["fn"] += 1
        for s in set_b - set_a:
            exact["fp"] += 1
            per_type[s[2]]["fp"] += 1
        # boundary agreement ignoring type
        bounds_b = {(s, e): t for s, e, t in spb}
        for s, e, t in spa:
            if (s, e) in bounds_b:
                span_tp += 1
                type_agree += int(bounds_b[(s, e)] == t)
        for s, e, t in sorted(set_a ^ set_b):
            who = "A_only" if (s, e, t) in set_a else "B_only"
            diffs.append({"sentence": si, "annotator": who, "span": " ".join(toks_a[s:e]),
                          "type": t, "start": s, "end": e})
    report = {
        "n_sentences": len(sents_a),
        "n_tokens": len(tags_a),
        "entities_A": n_a,
        "entities_B": n_b,
        "token_kappa_bio": round(cohen_kappa(tags_a, tags_b), 4),
        "entity_exact": _prf(exact["tp"], exact["fp"], exact["fn"]),
        "span_only_f1": _prf(span_tp, n_b - span_tp, n_a - span_tp)["f1"],
        "type_agreement_on_matching_spans": round(type_agree / span_tp, 4) if span_tp else 0.0,
        "per_type": {t: _prf(c["tp"], c["fp"], c["fn"]) for t, c in sorted(per_type.items())},
    }
    return report, diffs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotator_a")
    ap.add_argument("annotator_b")
    ap.add_argument("--out", help="write the JSON report here")
    ap.add_argument("--disagreements", help="write disagreements TSV here")
    args = ap.parse_args()
    report, diffs = compare(read_conll(args.annotator_a), read_conll(args.annotator_b))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.disagreements:
        lines = ["sentence\tannotator\ttype\tspan"] + [
            f"{d['sentence']}\t{d['annotator']}\t{d['type']}\t{d['span']}" for d in diffs]
        Path(args.disagreements).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
