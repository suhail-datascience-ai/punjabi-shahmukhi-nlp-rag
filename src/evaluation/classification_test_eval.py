"""
classification_test_eval.py
----------------------------
Step 0 of the thesis plan.

1. FREEZE the evaluation data. The original project had a train/val/test
   split, but only train and val were saved. The test set is exactly the
   rows of the full dataset that are in neither train nor val (145 rows).
   This script rebuilds it, checks there is no overlap or duplication, and
   writes data/splits/{train,val,test}.csv plus a MANIFEST.json with file
   hashes. Once written, re-running only VERIFIES the hashes: the splits
   never change silently.

2. EVALUATE the existing classifiers once on that test set, which was never
   used for training or model selection:
     - TF-IDF + Logistic Regression (re-fitted here on the frozen train
       split with the same settings as train_baseline.py, so there is no
       pickle/version dependency),
     - the fine-tuned XLM-R in models/xlmr_classifier (if present).
   Reports accuracy and macro F1 with bootstrap 95% confidence intervals,
   and an exact McNemar test between the two models.

Run from the repo root:
    python -m src.evaluation.classification_test_eval
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from src.preprocessing.normalize import normalize_shahmukhi

RAW = Path("data/raw")
SPLITS = Path("data/splits")
OUT = Path("data/processed/classification_test_results.json")
XLMR_DIR = Path("models/xlmr_classifier")
SEED = 42
N_BOOT = 1000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_or_freeze():
    names = ("train", "val", "test")
    paths = {n: SPLITS / f"{n}.csv" for n in names}
    manifest_path = SPLITS / "MANIFEST.json"

    if all(p.exists() for p in paths.values()) and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for n in names:
            assert sha256(paths[n]) == manifest[n]["sha256"], f"{n}.csv changed since it was frozen"
        print("Frozen splits found and verified against MANIFEST.json")
        return tuple(pd.read_csv(paths[n]) for n in names)

    full = pd.read_excel(RAW / "punjabi_shahmukhi_master_dataset.xlsx")[["Text", "Label"]]
    train = pd.read_excel(RAW / "master_train.xlsx")[["Text", "Label"]]
    val = pd.read_excel(RAW / "master_val.xlsx")[["Text", "Label"]]
    seen = set(train.Text) | set(val.Text)
    test = full[~full.Text.isin(seen)].reset_index(drop=True)

    assert not (set(train.Text) & set(val.Text)), "train and val overlap"
    assert not (set(test.Text) & seen), "test overlaps train/val"
    assert full.Text.is_unique, "full dataset has duplicate texts"
    assert len(train) + len(val) + len(test) == len(full), "split sizes do not add up to the full dataset"

    SPLITS.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for n, df in zip(names, (train, val, test)):
        df.to_csv(paths[n], index=False, encoding="utf-8-sig")
        manifest[n] = {
            "rows": len(df),
            "labels": df.Label.value_counts().to_dict(),
            "sha256": sha256(paths[n]),
        }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Froze new splits:", {n: manifest[n]["rows"] for n in names})
    return train, val, test


def make_tfidf_predictor(train: pd.DataFrame):
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True),
            ),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0)),
        ]
    )
    pipe.fit([normalize_shahmukhi(t) for t in train.Text], train.Label)
    return lambda texts: list(pipe.predict([normalize_shahmukhi(t) for t in texts]))


def make_xlmr_predictor(path: Path = XLMR_DIR, batch_size: int = 16):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    max_len = min(192, model.config.max_position_embeddings)

    def predict(texts):
        clean = [normalize_shahmukhi(t) for t in texts]
        preds = []
        for i in range(0, len(clean), batch_size):
            enc = tok(
                clean[i : i + batch_size],
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                padding=True,
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            preds += [model.config.id2label[int(j)] for j in logits.argmax(-1).cpu()]
        return preds

    return predict


def bootstrap_ci(y_true, y_pred, labels, rng):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    accs, f1s = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        accs.append((y_true[idx] == y_pred[idx]).mean())
        f1s.append(f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0))
    return (
        [float(x) for x in np.percentile(accs, [2.5, 97.5])],
        [float(x) for x in np.percentile(f1s, [2.5, 97.5])],
    )


def main() -> None:
    train, val, test = load_or_freeze()
    labels = sorted(train.Label.unique())
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")
    print("test label counts:", test.Label.value_counts().to_dict())

    y, texts = test.Label.tolist(), test.Text.tolist()
    predictors = {"tfidf_logreg": make_tfidf_predictor(train)}
    if XLMR_DIR.exists():
        predictors["xlmr_base_finetuned"] = make_xlmr_predictor()
    else:
        print(f"(skipping XLM-R: {XLMR_DIR} not found)")

    rng = np.random.default_rng(SEED)
    results, preds = {}, {}
    for name, predict in predictors.items():
        p = predict(texts)
        preds[name] = p
        acc = accuracy_score(y, p)
        mf1 = f1_score(y, p, labels=labels, average="macro", zero_division=0)
        acc_ci, f1_ci = bootstrap_ci(y, p, labels, rng)
        results[name] = {
            "accuracy": float(acc),
            "accuracy_ci95": acc_ci,
            "macro_f1": float(mf1),
            "macro_f1_ci95": f1_ci,
        }
        print(f"\n=== {name} on TEST (n={len(y)}) ===")
        print(f"accuracy {acc:.3f}  95% CI [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]")
        print(f"macro F1 {mf1:.3f}  95% CI [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]")
        print(classification_report(y, p, labels=labels, zero_division=0))

    summary = {"n_test": len(y), "results": results}
    if len(preds) == 2:
        a, b = (np.array(preds[k]) == np.array(y) for k in preds)
        only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
        n_disc = only_a + only_b
        p_val = float(binomtest(only_a, n_disc, 0.5).pvalue) if n_disc else 1.0
        names = list(preds)
        summary["mcnemar"] = {
            f"only_{names[0]}_correct": only_a,
            f"only_{names[1]}_correct": only_b,
            "p_value": p_val,
        }
        print(f"\nMcNemar (exact): only {names[0]} correct = {only_a}, only {names[1]} correct = {only_b}, p = {p_val:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
