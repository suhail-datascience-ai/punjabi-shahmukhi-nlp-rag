"""
train_baseline.py
------------------
A genuine, fully-offline, CPU-trainable baseline for the 5-way Punjabi
news classification task (politics / sports / geography / history /
entertainment).

Why a TF-IDF + Linear SVM baseline, and why it matters
=======================================================
Before you fine-tune a 270M-parameter transformer (XLM-R), you need a
number to beat, computed honestly on YOUR data. A classical bag-of-words
baseline:
  * Trains in seconds on a laptop, no GPU, no internet, no HF Hub download.
  * Gives you a real accuracy/F1 to compare the transformer against, so you
    can prove the transformer is actually earning its extra cost and
    complexity — rather than silently reusing the same hard-coded metric
    for every "pipeline" the way the original notebook did.
  * Is what a hiring manager expects to see: a sane baseline before the
    fancy model.

Run:
    python -m src.classification.train_baseline
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from src.preprocessing.normalize import normalize_shahmukhi

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def _load_split(filename: str) -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / filename)
    df["Text"] = df["Text"].apply(normalize_shahmukhi)
    df = df[df["Text"].str.len() > 0].reset_index(drop=True)
    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = _load_split("master_train.xlsx")
    val_df = _load_split("master_val.xlsx")

    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")
    print("Label distribution (train):")
    print(train_df["Label"].value_counts())

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",  # dataset is imbalanced (see label counts)
                    C=5.0,
                ),
            ),
        ]
    )

    pipeline.fit(train_df["Text"], train_df["Label"])

    val_preds = pipeline.predict(val_df["Text"])
    report = classification_report(val_df["Label"], val_preds, output_dict=True, zero_division=0)
    macro_f1 = f1_score(val_df["Label"], val_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(val_df["Label"], val_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(val_df["Label"], val_preds, labels=sorted(train_df["Label"].unique()))

    print("\n=== VALIDATION RESULTS (real, computed just now) ===")
    print(classification_report(val_df["Label"], val_preds, zero_division=0))
    print("Macro F1:", round(macro_f1, 4), "| Weighted F1:", round(weighted_f1, 4))

    joblib.dump(pipeline, MODELS_DIR / "tfidf_logreg_classifier.joblib")

    metrics_out = {
        "accuracy": report["accuracy"],
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "confusion_matrix": cm.tolist(),
        "labels": sorted(train_df["Label"].unique().tolist()),
        "n_train": len(train_df),
        "n_val": len(val_df),
    }
    with open(PROCESSED_DIR / "classification_baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    print(f"\nSaved model -> {MODELS_DIR / 'tfidf_logreg_classifier.joblib'}")
    print(f"Saved metrics -> {PROCESSED_DIR / 'classification_baseline_metrics.json'}")


if __name__ == "__main__":
    main()
