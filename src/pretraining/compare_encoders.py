"""
compare_encoders.py
---------------------
Step 4 of Phase 2: the actual headline result. Fine-tunes the SAME
classification task (reusing your existing labeled data) on top of both:
  (a) your new from-scratch compact encoder
  (b) xlm-roberta-base
and prints both real numbers side by side. This head-to-head is what makes
a "compact model matches/beats a much larger multilingual model" claim
legitimate — it only counts if you actually run both and report both,
including if your model *loses* on some metric (say so; a partial win on
a 10x-smaller model is still a real result).

Usage:
    python -m src.pretraining.compare_encoders
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.preprocessing.normalize import normalize_shahmukhi

PARAM_ESTIMATE = {"tiny": "~1.5M", "small": "~8M (measured: 7,986,629)", "compact": "~25M"}


def load_data():
    train_df = pd.read_excel("data/raw/master_train.xlsx")
    val_df = pd.read_excel("data/raw/master_val.xlsx")
    for df in (train_df, val_df):
        df["Text"] = df["Text"].apply(normalize_shahmukhi)
    labels = sorted(train_df["Label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    train_df["label"] = train_df["Label"].map(label2id)
    val_df["label"] = val_df["Label"].map(label2id)
    return train_df, val_df, label2id


def run_one(model_name: str, model_path: str, train_df, val_df, label2id) -> dict:
    from transformers import AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    cfg = AutoConfig.from_pretrained(model_path)
    max_len = min(192, getattr(cfg, "max_position_embeddings", 192))
    print(f"  (using max_length={max_len} for {model_name})")

    def tok(batch):
        return tokenizer(batch["Text"], truncation=True, max_length=max_len)

    train_ds = Dataset.from_pandas(train_df[["Text", "label"]]).map(tok, batched=True)
    val_ds = Dataset.from_pandas(val_df[["Text", "label"]]).map(tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(label2id))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": (preds == labels).mean(),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir=f"/tmp/compare_{model_name.split()[0]}",
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=8,
        learning_rate=3e-5,
        logging_steps=50,
        report_to=[],
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    n_params = sum(p.numel() for p in model.parameters())
    return {"model": model_name, "n_params": n_params, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["tiny", "small", "compact"], default="small",
                         help="Which tier of your pretrained encoder to compare (must match what you passed "
                              "to train_mlm.py — the output folder is models/punjabi_shahmukhi_encoder_<config>).")
    args = parser.parse_args()

    encoder_path = f"models/punjabi_shahmukhi_encoder_{args.config}"
    candidates = {
        f"punjabi_shahmukhi_encoder_{args.config} (yours, {PARAM_ESTIMATE[args.config]} params)": encoder_path,
        "xlm-roberta-base (270M params)": "xlm-roberta-base",
    }

    train_df, val_df, label2id = load_data()
    results = []
    for name, path in candidates.items():
        print(f"\n=== Fine-tuning: {name} ===")
        try:
            results.append(run_one(name, path, train_df, val_df, label2id))
        except OSError as e:
            print(f"Skipping {name}: {e} (train it first, or check the path)")

    print("\n=== HEAD-TO-HEAD RESULT (real, computed just now) ===")
    print(pd.DataFrame(results).to_string(index=False))
    print(
        "\nReport this table as-is in your README/model card/LinkedIn post — "
        "whichever direction it goes."
    )


if __name__ == "__main__":
    main()
