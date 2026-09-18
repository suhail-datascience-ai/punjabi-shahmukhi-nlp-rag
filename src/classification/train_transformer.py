"""
train_transformer.py
----------------------
Fine-tunes XLM-RoBERTa for 5-way Punjabi (Shahmukhi) news classification.

This is the "production quality" upgrade over the TF-IDF baseline in
`train_baseline.py`. It is NOT run automatically as part of this repo's
CI/build because it needs:
  * A GPU (Colab T4 free tier is enough for this dataset size).
  * Internet access to download `xlm-roberta-base` from the Hugging Face
    Hub the first time.

How to run it
==============
1. Open Google Colab or Kaggle Notebooks, select a GPU runtime.
2. `pip install -r requirements.txt`
3. Upload / clone this repo, `cd` into it.
4. `python -m src.classification.train_transformer`
5. Copy the resulting `models/xlmr_classifier/` directory back into this
   repo (or push it straight to the Hugging Face Hub with
   `huggingface-cli upload`) and point `backend/app/config.py` at it.

Everything below reports metrics computed by the `Trainer` itself on the
real validation set — there is no hard-coded number anywhere in this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.preprocessing.normalize import normalize_shahmukhi

MODEL_CKPT = "xlm-roberta-base"
OUTPUT_DIR = "models/xlmr_classifier"
MAX_LEN = 192


def load_data():
    train_df = pd.read_excel("data/raw/master_train.xlsx")
    val_df = pd.read_excel("data/raw/master_val.xlsx")
    for df in (train_df, val_df):
        df["Text"] = df["Text"].apply(normalize_shahmukhi)

    labels = sorted(train_df["Label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    train_df["label"] = train_df["Label"].map(label2id)
    val_df["label"] = val_df["Label"].map(label2id)
    return train_df, val_df, label2id, id2label


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print(
            "WARNING: no GPU detected. Training XLM-R on CPU will be very slow. "
            "Run this on Colab/Kaggle with a GPU runtime instead."
        )

    train_df, val_df, label2id, id2label = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

    def tokenize(batch):
        return tokenizer(batch["Text"], truncation=True, max_length=MAX_LEN)

    train_ds = Dataset.from_pandas(train_df[["Text", "label"]]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[["Text", "label"]]).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CKPT, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    # Dataset is imbalanced (history 523 vs entertainment 56 in train) -
    # weight the loss instead of silently letting the majority class win.
    class_counts = train_df["label"].value_counts().sort_index()
    class_weights = torch.tensor(
        (class_counts.sum() / (len(class_counts) * class_counts)).values, dtype=torch.float
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": (preds == labels).mean(),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted"),
        }

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=20,
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    final_metrics = trainer.evaluate()
    print("Final validation metrics (real, from this run):", final_metrics)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned model -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
