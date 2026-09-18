"""
train_transformer.py (NER)
----------------------------
Fine-tunes XLM-RoBERTa for token classification (BIO-tagged NER) on the
silver-standard labels produced by `src/ner/lexicon_bootstrap.py`.

Read the warning in `lexicon_bootstrap.py` first: this trains a real
transformer, but on distant-supervision labels, so evaluate honestly and
say so in your model card (e.g. "trained on lexicon-bootstrapped silver
labels; entity coverage ~2% of tokens; recommend gold-standard review
before production use").

Run on Colab/Kaggle with a GPU (see train_transformer.py in
src/classification/ for the same setup instructions).

    python -m src.ner.train_transformer
"""

from __future__ import annotations

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.ner.lexicon_bootstrap import annotate_dataframe

MODEL_CKPT = "xlm-roberta-base"
OUTPUT_DIR = "models/xlmr_ner"
LABEL_LIST = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def parse_annotated_sequence(seq: str) -> tuple[list[str], list[str]]:
    tokens, tags = [], []
    for pair in seq.split():
        if "/" not in pair:
            continue
        tok, tag = pair.rsplit("/", 1)
        tokens.append(tok)
        tags.append(tag if tag in LABEL2ID else "O")
    return tokens, tags


def build_dataset() -> Dataset:
    df = pd.read_excel("data/raw/punjabi_shahmukhi_master_dataset.xlsx")
    annotated = annotate_dataframe(df)
    records = []
    for seq in annotated["Annotated_Sequence"]:
        tokens, tags = parse_annotated_sequence(seq)
        if tokens:
            records.append({"tokens": tokens, "ner_tags": [LABEL2ID[t] for t in tags]})
    return Dataset.from_list(records)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = build_dataset().train_test_split(test_size=0.1, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

    def tokenize_and_align(batch):
        tokenized = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True, max_length=256)
        all_labels = []
        for i, labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    label_ids.append(labels[word_id])
                else:
                    # sub-word continuation -> -100 so loss ignores it
                    label_ids.append(-100)
                prev_word_id = word_id
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    tokenized_ds = dataset.map(tokenize_and_align, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CKPT, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID
    )

    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=2)
        true_preds, true_labels = [], []
        for pred_seq, label_seq in zip(preds, labels):
            p_seq, l_seq = [], []
            for p, l in zip(pred_seq, label_seq):
                if l != -100:
                    p_seq.append(ID2LABEL[p])
                    l_seq.append(ID2LABEL[l])
            true_preds.append(p_seq)
            true_labels.append(l_seq)
        results = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    final_metrics = trainer.evaluate()
    print("Final validation metrics (real, from this run, on silver labels):", final_metrics)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned NER model -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
