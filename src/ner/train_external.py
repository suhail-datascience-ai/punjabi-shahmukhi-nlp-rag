"""
train_external.py (NER v2)
--------------------------
Same recipe as src/ner/train_transformer.py (v1), plus an external, labelled
NER file added to the TRAINING data.

Honest evaluation:
  * silver data  = identical split to v1 (90/10, seed 42)
  * external file = split 70/15/15 (train/val/test, seed 42)
  * final report = v1 vs v2 on the held-out EXTERNAL test split, which neither
    model trained on. (F1 on lexicon-bootstrapped silver labels mostly measures
    how well a model reproduces the lexicon, so it is not a real accuracy number.)

External file format (CoNLL): one "token tag" per line, blank line between
sentences. Tags: O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC.
PERSON / LOCATION / GPE / ORGANIZATION are mapped to PER / LOC / LOC / ORG;
any other tag becomes O and is printed so nothing is dropped silently.

    python -m src.ner.train_external --external data/external/ner.conll
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.ner.train_transformer import (
    ID2LABEL,
    LABEL2ID,
    MODEL_CKPT,
    build_dataset,
)

V1_DIR = "models/xlmr_ner"
OUTPUT_DIR = "models/xlmr_ner_v2"
RESULTS_JSON = "data/processed/ner_v2_results.json"

CORE = {
    "PER": "PER", "PERSON": "PER",
    "LOC": "LOC", "LOCATION": "LOC", "GPE": "LOC",
    "ORG": "ORG", "ORGANIZATION": "ORG",
    "MISC": "MISC",
}


def norm_tag(tag: str, dropped: dict) -> str:
    if tag == "O":
        return "O"
    prefix, _, core = tag.partition("-")
    if prefix in ("B", "I") and core in CORE:
        return f"{prefix}-{CORE[core]}"
    dropped[tag] = dropped.get(tag, 0) + 1
    return "O"


def load_conll(path: str) -> list[tuple[list[str], list[str]]]:
    sentences, toks, tags = [], [], []
    dropped: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if toks:
                    sentences.append((toks, tags))
                toks, tags = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            toks.append(parts[0])
            tags.append(norm_tag(parts[-1], dropped))
    if toks:
        sentences.append((toks, tags))
    if dropped:
        print("Unmapped external tags turned into O:", dropped)
    return sentences


def to_dataset(sentences) -> Dataset:
    return Dataset.from_list(
        [{"tokens": t, "ner_tags": [LABEL2ID[x] for x in g]} for t, g in sentences]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True, help="CoNLL-format NER file")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

    # Silver data, same split as v1
    silver = build_dataset().train_test_split(test_size=0.1, seed=42)

    # External data, de-duplicated against the silver sentences to avoid leakage
    seen = {tuple(r["tokens"]) for split in silver.values() for r in split}
    ext_all = load_conll(a.external)
    ext = [s for s in ext_all if tuple(s[0]) not in seen]
    print(f"External sentences: {len(ext_all)} read, {len(ext)} kept after dedupe")
    if len(ext) < 20:
        raise SystemExit("Too few external sentences after dedupe; check the file/format.")

    ext_split = to_dataset(ext).train_test_split(test_size=0.30, seed=42)
    ext_vt = ext_split["test"].train_test_split(test_size=0.5, seed=42)
    ext_train, ext_val, ext_test = ext_split["train"], ext_vt["train"], ext_vt["test"]
    print(f"External split: train={len(ext_train)} val={len(ext_val)} test={len(ext_test)}")

    train_raw = concatenate_datasets([silver["train"], ext_train]).shuffle(seed=42)
    val_raw = concatenate_datasets([silver["test"], ext_val])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

    def tokenize_and_align(batch):
        tokenized = tokenizer(
            batch["tokens"], truncation=True, is_split_into_words=True, max_length=256
        )
        all_labels = []
        for i, labels in enumerate(batch["ner_tags"]):
            prev = None
            ids = []
            for word_id in tokenized.word_ids(batch_index=i):
                if word_id is None or word_id == prev:
                    ids.append(-100)
                else:
                    ids.append(labels[word_id])
                prev = word_id
            all_labels.append(ids)
        tokenized["labels"] = all_labels
        return tokenized

    train_ds = train_raw.map(tokenize_and_align, batched=True)
    val_ds = val_raw.map(tokenize_and_align, batched=True)
    test_ds = ext_test.map(tokenize_and_align, batched=True)

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
        r = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": r["overall_precision"],
            "recall": r["overall_recall"],
            "f1": r["overall_f1"],
            "accuracy": r["overall_accuracy"],
        }

    collator = DataCollatorForTokenClassification(tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CKPT,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=8,
        num_train_epochs=a.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="f1",
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Leftover checkpoints are ~3 GB each and the best model is already saved above
    for p in Path(OUTPUT_DIR).glob("checkpoint-*"):
        shutil.rmtree(p, ignore_errors=True)
    print(f"Saved NER v2 -> {OUTPUT_DIR}")

    del trainer, model
    torch.cuda.empty_cache()

    # v1 vs v2 on the held-out external test split
    results = {"external_test_sentences": len(ext_test)}
    for name, path in (("v1", V1_DIR), ("v2", OUTPUT_DIR)):
        m = AutoModelForTokenClassification.from_pretrained(path)
        t = Trainer(
            model=m,
            args=TrainingArguments(
                output_dir="/tmp/ner_eval", per_device_eval_batch_size=8, report_to=[]
            ),
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        r = t.evaluate(test_ds)
        results[name] = {
            k: float(v)
            for k, v in r.items()
            if k in ("eval_precision", "eval_recall", "eval_f1", "eval_accuracy")
        }
        print(name, results[name])
        del t, m
        torch.cuda.empty_cache()

    Path(RESULTS_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULTS_JSON).write_text(json.dumps(results, indent=2))
    print(f"Wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
