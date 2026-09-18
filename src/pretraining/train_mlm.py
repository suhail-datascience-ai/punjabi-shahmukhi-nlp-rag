"""
train_mlm.py
-------------
Step 3 of Phase 2: pretrain the compact encoder from scratch with masked
language modeling on your raw corpus.

Sized for a single RTX 3060 (12GB VRAM) / 16GB system RAM box:
  * Streams the dataset from disk (HF `datasets` memory-maps Arrow files)
    instead of loading everything into RAM — important given 16GB system RAM.
  * fp16 mixed precision + gradient accumulation to fit comfortably in 12GB.
  * Checkpoints every N steps so a multi-day run survives a reboot/crash —
    resume with `--resume_from_checkpoint models/punjabi_shahmukhi_encoder/checkpoint-XXXX`.

Realistic expectations on this hardware: with a ~25M param model, sequence
length 128, and a corpus in the tens of millions of tokens, budget **low
single-digit days** for a reasonable number of epochs, not hours. Start a
short run first (`--max_steps 2000`) to sanity-check throughput and loss
curves before committing to a multi-day run.

Usage (staged — run tiny first, always):
    # TIER 0 - pipeline smoke test, minutes, on whatever text you already have
    python -m src.pretraining.train_tokenizer --config tiny --corpus data/pretraining_raw/quick_corpus.txt
    python -m src.pretraining.train_mlm --config tiny --corpus data/pretraining_raw/quick_corpus.txt --max_steps 2000

    # TIER 1 - first real checkpoint, hours, on the full Wikipedia dump
    python -m src.pretraining.train_tokenizer --config small
    python -m src.pretraining.train_mlm --config small

    # TIER 2 - Sindhi-scale target, multi-day, once you've added more corpus
    python -m src.pretraining.train_tokenizer --config compact
    python -m src.pretraining.train_mlm --config compact

    # resuming any of the above after an interruption:
    python -m src.pretraining.train_mlm --config small --resume_from_checkpoint models/punjabi_shahmukhi_encoder_small/checkpoint-5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import (
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.pretraining.model_config import CONFIGS

DEFAULT_CORPUS_PATH = "data/pretraining_raw/corpus.txt"
MAX_SEQ_LEN_BY_TIER = {"tiny": 64, "small": 128, "compact": 256}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS), default="tiny")
    parser.add_argument("--corpus", type=str, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--max_steps", type=int, default=-1, help="-1 = full run based on num_train_epochs")
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    tokenizer_dir = f"models/punjabi_shahmukhi_tokenizer_{args.config}"
    output_dir = f"models/punjabi_shahmukhi_encoder_{args.config}"
    max_seq_len = MAX_SEQ_LEN_BY_TIER[args.config]

    if not Path(tokenizer_dir, "tokenizer.json").exists():
        raise FileNotFoundError(f"Run: python -m src.pretraining.train_tokenizer --config {args.config}  first.")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    print(f"[{args.config}] Loading + tokenizing corpus (memory-mapped, safe for 16GB RAM)...")
    raw_ds = load_dataset("text", data_files=args.corpus, split="train")

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )

    tokenized_ds = raw_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        num_proc=4,  # matches the 7700X's core count reasonably; lower if RAM-constrained
    )

    model = BertForMaskedLM(CONFIGS[args.config])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.config}] Model parameter count: {n_params:,}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=True,  # RTX 3060 supports fp16 tensor cores well
        learning_rate=5e-4,
        warmup_ratio=0.02,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=3,
        logging_steps=100,
        report_to=[],
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[{args.config}] Saved pretrained encoder -> {output_dir}")
    print("Final training loss is in the Trainer log history above — record it honestly in your model card.")


if __name__ == "__main__":
    main()
