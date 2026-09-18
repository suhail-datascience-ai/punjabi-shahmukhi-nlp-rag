"""
train_tokenizer.py
--------------------
Step 2 of Phase 2: train a fresh 32K WordPiece vocabulary on your raw
Shahmukhi corpus, instead of reusing XLM-R's tokenizer. This is what makes
a "compact, language-specific" model actually compact — XLM-R's vocabulary
is spread across 100 languages, so most of its 250K entries are wasted for
a single-language model, and its sub-word fertility on Punjabi is worse
than a tokenizer trained directly on Punjabi text.

Usage:
    python -m src.pretraining.train_tokenizer

Needs `data/pretraining_raw/corpus.txt` (one document per line) produced by
build_pretraining_corpus.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

from src.pretraining.model_config import CONFIGS

CORPUS_PATH = Path("data/pretraining_raw/corpus.txt")
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS), default="tiny",
                         help="tiny = pipeline smoke test, small = first real release, compact = Sindhi-scale target")
    parser.add_argument("--corpus", type=str, default=str(CORPUS_PATH),
                         help="Defaults to the full pretraining corpus; for a tiny/fast sanity run point this at "
                              "a smaller file, e.g. your existing data/processed export.")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    vocab_size = CONFIGS[args.config].vocab_size
    out_dir = Path(f"models/punjabi_shahmukhi_tokenizer_{args.config}")

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found. Run build_pretraining_corpus.py first, "
            "or pass --corpus pointing at whatever text file you want to use for a quick test."
        )

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    # NFKC normalization matches src/preprocessing/normalize.py so downstream
    # fine-tuning code and the pretraining tokenizer see the same text.
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        continuing_subword_prefix="##",
    )

    tokenizer.train([str(corpus_path)], trainer)

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / "tokenizer.json"))

    # Also save in the format transformers.AutoTokenizer expects
    from transformers import PreTrainedTokenizerFast

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    fast_tokenizer.save_pretrained(str(out_dir))

    print(f"[{args.config}] Trained {vocab_size}-token WordPiece vocab -> {out_dir}")
    sample = "پنجاب دی راجدھانی لاهور اے۔"
    print("Sanity check:", fast_tokenizer.tokenize(sample))


if __name__ == "__main__":
    main()
