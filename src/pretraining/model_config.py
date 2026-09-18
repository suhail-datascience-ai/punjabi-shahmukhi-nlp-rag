"""
model_config.py
-----------------
Three staged encoder configs, smallest first, so you get a working
end-to-end pipeline in minutes before committing GPU-days to the full-size
run. Pick one with `--config tiny|small|compact` in train_mlm.py.

Staged plan
===========
  TIER 0 — "tiny": ~6-8M params, tiny vocab, short sequences. Purpose:
      prove the whole pipeline (tokenizer -> packing -> training ->
      checkpoint -> reload) actually works, on whatever corpus you already
      have (even your existing ~150K-token scraped set). Should finish a
      few thousand steps in well under an hour on an RTX 3060. This is a
      pipeline test, not a usable model — don't publish it.

  TIER 1 — "small": ~12-15M params, full 32K vocab, seq len 128. Purpose:
      your first REAL, publishable checkpoint, trained on the full
      pnb.wikipedia.org dump (low tens of millions of tokens — see
      build_pretraining_corpus.py). Budget roughly 4-10 hours on a 3060
      depending on how many epochs you run. This is a legitimate "v0.1"
      release: honestly small-corpus, honestly says so, still real.

  TIER 2 — "compact": ~25M params, matches the Sindhi Avey-B scale.
      Purpose: your "v1.0" release, once you've layered in more corpus
      (OSCAR/CC-100 filtered + Wikipedia). This is a multi-day run — start
      it, let it checkpoint, and keep working on other parts of the
      project (frontend polish, writing the model card, etc.) while it
      trains in the background (see docs/PHASE2_TERMINAL_WORKFLOW.md).

Move up a tier only after the previous tier's loss curve has actually
flattened out — a bigger model on the same tiny corpus just overfits
faster, it doesn't make the result better.
"""

from transformers import BertConfig

TINY_ENCODER_CONFIG = BertConfig(
    vocab_size=8_000,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=512,
    max_position_embeddings=64,
    type_vocab_size=1,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

SMALL_ENCODER_CONFIG = BertConfig(
    vocab_size=32_000,
    hidden_size=192,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=768,
    max_position_embeddings=128,
    type_vocab_size=1,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

# Tuned to land close to ~25M total parameters at vocab_size=32000 —
# comparable to the Sindhi Avey-B release (24.9M params, 32K vocab).
COMPACT_ENCODER_CONFIG = BertConfig(
    vocab_size=32_000,
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=4,
    intermediate_size=1024,
    max_position_embeddings=256,
    type_vocab_size=1,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

CONFIGS = {
    "tiny": TINY_ENCODER_CONFIG,
    "small": SMALL_ENCODER_CONFIG,
    "compact": COMPACT_ENCODER_CONFIG,
}
