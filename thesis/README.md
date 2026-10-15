# Thesis workspace: Silver vs Gold Supervision for Shahmukhi NER

Working title: *Silver vs Gold Supervision for Shahmukhi NER: A Multi-Domain Benchmark and Compact-Encoder Study*

## Research questions

1. How good a proxy are lexicon-bootstrapped silver labels for gold labels, and what does mixing them buy? (silver-only, gold-only, silver+gold, learning curves)
2. How noisy is the existing Shahmukhi NER data, and what changes when it is cleaned?
3. How does a from-scratch Punjabi encoder scale with data and parameters against XLM-R?
4. How much does cross-lingual data (Urdu NER, transliterated Gurmukhi) help?

## Evaluation protocol

- Entity-level seqeval F1, at least 5 seeds per configuration, bootstrap confidence intervals, paired significance tests.
- Frozen gold test set, never used for training or model selection (hash in `data/gold/MANIFEST.json`).
- Every run writes a JSON result into `data/processed/` with config, seed, data hashes and metrics.

## Status

- [x] Step 1: thesis workspace, annotation guideline v0.1, agreement tooling (`src/evaluation/agreement.py`)
- [ ] Step 2: external data (existing Shahmukhi NER, MK-PUCIT) and NER v2 baseline
- [ ] Step 3: pilot annotation (300 sentences, 2 annotators) and agreement report
- [ ] Step 4: corpus expansion (scraper, dedup, script filter, licence log)
- [ ] Step 5: experiment runner and multi-seed grid
- [ ] Step 6: scaling study and writing
