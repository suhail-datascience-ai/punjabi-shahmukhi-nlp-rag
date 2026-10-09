# Model Card — Punjabi Shahmukhi News Intelligence

## Overview

This project ships two working model tiers per task, so the repo is useful the moment someone clones it (no GPU, no internet, no Hugging Face download) *and* has a clear upgrade path to transformer quality once you have GPU access.

| Task | Tier 1 (ships in repo, offline) | Tier 2 (train yourself on Colab/Kaggle GPU) |
|---|---|---|
| Classification | TF-IDF (1-2 grams) + Logistic Regression | `xlm-roberta-base` fine-tuned, weighted loss |
| NER | Gazetteer/lexicon distant-supervision BIO tagger | `xlm-roberta-base` token-classification head, trained on the silver labels (or better: silver+gold mix) |
| RAG retrieval | TF-IDF cosine similarity | `paraphrase-multilingual-MiniLM-L12-v2` embeddings + FAISS |

Both tiers implement the same interface (`ClassifierService`, `NERService`, `RAGService` in `backend/app/services/`), so switching is a one-line config change (`USE_TRANSFORMER_CLASSIFIER=true`, etc.) — see `backend/app/config.py`.

## Tier 1 Pretrained Encoder — Real Head-to-Head vs XLM-R (computed on this exact run)

Pretrained a from-scratch 8M-parameter encoder (`--config small`) on the full pnb.wikipedia.org dump (76,571 articles, ~32.5M tokens — see Data Card for how this compares honestly to the Sindhi release's 327M tokens). Fine-tuned the same classification task on both this encoder and `xlm-roberta-base`, using `src/pretraining/compare_encoders.py`:

| | Your encoder | XLM-R-base |
|---|---|---|
| Params | **7,986,629** | 278,047,493 (35x bigger) |
| Accuracy | 87.6% | 90.3% |
| Macro F1 | 0.726 | 0.872 |
| Inference speed | 4,170 samples/sec | 184 samples/sec (23x slower) |

**Honest read:** XLM-R-base wins clearly on quality, especially macro F1 — meaning the compact encoder struggles more on the minority classes (politics, entertainment) than the pretrained multilingual model does. This is NOT a "small model matches a big one" result like the Sindhi release reported. The real, legitimate result here is a size/speed trade-off: 97% fewer parameters and ~23x faster inference, at a real cost in accuracy (-2.7 points) and macro F1 (-0.146). State it this way in any public write-up rather than reframing it as a win it isn't.

Training details: 20 epochs over the 32.5M-token corpus, loss plateaued around 3.20-3.24 after ~epoch 15 (further epochs on this same corpus are unlikely to help — more raw data or a larger config, per the Tier 2 plan below, are the next real levers).

## Held-Out Test Set Results (frozen splits)

The original project had a train/val/test split but only train and val were saved. The test set (145 rows) was recovered as the full dataset minus train minus val, checked for overlap, duplicates and label conflicts (none found), and frozen with file hashes in `data/splits/MANIFEST.json`. It was never used for training or model selection. Produced by `python -m src.evaluation.classification_test_eval`.

| Model | Accuracy (95% CI) | Macro F1 (95% CI) |
|---|---|---|
| TF-IDF + Logistic Regression | 0.876 (0.814-0.931) | 0.831 (0.718-0.913) |
| XLM-R-base fine-tuned | 0.910 (0.862-0.952) | 0.909 (0.836-0.957) |

Confidence intervals are bootstrap 95% intervals (1,000 resamples). An exact McNemar test on the 13 documents where the models disagree (4 only TF-IDF correct, 9 only XLM-R correct) gives p = 0.267, so the accuracy difference is not statistically significant on this test set. The macro F1 gap comes mostly from the two smallest classes (politics: 9 test documents, entertainment: 7), where about 4 documents separate the models. Read this as "XLM-R point estimates are higher, not shown to be significantly better"; cross-validation on the full dataset is the next step to firm this up.

## Reported metrics — Tier 1 (computed on this exact dataset; reproducible via the commands shown)

### Classification (TF-IDF + Logistic Regression)
`python -m src.classification.train_baseline`

| Metric | Value |
|---|---|
| Validation accuracy | **91.0%** |
| Macro F1 | **0.893** |
| Weighted F1 | **0.910** |

Per-class F1: entertainment 0.93, geography 0.88, history 0.92, politics 0.80, sports 0.93 — politics lags the others, consistent with it being one of the two under-represented classes (see Data Card). Full confusion matrix in `data/processed/classification_baseline_metrics.json`.

### NER (lexicon bootstrap)
`python -m src.ner.lexicon_bootstrap`

| Metric | Value |
|---|---|
| Token entity coverage | **2.07%** (3,136 / 151,665 tokens tagged non-O) |
| Entities by type | LOC 1,399 · ORG 872 · PER 514 · MISC 351 |

**This is a coverage statistic, not an accuracy statistic** — there is no accuracy number here because there's no independent gold label to score against yet. Do not present this as "F1" anywhere. See the "Path to a defensible NER number" section below.

### RAG retrieval (TF-IDF)
`python -m src.evaluation.evaluate_rag`

Task: given a document's own Summary as the query, can the retriever find that document among the top-k of all 296?

| Metric | Value |
|---|---|
| Recall@1 | **57.1%** |
| Recall@3 | **77.7%** |
| Recall@5 | **83.1%** |
| MRR | **0.675** |

These are real numbers from an actual eval run on the shipped corpus — not the flat, suspiciously-identical placeholder numbers (`92.45 / 92.45 / 92.45`) that appeared across three different "pipelines" in the original notebook. If you rerun the eval after changing anything, these numbers will move — that's expected and correct.

## Path to a defensible NER number

Publishing "NER F1: X%" without a gold-standard test set is the single biggest credibility risk in this project if you're showcasing it as professional work. In order of cost:

1. **Cheapest — be transparent instead of inventing a score.** State clearly: "silver-standard, lexicon-bootstrapped NER; entity coverage 2.07% of tokens; no gold-standard evaluation yet." This is honest and still shows real engineering.
2. **Medium — spot-check + report agreement.** Manually review ~150-300 auto-tagged sentences, correct them, and report precision/recall of the auto-tagger against your corrections. A few hours of work, gives you a real, quotable number.
3. **Best — inter-annotator agreement + fine-tune on the corrected set.** Have a second person independently label the same sample, compute Cohen's Kappa (scaffolded in the original notebook, cell 19 — refactor it into `src/evaluation/`), then fine-tune the transformer NER model on silver+gold combined. This is genuinely publishable.

## Reproducing Tier 2 (transformer) training

Both `src/classification/train_transformer.py` and `src/ner/train_transformer.py` are complete, runnable scripts. They are not run automatically in this repo because:
- They need `xlm-roberta-base` downloaded from the Hugging Face Hub (~1.1GB) — this sandboxed build environment used to assemble this repo has no route to `huggingface.co`.
- Fine-tuning is meaningfully faster on a GPU (a free Colab T4 handles this dataset size in a few minutes per epoch).

Steps:
1. Open a Colab/Kaggle notebook with a GPU runtime.
2. `git clone` your repo, `cd` into it, `pip install -r requirements.txt`.
3. `python -m src.classification.train_transformer`
4. `python -m src.ner.train_transformer`
5. Both scripts print real validation metrics from `Trainer.evaluate()` at the end — copy those numbers (not placeholders) into this file and into your README/LinkedIn post.
6. Copy `models/xlmr_classifier/` and `models/xlmr_ner/` back into this repo (or push them straight to the Hugging Face Hub with `huggingface-cli upload <your-username>/punjabi-shahmukhi-classifier models/xlmr_classifier`), then set `USE_TRANSFORMER_CLASSIFIER=true` / `USE_TRANSFORMER_NER=true` as environment variables before starting the backend.

## Phase 2 (stretch goal): pretraining a compact Punjabi Shahmukhi encoder

The Sindhi project this repo is modeled after (`Avey-B Sindhi 25M`) is a **different scale of project** from what's in this repo: it's a from-scratch masked-language-model pretraining run on 327M+ tokens of raw (unlabeled) Sindhi text, producing a general-purpose encoder others can fine-tune. What's in this repo so far is task-specific fine-tuning/classical-ML on ~1,450 labeled rows — a smaller, faster, and much cheaper undertaking, and a reasonable first release on its own.

If you want to eventually do the Sindhi-style release for Punjabi Shahmukhi, that's a separate, larger project:

1. **Corpus collection** (the hard part): aim for tens-to-hundreds of millions of tokens of raw Shahmukhi text. Candidate sources: full `pnb.wikipedia.org` dump (not just 5 categories), the HPLT project's Punjabi (Shahmukhi) split if/when available, Common Crawl filtered for Shahmukhi script, OSCAR corpus, and any Shahmukhi-script books/news sites you have rights to use.
2. **Tokenizer.** Train a fresh WordPiece or BPE vocabulary (e.g. 32K, matching the Sindhi release) on your raw corpus — don't reuse XLM-R's vocabulary, or you lose the "compact, language-specific" advantage that makes this kind of release notable.
3. **Architecture.** A small BERT-style encoder (12-24M params, similar to the Sindhi release) trained with masked-language-modeling from scratch. This needs a real GPU budget (expect this to run on a rented GPU or Colab Pro, likely days not minutes) — a materially bigger commitment than anything else in this README.
4. **Evaluation & release.** Reuse this repo's classification/NER heads to fine-tune on top of your new encoder and directly compare against `xlm-roberta-base` — that comparison ("our compact model matches/exceeds a much larger multilingual model on Punjabi-specific tasks") is exactly the headline the Sindhi release used, and it only works if you actually run and report both.
5. Release the checkpoint, tokenizer, training config, and eval numbers on the Hugging Face Hub with a model card in this same honest style.

This is worth scoping as its own follow-up project rather than folding into this repo's v1.0 release.
