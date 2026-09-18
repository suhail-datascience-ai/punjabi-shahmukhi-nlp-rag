# Data Card — Punjabi (Shahmukhi) News & Wikipedia Corpus

## Summary

| Split / file | Rows | Purpose |
|---|---|---|
| `master_train.xlsx` | 1,159 | Classification train split |
| `master_val.xlsx` | 145 | Classification validation split |
| `punjabi_shahmukhi_master_dataset.xlsx` | 1,449 | Full classification set (pre-split), also the source for NER bootstrap tagging |
| `master punjabi dataset.xlsx` | 296 | Article/Summary pairs, used as the RAG knowledge base + retrieval eval set |
| `annotated sequence dataset.xlsx` | 1,449 | Same as master, plus a pre-computed `Annotated_Sequence` BIO column (regenerable via `src/ner/lexicon_bootstrap.py`) |

## Provenance

Collected via a custom scraper (`src/data_collection/`) targeting `pnb.wikipedia.org` (Western Punjabi / Shahmukhi Wikipedia), organized into 5 topical hubs: politics, sports, geography, history, entertainment. See the original notebook cells for the exact category seed URLs.

**License/rights note:** Wikipedia content is available under CC BY-SA 4.0 / GFDL. If you publish this dataset publicly (e.g. on the Hugging Face Hub), include that attribution and license, and check `pnb.wikipedia.org`'s current terms before redistributing scraped derivatives at scale.

## Label distribution (classification)

Combined imbalance across train+val:

| Label | Count | % |
|---|---|---|
| history | 654 | 45.1% |
| sports | 388 | 26.8% |
| geography | 247 | 17.0% |
| politics | 90 | 6.2% |
| entertainment | 70 | 4.8% |

This is a real class imbalance — `politics` and `entertainment` are under-represented. Both the baseline (`class_weight="balanced"`) and the transformer script (weighted cross-entropy) correct for this at training time, but it also means:
- Precision/recall on the minority classes is noisier (small support = wide confidence intervals).
- If you extend this project, prioritize collecting more `politics`/`entertainment` documents over blindly scraping more `history`.

## NER labels

The `Annotated_Sequence` column is **silver-standard**: produced by matching a ~65-entry gazetteer of people/places/organizations/misc terms against the tokenized text (see `src/ner/lexicon_bootstrap.py`), not by human annotators. On this dataset, only **2.07% of tokens** receive a non-`O` tag — meaning recall on entities outside the gazetteer is unknown and likely materially higher-value gains for your project come from *expanding the gazetteer* or doing a *manual gold-standard pass on a sample* before quoting NER F1 publicly. See `docs/MODEL_CARD.md` for how to state this honestly.

## RAG knowledge base

296 (Article, Summary) pairs scraped from Shahmukhi Wikipedia. Articles average roughly a paragraph in length — small by production RAG standards, so retrieval quality partly reflects the collection's size, not just the retriever's quality. Expanding this corpus (more articles, longer documents, chunking long articles into passages) is the highest-leverage next step for the RAG component specifically.

## Known limitations (be upfront about these in any public write-up)

1. **Small scale.** ~1,450 classification rows and 296 RAG documents is enough for a credible portfolio project and a real baseline, but well below what you'd want for a production system or a claim of "state of the art."
2. **Single source domain.** Everything comes from Wikipedia, so the models will underperform on informal or social-media Punjabi text.
3. **Distant-supervision NER labels** (see above) — disclose this, don't claim gold-standard performance.
4. **No de-duplication audit performed here** — if you scrape more data later, check for near-duplicate articles before re-splitting train/val/test, or you'll leak information across splits.
