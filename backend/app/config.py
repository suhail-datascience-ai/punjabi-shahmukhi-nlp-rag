"""
Central configuration for which model backend the API serves.

Everything defaults to the artifacts that ship in this repo and run fully
offline (TF-IDF classifier + lexicon NER + TF-IDF RAG retriever), so
`uvicorn app.main:app` works immediately after `pip install -r
requirements.txt` with no GPU and no model download.

Once you've run the heavier transformer training scripts (see
src/classification/train_transformer.py and src/ner/train_transformer.py)
on Colab/Kaggle and copied the resulting folders back into `models/`,
flip the two USE_* flags below (or set the equivalent environment
variables) to switch the API over to the transformer models without
touching any route code.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

# --- Classification backend -------------------------------------------------
USE_TRANSFORMER_CLASSIFIER = os.getenv("USE_TRANSFORMER_CLASSIFIER", "false").lower() == "true"
TFIDF_CLASSIFIER_PATH = MODELS_DIR / "tfidf_logreg_classifier.joblib"
TRANSFORMER_CLASSIFIER_PATH = MODELS_DIR / "xlmr_classifier"

# --- NER backend -------------------------------------------------------------
USE_TRANSFORMER_NER = os.getenv("USE_TRANSFORMER_NER", "false").lower() == "true"
TRANSFORMER_NER_PATH = MODELS_DIR / "xlmr_ner"

# --- RAG backend --------------------------------------------------------------
USE_EMBEDDING_RETRIEVER = os.getenv("USE_EMBEDDING_RETRIEVER", "false").lower() == "true"
RAG_INDEX_DIR = MODELS_DIR / "rag_index"

# --- Optional: call an LLM to generate a natural-language answer from the
# retrieved passages, instead of just returning the raw passages. Off by
# default (keeps the demo fully offline / free to run). If you set
# ANTHROPIC_API_KEY, `backend/app/rag_service.py` will use it.
ENABLE_GENERATIVE_ANSWER = os.getenv("ANTHROPIC_API_KEY") is not None
