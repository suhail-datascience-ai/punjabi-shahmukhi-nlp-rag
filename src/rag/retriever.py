"""
retriever.py
------------
Retrieval component of the RAG (Retrieval-Augmented Generation) pipeline.

Two retriever backends are provided behind one interface:

  1. `TfidfRetriever`  — classic sparse retrieval (cosine similarity over
     TF-IDF vectors). Fully offline, no GPU, no model download. This is
     what actually runs in this repository right now and what the FastAPI
     backend serves by default, so the demo works out of the box.

  2. `EmbeddingRetriever` — dense retrieval using a multilingual sentence
     embedding model (`sentence-transformers`) + FAISS for approximate
     nearest-neighbour search. This is the production-quality upgrade:
     better recall on paraphrased/semantic queries where no keyword
     overlaps with the source document. It requires downloading a model
     from the Hugging Face Hub, so it needs internet access and (for
     larger corpora) a GPU is nice-to-have but not required.

Swapping backends is a one-line change in `backend/app/main.py`
(`RETRIEVER_BACKEND` env var) — nothing else in the API layer changes,
because both classes implement the same `.search(query, k)` method.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.normalize import normalize_shahmukhi

INDEX_DIR = Path("models/rag_index")


@dataclass
class RetrievedDoc:
    doc_id: int
    score: float
    text: str
    summary: str | None = None


class TfidfRetriever:
    """Sparse cosine-similarity retriever. Offline, deterministic, fast."""

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_matrix = None
        self.documents: list[str] = []
        self.summaries: list[str | None] = []

    def fit(self, documents: list[str], summaries: list[str | None] | None = None) -> "TfidfRetriever":
        self.documents = [normalize_shahmukhi(d) for d in documents]
        self.summaries = summaries or [None] * len(documents)
        self.vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.doc_matrix = self.vectorizer.fit_transform(self.documents)
        return self

    def search(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        if self.vectorizer is None:
            raise RuntimeError("Retriever not fitted/loaded. Call fit() or load().")
        q = normalize_shahmukhi(query)
        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        top_idx = np.argsort(-sims)[:k]
        return [
            RetrievedDoc(doc_id=int(i), score=float(sims[i]), text=self.documents[i], summary=self.summaries[i])
            for i in top_idx
        ]

    def save(self, path: Path = INDEX_DIR) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "tfidf_retriever.pkl", "wb") as f:
            pickle.dump(
                {"vectorizer": self.vectorizer, "doc_matrix": self.doc_matrix, "documents": self.documents, "summaries": self.summaries},
                f,
            )

    @classmethod
    def load(cls, path: Path = INDEX_DIR) -> "TfidfRetriever":
        with open(path / "tfidf_retriever.pkl", "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj.vectorizer = state["vectorizer"]
        obj.doc_matrix = state["doc_matrix"]
        obj.documents = state["documents"]
        obj.summaries = state["summaries"]
        return obj


class EmbeddingRetriever:
    """Dense retriever using sentence-transformers + FAISS.

    NOT active by default in this repo because it requires downloading
    `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (or a
    Punjabi/Urdu-tuned equivalent) from the Hugging Face Hub. Wire it in
    once you have internet access in your training environment:

        pip install sentence-transformers faiss-cpu
        python -m src.rag.build_dense_index
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._model = None
        self._index = None
        self.documents: list[str] = []
        self.summaries: list[str | None] = []

    def _lazy_load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # local import: optional dep

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, documents: list[str], summaries: list[str | None] | None = None) -> "EmbeddingRetriever":
        import faiss  # local import: optional dep

        model = self._lazy_load_model()
        self.documents = [normalize_shahmukhi(d) for d in documents]
        self.summaries = summaries or [None] * len(documents)
        embeddings = model.encode(self.documents, normalize_embeddings=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(np.asarray(embeddings, dtype="float32"))
        return self

    def search(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        model = self._lazy_load_model()
        q = normalize_shahmukhi(query)
        q_emb = model.encode([q], normalize_embeddings=True)
        scores, idx = self._index.search(np.asarray(q_emb, dtype="float32"), k)
        return [
            RetrievedDoc(doc_id=int(i), score=float(s), text=self.documents[i], summary=self.summaries[i])
            for s, i in zip(scores[0], idx[0])
            if i != -1
        ]


def build_index_from_wikipedia_dataset(path: str = "data/raw/master punjabi dataset.xlsx") -> TfidfRetriever:
    df = pd.read_excel(path)
    df = df.dropna(subset=["Article"]).reset_index(drop=True)
    retriever = TfidfRetriever().fit(df["Article"].tolist(), df["Summary"].tolist())
    retriever.save()
    return retriever


if __name__ == "__main__":
    retriever = build_index_from_wikipedia_dataset()
    print(f"Indexed {len(retriever.documents)} documents into {INDEX_DIR}")
