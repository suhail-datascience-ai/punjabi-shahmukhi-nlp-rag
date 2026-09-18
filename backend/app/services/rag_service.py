from __future__ import annotations

import os

from src.rag.retriever import TfidfRetriever
from backend.app import config


class RAGService:
    def __init__(self) -> None:
        if config.USE_EMBEDDING_RETRIEVER:
            from src.rag.retriever import EmbeddingRetriever

            # NOTE: expects you've already built + saved a dense index via
            # `python -m src.rag.build_dense_index` (see retriever.py docstring).
            raise NotImplementedError(
                "Dense retriever is scaffolded in src/rag/retriever.py but needs "
                "a build_dense_index.py entrypoint written once you have internet "
                "access to download the embedding model. Falling back to TF-IDF "
                "until then is safer than silently failing."
            )
        else:
            if not (config.RAG_INDEX_DIR / "tfidf_retriever.pkl").exists():
                raise FileNotFoundError(
                    f"No RAG index found at {config.RAG_INDEX_DIR}. "
                    "Run `python -m src.rag.retriever` first."
                )
            self.retriever = TfidfRetriever.load(config.RAG_INDEX_DIR)
            self.backend_name = "tfidf-retrieval (offline)"

    def query(self, query: str, top_k: int = 3):
        results = self.retriever.search(query, k=top_k)
        generated_answer = None
        if config.ENABLE_GENERATIVE_ANSWER and results:
            generated_answer = self._generate_answer(query, results)
        return results, generated_answer

    def _generate_answer(self, query: str, results) -> str | None:
        """Optional: synthesize a natural-language answer from retrieved
        passages using the Anthropic API. Only runs if ANTHROPIC_API_KEY is
        set — otherwise the API just returns the raw retrieved passages,
        which is a perfectly legitimate RAG demo on its own.
        """
        try:
            import anthropic
        except ImportError:
            return None

        context = "\n\n".join(f"[{i+1}] {r.text}" for i, r in enumerate(results))
        prompt = (
            "You are a Punjabi (Shahmukhi script) question-answering assistant. "
            "Answer the user's question in Punjabi, using ONLY the passages below. "
            "If the passages don't contain the answer, say so honestly.\n\n"
            f"Passages:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text")
