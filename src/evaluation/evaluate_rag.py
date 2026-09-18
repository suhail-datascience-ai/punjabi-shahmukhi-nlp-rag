"""
evaluate_rag.py
----------------
Genuine retrieval evaluation for the RAG pipeline.

Task setup: for each (Article, Summary) pair, the Summary is treated as a
query and the retriever must find its own source Article among the top-k
results from the full document collection. This is the standard "does the
retriever find the right needle in the haystack" test, and it is
reproducible and honest — unlike a hard-coded metric.

Reports:
  * Recall@1, Recall@3, Recall@5
  * Mean Reciprocal Rank (MRR)

Run:
    python -m src.evaluation.evaluate_rag
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.rag.retriever import TfidfRetriever

DATA_PATH = "data/raw/master punjabi dataset.xlsx"
OUT_PATH = Path("data/processed/rag_retrieval_metrics.json")


def main() -> None:
    df = pd.read_excel(DATA_PATH).dropna(subset=["Article", "Summary"]).reset_index(drop=True)
    retriever = TfidfRetriever().fit(df["Article"].tolist(), df["Summary"].tolist())

    k_max = 5
    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []

    for true_idx, summary in enumerate(df["Summary"].tolist()):
        results = retriever.search(summary, k=k_max)
        ranked_ids = [r.doc_id for r in results]
        if true_idx in ranked_ids:
            rank = ranked_ids.index(true_idx) + 1
            reciprocal_ranks.append(1.0 / rank)
            for k in hits_at:
                if rank <= k:
                    hits_at[k] += 1
        else:
            reciprocal_ranks.append(0.0)

    n = len(df)
    metrics = {
        "n_queries": n,
        "recall_at_1": round(hits_at[1] / n, 4),
        "recall_at_3": round(hits_at[3] / n, 4),
        "recall_at_5": round(hits_at[5] / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "retriever_backend": "tfidf",
    }

    print("=== RAG RETRIEVAL METRICS (real, computed just now) ===")
    for k_name, v in metrics.items():
        print(f"{k_name}: {v}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
