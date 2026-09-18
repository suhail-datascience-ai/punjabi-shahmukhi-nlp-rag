"""Smoke tests — run with `pytest` from the repo root.

These exercise the real, offline artifacts (TF-IDF classifier, lexicon NER,
TF-IDF RAG index). They assume you've already run:
    python -m src.classification.train_baseline
    python -m src.rag.retriever
"""

import pytest

from src.preprocessing.normalize import normalize_shahmukhi
from src.ner.lexicon_bootstrap import Gazetteer, clean_and_tokenize


def test_normalize_strips_urls_and_citations():
    text = "آزادی دا مطلب [1] https://example.com/x اے"
    out = normalize_shahmukhi(text)
    assert "http" not in out
    assert "[1]" not in out
    assert "آزادی" in out


def test_normalize_is_idempotent():
    text = "پنجاب دی راجدھانی لاهور اے۔"
    once = normalize_shahmukhi(text)
    twice = normalize_shahmukhi(once)
    assert once == twice


def test_gazetteer_tags_known_entity():
    gaz = Gazetteer()
    tokens = clean_and_tokenize("عمران خان لاهور گیا")
    tags = gaz.tag(tokens)
    assert "B-PER" in tags
    assert "B-LOC" in tags


def test_classifier_service_smoke():
    pytest.importorskip("joblib")
    from pathlib import Path

    if not Path("models/tfidf_logreg_classifier.joblib").exists():
        pytest.skip("Run `python -m src.classification.train_baseline` first.")

    from backend.app.services.classifier_service import ClassifierService

    svc = ClassifierService()
    label, scores = svc.predict("پاکستان کرکٹ ٹیم نے میچ جتیا۔")
    assert label in scores
    assert abs(sum(scores.values()) - 1.0) < 0.05


def test_rag_service_smoke():
    from pathlib import Path

    if not Path("models/rag_index/tfidf_retriever.pkl").exists():
        pytest.skip("Run `python -m src.rag.retriever` first.")

    from backend.app.services.rag_service import RAGService

    svc = RAGService()
    results, _ = svc.query("پنجاب دی تاریخ", top_k=3)
    assert len(results) <= 3
    assert all(r.score >= 0 for r in results)
