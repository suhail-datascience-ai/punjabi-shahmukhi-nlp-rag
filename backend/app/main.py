"""
FastAPI backend for the Punjabi (Shahmukhi) NLP Intelligence + RAG project.

Run locally:
    uvicorn backend.app.main:app --reload --port 8000

Then open http://localhost:8000/docs for interactive Swagger docs, or open
`frontend/index.html` in a browser (it talks to this server on
http://localhost:8000 by default).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    NEREntity,
    NERRequest,
    NERResponse,
    RAGPassage,
    RAGQueryRequest,
    RAGQueryResponse,
)
from backend.app.services.classifier_service import ClassifierService
from backend.app.services.ner_service import NERService
from backend.app.services.rag_service import RAGService

_services: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models once at startup, not per-request.
    _services["classifier"] = ClassifierService()
    _services["ner"] = NERService()
    try:
        _services["rag"] = RAGService()
    except FileNotFoundError as e:
        print(f"[startup] RAG service unavailable: {e}")
        _services["rag"] = None
    yield
    _services.clear()


app = FastAPI(
    title="Punjabi Shahmukhi NLP Intelligence API",
    description=(
        "News classification, Named Entity Recognition, and Retrieval-"
        "Augmented Generation over a Shahmukhi-script Punjabi news corpus."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    rag = _services.get("rag")
    return HealthResponse(
        status="ok",
        classifier_backend=_services["classifier"].backend_name,
        ner_backend=_services["ner"].backend_name,
        rag_backend=rag.backend_name if rag else "unavailable",
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    label, scores = _services["classifier"].predict(req.text)
    return ClassifyResponse(
        label=label,
        confidence=scores[label],
        all_scores=scores,
        backend=_services["classifier"].backend_name,
    )


@app.post("/ner", response_model=NERResponse)
def ner(req: NERRequest):
    svc = _services["ner"]
    tokens, tags = svc.tag(req.text)
    entities = [NEREntity(**e) for e in svc.extract_entities(tokens, tags)]
    return NERResponse(tokens=tokens, tags=tags, entities=entities, backend=svc.backend_name)


@app.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(req: RAGQueryRequest):
    svc = _services.get("rag")
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="RAG index not built yet. Run `python -m src.rag.retriever` and restart the server.",
        )
    results, answer = svc.query(req.query, top_k=req.top_k)
    passages = [RAGPassage(doc_id=r.doc_id, score=r.score, text=r.text, summary=r.summary) for r in results]
    return RAGQueryResponse(query=req.query, passages=passages, generated_answer=answer, backend=svc.backend_name)
