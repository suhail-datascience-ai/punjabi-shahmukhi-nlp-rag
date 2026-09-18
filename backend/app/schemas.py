from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw Shahmukhi Punjabi text to classify.")


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    all_scores: dict[str, float]
    backend: str


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw Shahmukhi Punjabi text to tag.")


class NEREntity(BaseModel):
    text: str
    type: str
    start_token: int
    end_token: int


class NERResponse(BaseModel):
    tokens: list[str]
    tags: list[str]
    entities: list[NEREntity]
    backend: str


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question or search query in Punjabi.")
    top_k: int = Field(3, ge=1, le=10)


class RAGPassage(BaseModel):
    doc_id: int
    score: float
    text: str
    summary: str | None = None


class RAGQueryResponse(BaseModel):
    query: str
    passages: list[RAGPassage]
    generated_answer: str | None = None
    backend: str


class HealthResponse(BaseModel):
    status: str
    classifier_backend: str
    ner_backend: str
    rag_backend: str
