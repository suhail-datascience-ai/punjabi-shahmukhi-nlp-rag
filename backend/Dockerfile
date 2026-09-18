# Minimal image for the offline demo (TF-IDF classifier + lexicon NER +
# TF-IDF RAG retriever). If you switch USE_TRANSFORMER_* flags on, install
# torch/transformers too (they're already in requirements.txt at repo root,
# just uncomment the heavier layer below and expect a much bigger image).

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    pandas numpy scikit-learn joblib openpyxl fastapi "uvicorn[standard]" pydantic

COPY . .

# Build the artifacts the API needs if they aren't already present
RUN python -m src.classification.train_baseline || true
RUN python -m src.rag.retriever || true

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
