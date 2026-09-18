from __future__ import annotations

import joblib

from src.preprocessing.normalize import normalize_shahmukhi
from backend.app import config


class ClassifierService:
    """Wraps whichever classification backend is configured.

    Both backends expose the same `.predict(text) -> (label, scores_dict)`
    surface so `routes.py` never needs to know which one is active.
    """

    def __init__(self) -> None:
        self.backend_name: str
        if config.USE_TRANSFORMER_CLASSIFIER and config.TRANSFORMER_CLASSIFIER_PATH.exists():
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_CLASSIFIER_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(config.TRANSFORMER_CLASSIFIER_PATH)
            self.model.eval()
            self._torch = torch
            self.backend_name = "xlm-roberta (fine-tuned)"
        else:
            if not config.TFIDF_CLASSIFIER_PATH.exists():
                raise FileNotFoundError(
                    f"No classifier found at {config.TFIDF_CLASSIFIER_PATH}. "
                    "Run `python -m src.classification.train_baseline` first."
                )
            self.pipeline = joblib.load(config.TFIDF_CLASSIFIER_PATH)
            self.backend_name = "tfidf+logreg (baseline)"

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        clean = normalize_shahmukhi(text)
        if self.backend_name.startswith("xlm-roberta"):
            inputs = self.tokenizer(clean, return_tensors="pt", truncation=True, max_length=192)
            with self._torch.no_grad():
                logits = self.model(**inputs).logits[0]
                probs = self._torch.softmax(logits, dim=-1).tolist()
            id2label = self.model.config.id2label
            scores = {id2label[i]: round(p, 4) for i, p in enumerate(probs)}
        else:
            probs = self.pipeline.predict_proba([clean])[0]
            classes = self.pipeline.classes_
            scores = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}

        label = max(scores, key=scores.get)
        return label, scores
