from __future__ import annotations

from src.ner.lexicon_bootstrap import Gazetteer, clean_and_tokenize
from src.preprocessing.normalize import normalize_shahmukhi
from backend.app import config


class NERService:
    def __init__(self) -> None:
        if config.USE_TRANSFORMER_NER and config.TRANSFORMER_NER_PATH.exists():
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_NER_PATH)
            self.model = AutoModelForTokenClassification.from_pretrained(config.TRANSFORMER_NER_PATH)
            self.model.eval()
            self._torch = torch
            self.backend_name = "xlm-roberta-ner (fine-tuned)"
        else:
            self.gazetteer = Gazetteer()
            self.backend_name = "lexicon-bootstrap (silver-standard)"

    def tag(self, text: str) -> tuple[list[str], list[str]]:
        clean = normalize_shahmukhi(text)
        if self.backend_name.startswith("xlm-roberta"):
            tokens = clean.split()
            enc = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
            with self._torch.no_grad():
                logits = self.model(**enc).logits[0]
            preds = logits.argmax(-1).tolist()
            word_ids = enc.word_ids()
            id2label = self.model.config.id2label
            tags = ["O"] * len(tokens)
            seen = set()
            for pred, word_id in zip(preds, word_ids):
                if word_id is None or word_id in seen:
                    continue
                seen.add(word_id)
                tags[word_id] = id2label[pred]
            return tokens, tags
        else:
            tokens = clean_and_tokenize(clean)
            tags = self.gazetteer.tag(tokens)
            return tokens, tags

    @staticmethod
    def extract_entities(tokens: list[str], tags: list[str]) -> list[dict]:
        entities = []
        current = None
        for i, (tok, tag) in enumerate(zip(tokens, tags)):
            if tag.startswith("B-"):
                if current:
                    entities.append(current)
                current = {"text": tok, "type": tag[2:], "start_token": i, "end_token": i}
            elif tag.startswith("I-") and current and tag[2:] == current["type"]:
                current["text"] += " " + tok
                current["end_token"] = i
            else:
                if current:
                    entities.append(current)
                    current = None
        if current:
            entities.append(current)
        return entities
