# Gold-standard Shahmukhi NER data

Nothing in this folder is gold until it has passed the agreement check in `thesis/annotation_guidelines.md`.

Files are added as the annotation progresses: pilot_A.conll, pilot_B.conll (two annotators, same sentences), pilot_agreement.json, pilot_disagreements.tsv, and MANIFEST.json (sha256 and guideline version for every released file).

Format: CoNLL, one `token tag` per line, blank line between sentences, tags in BIO (O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC).

Rules:
- The final test split is frozen with a hash in MANIFEST.json before any model is trained and is never used for training or model selection.
- Keep the raw per-annotator files, not only the adjudicated result.
- Record the source (domain, URL or dataset) and licence of every text source.
