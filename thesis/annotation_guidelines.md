# Shahmukhi NER Annotation Guidelines (v0.1, draft for pilot)

Version 0.1 is a working draft, revised after the pilot. Every change gets a new version number and a changelog line at the bottom.

## 1. Scope

Annotate named entities in Shahmukhi (Perso-Arabic script) Punjabi text. Annotate tokens as given: do not re-tokenize, normalize or edit the text while annotating. Normalization happens before annotation (src/preprocessing/normalize.py).

## 2. Entity types

- PER: individual people (real, historical, religious, fictional). Example: عمران خان
- LOC: countries, cities, districts, regions, rivers, mountains. Examples: لاہور، پنجاب، پاکستان
- ORG: government bodies, companies, parties, universities, sports clubs, media outlets. Example: پنجاب یونیورسٹی
- MISC: named events and titles of works (optional layer; results are reported with and without it)

PER, LOC and ORG match the existing Shahmukhi NER dataset, so cross-dataset comparison is possible.

## 3. Scheme

BIO (IOB2): B-TYPE for the first token of an entity, I-TYPE for continuation tokens, O otherwise. An I- tag must follow a B- or I- of the same type. File format: one "token tag" per line, blank line between sentences.

## 4. Decision rules

1. No capitalization in Shahmukhi: decide from context, not from form.
2. Titles and honorifics are excluded from the span; tag the name only.
3. Postpositions are excluded from the span (for example دا، دی، دے، نے، نوں).
4. Common nouns, pronouns, dates, numbers and percentages are not tagged.
5. Geopolitical entities are LOC, even when used metonymically for a government.
6. A national sports team is ORG when the text clearly refers to the team.
7. Nested entities: tag the outermost span. A place name inside an organization name is part of the ORG.
8. Nationalities, languages and religions are not tagged.
9. Deities are not tagged. Historical and religious persons are PER (to be reviewed after the pilot).
10. Uncertain cases: note the sentence ID, choose the most likely tag, never leave it blank.

## 5. Pilot and agreement

1. Two annotators independently annotate the same pilot set of 300 sentences, drawn across domains.
2. Run: python -m src.evaluation.agreement A.conll B.conll --out report.json --disagreements diffs.tsv
3. Working targets: token kappa over BIO tags >= 0.80 and entity-level exact F1 >= 0.85.
4. Review the disagreements together, update the guidelines, bump the version, and re-annotate a fresh sample if the target was missed.
5. Only after the targets are met does full annotation start. Adjudicate the remaining disagreements and store the final gold files plus both raw annotations.

## Changelog

- v0.1: initial draft.
