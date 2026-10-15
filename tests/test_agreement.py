import pytest
from src.evaluation.agreement import bio_to_spans, cohen_kappa, compare


def sent(pairs):
    return [(t, g) for t, g in pairs]


def test_bio_to_spans_basic_and_stray_i():
    assert bio_to_spans(["B-PER", "I-PER", "O", "B-LOC"]) == [(0, 2, "PER"), (3, 4, "LOC")]
    assert bio_to_spans(["I-ORG", "I-ORG", "O"]) == [(0, 2, "ORG")]
    assert bio_to_spans(["B-PER", "I-LOC"]) == [(0, 1, "PER"), (1, 2, "LOC")]


def test_bio_invalid_tag():
    with pytest.raises(ValueError):
        bio_to_spans(["PER"])


def test_kappa_perfect_and_chance():
    assert cohen_kappa(list("ababab"), list("ababab")) == 1.0
    assert cohen_kappa(list("aaaa"), list("aaaa")) == 1.0
    assert abs(cohen_kappa(list("abab"), list("aabb"))) < 1e-9


def test_identical_annotations_full_agreement():
    s = [sent([("a", "B-PER"), ("b", "I-PER"), ("c", "O")]), sent([("d", "B-LOC")])]
    rep, diffs = compare(s, s)
    assert rep["token_kappa_bio"] == 1.0
    assert rep["entity_exact"]["f1"] == 1.0
    assert diffs == []


def test_boundary_and_type_disagreement():
    a = [sent([("x", "B-PER"), ("y", "I-PER"), ("z", "B-LOC")])]
    b = [sent([("x", "B-PER"), ("y", "O"), ("z", "B-ORG")])]
    rep, diffs = compare(a, b)
    assert rep["entity_exact"]["tp"] == 0
    assert rep["span_only_f1"] == 0.5          # only z matches by boundaries
    assert rep["type_agreement_on_matching_spans"] == 0.0
    assert len(diffs) == 4


def test_token_mismatch_raises():
    a = [sent([("x", "O")])]
    b = [sent([("y", "O")])]
    with pytest.raises(ValueError):
        compare(a, b)
