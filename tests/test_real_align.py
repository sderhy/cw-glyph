"""Tests for forced alignment of detected elements to a truth string."""

from morse_char_recognizer.morse_table import MORSE_TABLE
from morse_char_recognizer.real_align import align_edit, clean_char_spans


def _codes(text: str) -> list[tuple[str, str]]:
    return [(ch, MORSE_TABLE[ch]) for ch in text]


def test_clean_alignment_recovers_every_char():
    truth = _codes("TNX")  # - / -. / -..-
    symbols = list("-" "-." "-..-")
    spans = clean_char_spans(symbols, truth)
    assert spans == [("T", 0, 1), ("N", 1, 3), ("X", 3, 7)]


def test_missed_element_drops_a_corrupted_char():
    # one element lost across the T+N pair: the two cannot both align cleanly,
    # so at most one is harvested (which one is an alignment tie, either is fine).
    truth = _codes("TN")  # - / -.
    symbols = list("-.")  # only two elements survive of the three
    spans = clean_char_spans(symbols, truth)
    assert len(spans) <= 1
    for tok, a, b in spans:  # anything harvested must exactly match its code
        assert "".join(symbols[a:b]) == MORSE_TABLE[tok]


def test_substituted_element_is_not_harvested():
    # symbol misclassified: S sends "..." but middle dit read as dah -> "..-"
    truth = _codes("S")
    spans = clean_char_spans(list("..-"), truth)
    assert spans == []  # pattern mismatch, nothing clean


def test_align_edit_reports_indels():
    ops = align_edit("-.", "--.")  # one deletion of the extra dah
    assert (None, 1) in ops or (None, 0) in ops
    matches = [(i, j) for i, j in ops if i is not None and j is not None]
    assert len(matches) == 2
