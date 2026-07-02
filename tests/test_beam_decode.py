"""Tests for the element-level beam/DP Morse decoder."""

from morse_char_recognizer.beam_decode import BeamConfig, beam_decode_regions, enumerate_valid_spans
from morse_char_recognizer.morse_table import MORSE_TABLE
from morse_char_recognizer.segment import Region

UNIT = 100  # samples per dit


def regions_from_text(
    text: str,
    *,
    unit: int = UNIT,
    char_gap_units: float = 3.0,
    word_gap_units: float = 7.0,
) -> list[Region]:
    """Build clean-timing key-down regions for an upper-case message."""
    pos = 0
    regions: list[Region] = []
    for wi, word in enumerate(text.split(" ")):
        if wi > 0:
            pos += int(word_gap_units * unit)
        for ci, ch in enumerate(word):
            if ci > 0:
                pos += int(char_gap_units * unit)
            for ei, sym in enumerate(MORSE_TABLE[ch]):
                if ei > 0:
                    pos += unit  # intra-element gap
                duration = (3 if sym == "-" else 1) * unit
                regions.append(Region(pos, pos + duration))
                pos += duration
    return regions


def test_decodes_simple_message():
    regions = regions_from_text("SOS")
    assert beam_decode_regions(regions, UNIT) == "SOS"


def test_single_dit_is_kept_as_e():
    # the failure mode of the min-segment filter: a lone dit must stay an E
    regions = regions_from_text("E")
    assert beam_decode_regions(regions, UNIT) == "E"


def test_word_gap_emits_space():
    regions = regions_from_text("ON AIR")
    assert beam_decode_regions(regions, UNIT) == "ON AIR"


def test_morse_validity_forces_split_on_small_gap():
    # "...--." is not a valid glyph, so S+G must split even with a tight gap
    regions = regions_from_text("SG", char_gap_units=1.6)
    assert beam_decode_regions(regions, UNIT) == "SG"


def test_short_noise_element_is_dropped():
    regions = regions_from_text("K")
    # inject a sub-unit speck inside the inter-character space; it must not
    # become a spurious E
    speck_start = regions[0].end + UNIT // 2
    noisy = [regions[0], Region(speck_start, speck_start + UNIT // 5), *regions[1:]]
    config = BeamConfig(min_element_units=0.45)
    assert beam_decode_regions(noisy, UNIT, config) == "K"


def test_empty_returns_empty():
    assert beam_decode_regions([], UNIT) == ""


def test_allowed_tokens_restrict_decode_vocabulary():
    # "..-.." is É in the full table. The WebSDR ham-copy profile intentionally
    # excludes accented glyphs so noisy timing cannot emit them as copy.
    regions = []
    pos = 0
    for sym in "..-..":
        duration = (3 if sym == "-" else 1) * UNIT
        regions.append(Region(pos, pos + duration))
        pos += duration + UNIT

    ham_copy = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + (
        "/",
        ".",
        ",",
        "?",
        "=",
        "+",
        "<SK>",
        "<SN>",
    )

    assert beam_decode_regions(regions, UNIT) == "É"
    constrained = beam_decode_regions(regions, UNIT, BeamConfig(allowed_tokens=ham_copy))
    assert "É" not in constrained
    assert (0, 5, "É") not in enumerate_valid_spans(
        list("..-.."),
        max_elements_per_char=8,
        allowed_tokens=ham_copy,
    )
