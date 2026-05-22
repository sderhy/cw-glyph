"""Class vocabularies for Morse glyph recognition."""

from __future__ import annotations

from morse_char_recognizer.morse_table import MORSE_TABLE, PROSIGNS

BASIC_CLASSES: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
PUNCTUATION_CLASSES: tuple[str, ...] = ("?", ",", ".", "=", "+", "'", "/", "$")
COPY_CLASSES: tuple[str, ...] = (*BASIC_CLASSES, *PUNCTUATION_CLASSES)
# <BT> is acoustically identical to "=" and <AR> is identical to "+". Keep the
# printed punctuation forms in REAL_CLASSES so the classifier has one class per
# distinct sound pattern.
PROSIGN_CLASSES: tuple[str, ...] = ("<SK>", "<KN>", "<AS>", "<HH>", "<SN>", "<CT>")
RUN_ON_CLASSES: tuple[str, ...] = ("<UR>", "<BK>")
REAL_CLASSES: tuple[str, ...] = (
    *COPY_CLASSES,
    *PROSIGN_CLASSES,
    *RUN_ON_CLASSES,
)
DEFAULT_CLASSES = BASIC_CLASSES

CLASS_PRESETS: dict[str, tuple[str, ...]] = {
    "basic": BASIC_CLASSES,
    "copy": COPY_CLASSES,
    "real": REAL_CLASSES,
}


def parse_class_spec(
    classes: str | None,
    *,
    preset: str = "basic",
) -> tuple[str, ...]:
    """Parse a CLI class specification.

    Plain strings keep the historical behaviour: ``"ABC123"`` means one class
    per character. Comma-separated input supports multi-character glyph tokens,
    for example ``"A,B,?,<KN>,<UR>"``.
    """
    if classes is None:
        return CLASS_PRESETS[preset]
    spec = classes.strip()
    if not spec:
        return CLASS_PRESETS[preset]
    if "," in spec:
        return tuple(part.strip().upper() for part in spec.split(",") if part.strip())
    return tuple(spec.upper())


def morse_code_for_class(token: str) -> str:
    """Return the dit/dah code for a class token, or ``""`` if unknown."""
    token = token.upper()
    if token in MORSE_TABLE:
        return MORSE_TABLE[token]
    if token in PROSIGNS:
        return PROSIGNS[token]
    return ""
