"""CTC vocabulary for Morse sequence recognition.

The CTC head emits one of ``num_classes + 1`` symbols per frame. Index ``0``
is the CTC blank, indices ``1..N`` correspond to the visible characters in
``CTC_CHARS``. Word boundaries are encoded as a literal space character; CTC
learns to emit a space frame when the inter-character gap exceeds the
within-character spacing.
"""

from __future__ import annotations

from dataclasses import dataclass

CTC_CHARS: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
BLANK_INDEX: int = 0


@dataclass(frozen=True)
class CtcVocab:
    """Map characters to CTC token indices and back."""

    chars: tuple[str, ...] = CTC_CHARS

    @property
    def num_classes(self) -> int:
        return len(self.chars) + 1  # +1 for blank

    @property
    def blank_index(self) -> int:
        return BLANK_INDEX

    def char_to_index(self, char: str) -> int:
        if len(char) != 1:
            raise ValueError(f"expected single character, got {char!r}")
        upper = char.upper()
        try:
            return self.chars.index(upper) + 1
        except ValueError as exc:
            raise ValueError(f"character not in vocab: {char!r}") from exc

    def index_to_char(self, index: int) -> str:
        if index == self.blank_index:
            return ""
        position = index - 1
        if position < 0 or position >= len(self.chars):
            raise ValueError(f"index out of range: {index}")
        return self.chars[position]

    def encode(self, text: str) -> list[int]:
        """Encode a normalised text string to CTC target indices.

        Unknown characters are silently dropped. Runs of internal whitespace
        collapse to a single space token.
        """
        out: list[int] = []
        previous_space = True  # suppress leading spaces
        for char in text.upper():
            if char.isspace():
                if not previous_space:
                    out.append(self.char_to_index(" "))
                    previous_space = True
                continue
            if char not in self.chars:
                continue
            out.append(self.char_to_index(char))
            previous_space = False
        while out and out[-1] == self.char_to_index(" "):
            out.pop()
        return out

    def decode(self, indices: list[int]) -> str:
        return "".join(self.index_to_char(idx) for idx in indices)
