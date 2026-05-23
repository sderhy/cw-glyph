"""Tiny character n-gram language model for CTC beam rescoring."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from morse_char_recognizer.ctc.vocab import CtcVocab

START_TOKEN = "^"
END_TOKEN = "$"


@dataclass
class CharNgramLm:
    """Stupid-backoff character n-gram LM over the CTC vocabulary.

    The LM scores text strings (without CTC blanks). Tokens are uppercase
    characters from ``vocab.chars``. A sentinel ``^`` marks the start of a
    string and ``$`` marks the end; spaces are first-class tokens since the
    CTC vocabulary includes a literal space.
    """

    vocab: CtcVocab = field(default_factory=CtcVocab)
    order: int = 3
    smoothing: float = 0.1
    counts: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(dict))
    context_totals: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    backoff: float = 0.4

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("order must be >= 1")
        if self.smoothing < 0:
            raise ValueError("smoothing must be >= 0")

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocab.chars) + 1  # +1 for END_TOKEN

    def train(self, corpus: list[str]) -> None:
        for sentence in corpus:
            tokens = self._tokenize(sentence)
            if not tokens:
                continue
            padded = [START_TOKEN] * (self.order - 1) + tokens + [END_TOKEN]
            for i in range(self.order - 1, len(padded)):
                target = padded[i]
                for n in range(1, self.order + 1):
                    context = "".join(padded[i - n + 1 : i])
                    self.counts.setdefault(context, {})
                    self.counts[context][target] = self.counts[context].get(target, 0) + 1
                    self.context_totals[context] = self.context_totals.get(context, 0) + 1

    def log_prob(self, target: str, context: str) -> float:
        """Stupid-backoff log probability of ``target`` given ``context``.

        ``context`` is the last up-to-``order-1`` characters of the decoded
        text (without sentinel padding); the LM internally pads with start
        tokens at the beginning of decoding.
        """
        context = context[-(self.order - 1) :] if self.order > 1 else ""
        for n in range(len(context), -1, -1):
            sub = context[-n:] if n > 0 else ""
            total = self.context_totals.get(sub, 0)
            if total <= 0:
                continue
            target_count = self.counts.get(sub, {}).get(target, 0)
            if target_count > 0:
                numerator = target_count + self.smoothing
                denom = total + self.smoothing * self.vocabulary_size
                prob = numerator / denom
                backoff_penalty = (len(context) - n) * math.log(self.backoff)
                return math.log(prob) + backoff_penalty
        # Final fallback: uniform smoothing
        denom = max(1, self.smoothing * self.vocabulary_size)
        return math.log(self.smoothing / denom) + len(context) * math.log(self.backoff)

    def score(self, text: str) -> float:
        """Sum of conditional log probs along ``text`` (without end token)."""
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        context = START_TOKEN * (self.order - 1)
        total = 0.0
        for token in tokens:
            total += self.log_prob(token, context)
            context = (context + token)[-(self.order - 1) :] if self.order > 1 else ""
        return total

    def _tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for char in text.upper():
            if char.isspace():
                if " " in self.vocab.chars and (not out or out[-1] != " "):
                    out.append(" ")
                continue
            if char in self.vocab.chars:
                out.append(char)
        while out and out[-1] == " ":
            out.pop()
        return out


def default_corpus() -> list[str]:
    """A small Morse-realistic corpus.

    The strings here exercise common amateur-radio phrases, call-sign
    patterns and English wordlist letters. Not exhaustive, but enough to
    bias an LM toward sensible character transitions during beam search.
    """
    return [
        "CQ CQ CQ DE F4ABC F4ABC F4ABC K",
        "CQ DX DE W1AW W1AW K",
        "DE K6KPH QSL TU 73",
        "TNX FER QSO DR OM 73 SK",
        "UR RST 599 NAME BOB QTH PARIS",
        "PSE QSL VIA BUREAU TU",
        "GM GA GE OM HW CPY",
        "QRZ DE F8ABC QRZ",
        "TNX FB QSO 73 ES GL",
        "WX HR FB TEMP 18 C HUMID 70 PERCENT",
        "RIG ICOM 7300 ANT DIPOLE 40M",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789",
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
        "PARIS PARIS PARIS PARIS",
        "DE F4ABC TEST DE F4ABC TEST",
        "CFM QSL TNX FER NICE QSO",
        "73 SK ES GL DR OM",
        "BK QSY 14060 CW",
        "QRT NW 73 ES TU",
        "PSE K AGN AGN",
    ]
