"""Morse-realistic text corpus for LM training and noisy-text evaluation.

The corpus is intentionally small but designed so that a character n-gram
LM can pick up:

- ``CQ``, ``DE``, ``TU``, ``73``, ``SK`` and similar amateur-radio idioms;
- callsign-shaped tokens such as ``F4ABC``, ``W1AW``, ``ON4KST``;
- common English words mixed with Q-codes.

``train_test_split`` provides a deterministic, leakage-free train/test
split so the LM is never trained on the exact strings used for evaluation.
"""

from __future__ import annotations

from random import Random

CALLSIGNS = (
    "F4ABC", "F5XYZ", "F8KQR", "F9OPQ", "G3RWT", "G7XYZ", "EI8GH", "OZ1ABC",
    "ON4KST", "OE3NHQ", "SP5XQR", "DL1ABC", "DK5BC", "PA0XCD", "SM5MX",
    "W1AW", "W2XK", "W4BWT", "K6KPH", "K9LA", "WA6FXT", "KB1RZL", "VE3NXY",
    "VE7AHA", "JA1NUT", "JE3HHT", "VK2QF", "VK6HG", "ZL1ABC", "BV2KI",
    "LU1ABC", "PY2NX", "CE3WW", "HK6PSG", "TI2HMG", "5B4ABC", "9A5BB",
    "OK1MP", "OL1A", "HA8JV", "YO3CTK", "YU1ABC", "S57DX", "EA3GHQ",
    "IK4LZH", "IZ8DFO", "HB9AGH", "HB0WR", "9V1HY", "DU1IST",
)

QCODES = (
    "QRA", "QRG", "QRH", "QRI", "QRK", "QRL", "QRM", "QRN", "QRO",
    "QRP", "QRQ", "QRS", "QRT", "QRU", "QRV", "QRZ", "QSA", "QSB",
    "QSL", "QSO", "QSY", "QTH", "QTR",
)

PHRASES = (
    "CQ CQ CQ DE",
    "CQ DX DE",
    "QRZ DE",
    "TU 73 DE",
    "TU FER QSO",
    "TNX FB QSO 73",
    "PSE QSL TU",
    "PSE QSL VIA BUREAU",
    "CFM QSL TNX",
    "GM ES TU FER CALL",
    "GA OM HW CPY",
    "GE OM TNX FER QSO",
    "UR RST 599 5NN",
    "NAME BOB QTH PARIS",
    "NAME JOHN QTH BERLIN",
    "NAME PETE QTH ROMA",
    "WX HR FB ES SUNNY",
    "WX HR RAIN TEMP 8 C",
    "RIG ICOM 7300 ANT DIPOLE",
    "RIG FT857 ANT YAGI 3 EL",
    "TRX K3 ANT VERTICAL",
    "ANT G5RV PWR 100 W",
    "PWR 5 W ANT END FED",
    "BK QSY 14060 CW",
    "QSY UP 5 TU",
    "73 SK ES GL",
    "73 ES TU",
    "QRT NW 73 ES TU",
    "TU AGN OM 73",
    "FB OM TU FER NICE QSO",
    "DR OM TNX FB QSO 73",
    "K AGN AGN",
    "PSE K AGN",
    "QSL TU 73 SK",
    "TU OM CUL",
    "TEST DE F4ABC TEST",
    "RUN ON 14025 KHZ",
    "CQ TEST CQ TEST DE",
    "CQ WW DE",
    "ARRL DX TEST",
    "CFM 599 NR 001",
    "NR 042 ME 599",
    "PRACTICE MAKES PERFECT",
    "THE QUICK BROWN FOX",
    "JUMPS OVER THE LAZY DOG",
    "MORSE CODE IS A LANGUAGE",
    "HAM RADIO IS FUN",
    "AMATEUR RADIO LOVES CW",
    "GOOD CW SKILLS COME WITH TIME",
    "LEARN ONE CHARACTER AT A TIME",
)

WORDLIST = (
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "ANY", "CAN",
    "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM",
    "HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO",
    "BOY", "DID", "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE", "QSO",
    "PARIS", "LONDON", "BERLIN", "MADRID", "ROMA", "TOKYO", "MOSCOW",
    "ARRL", "RSGB", "DARC", "REF", "JARL", "WIA",
)


def build_corpus(seed: int = 0, num_sentences: int = 600) -> list[str]:
    """Generate a deterministic Morse-realistic, deduplicated corpus.

    Sentences mix call-sign patterns, Q-codes and English/CW idioms.
    """
    rng = Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    for phrase in PHRASES:
        if phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
    attempts = 0
    while len(out) < num_sentences and attempts < num_sentences * 20:
        attempts += 1
        kind = rng.random()
        if kind < 0.35:
            candidate = _callsign_phrase(rng)
        elif kind < 0.65:
            candidate = _qso_phrase(rng)
        elif kind < 0.85:
            candidate = _english_phrase(rng)
        else:
            candidate = _qcode_phrase(rng)
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out[:num_sentences]


def train_test_split(
    corpus: list[str],
    test_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Split a corpus into train / test halves with deterministic shuffling."""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must lie in (0, 1)")
    rng = Random(seed)
    indices = list(range(len(corpus)))
    rng.shuffle(indices)
    cutoff = int(round(len(indices) * (1.0 - test_ratio)))
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]
    return [corpus[i] for i in train_indices], [corpus[i] for i in test_indices]


def _callsign_phrase(rng: Random) -> str:
    prefix = rng.choice(("CQ DE", "QRZ DE", "DE", "TEST DE", "CQ DX DE", "CFM DE"))
    a = rng.choice(CALLSIGNS)
    if rng.random() < 0.6:
        return f"{prefix} {a}"
    b = rng.choice(CALLSIGNS)
    return f"{prefix} {a} {b}"


def _qso_phrase(rng: Random) -> str:
    rst = rng.randint(479, 599)
    template = rng.choice(
        (
            "UR RST {rst} NAME {name} QTH {qth}",
            "NAME {name} QTH {qth} 73",
            "TNX FB QSO {rst} 73",
            "CFM {rst} TU 73 SK",
            "RIG K3 ANT DIPOLE PWR 100 W",
            "WX HR FB ES SUNNY",
        )
    )
    return template.format(
        rst=rst,
        name=rng.choice(("BOB", "JOHN", "PETE", "MAX", "DAN", "ANNA", "MARIE")),
        qth=rng.choice(("PARIS", "LONDON", "BERLIN", "MADRID", "ROMA")),
    )


def _english_phrase(rng: Random) -> str:
    length = rng.randint(3, 6)
    words = [rng.choice(WORDLIST) for _ in range(length)]
    return " ".join(words)


def _qcode_phrase(rng: Random) -> str:
    return f"{rng.choice(QCODES)} {rng.choice(QCODES)} {rng.choice(QCODES)}"
