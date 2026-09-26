"""A personal test set: your own voice, labelled by what you *meant* to say.

The scorer has so far been measured on synthetic speech, and research on
pronunciation assessment keeps finding that synthetic data is a weak stand-in for
real learners (the IQRA 2026 challenge: a small set of real mispronunciations
beat far larger synthetic corpora). The obvious fix — label your own recordings
right or wrong — does not work for a learner, who often cannot hear their own
errors. That inability is the reason perception training exists.

So the labels here come from intent, which you do know. Each word is recorded
twice: your best attempt, and a deliberate, typical English-speaker error
("Tür" with an English "oo"). That gives:

- **catch rate**: flags on the deliberate error. The label is certain.
- **false-alarm rate**: flags on your best attempt. The label can only be wrong
  one way (a "best attempt" that was actually off gets counted as a false alarm
  when the flag was deserved), so this estimate errs high, never low.

Faked errors are clearer than natural ones, so the catch rate is an upper bound
on how well subtle, unintended errors are caught.

Recordings live in ~/.mdd/testset/<lang>/<contrast>/<word>__<kind>__<take>.wav
(override the root with `$MDD_TESTSET`). The file name carries the label, so the
set survives any bookkeeping bug and can be inspected by hand.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

#: Takes per word and version. Two lets a flag on one but not the other show up
#: as inconsistency rather than being mistaken for a property of the word.
TAKES = 2
KINDS = ("best", "error")
MIN_SECONDS = 0.25
MIN_PEAK = 0.01


@dataclass(frozen=True)
class Contrast:
    id: str
    label: str
    words: tuple[str, ...]
    #: The phones under test; a take is "flagged" if any of these is flagged.
    targets: tuple[str, ...]
    #: How to make the deliberate error, completing "say it ...".
    error: str
    #: What the recogniser should hear if it diagnoses the error correctly.
    heard_as: tuple[str, ...]
    #: Only the word's first / last expected phone is under test.
    position: str | None = None
    #: Vowel length: the app never flags length alone (37% false alarms on
    #: native speech), so the evaluator also scores it with length flags on.
    length: bool = False


CONTRASTS: dict[str, tuple[Contrast, ...]] = {
    "de": (
        Contrast("u-umlaut", "ü", ("Tür", "grün", "fünf", "Brücke", "müde"),
                 ("yː", "y"), "with an English 'oo', as in 'tour'", ("u", "uː", "ʊ")),
        Contrast("o-umlaut", "ö", ("schön", "möchte", "Köln", "hören", "Brötchen"),
                 ("øː", "œ"), "with a plain 'o', as in 'show' or 'hot'", ("o", "oː", "ɔ")),
        Contrast("ich", "ich-Laut", ("ich", "nicht", "Milch", "leicht", "sprechen"),
                 ("ç",), "with a 'k' instead ('ick', 'nickt')", ("k",)),
        Contrast("ach", "ach-Laut", ("Buch", "auch", "noch", "machen", "Nacht"),
                 ("x",), "with a 'k' instead ('Buck', 'mack-en')", ("k",)),
        Contrast("r", "r at the start", ("rot", "Rosen", "Reise", "rund", "Rad"),
                 ("ʁ",), "with an English r, as in 'road'", ("ɹ", "r", "ɻ"),
                 position="first"),
        Contrast("z", "z = ts", ("Zeit", "zehn", "Zug", "Zimmer", "zwei"),
                 ("ts",), "with an English z, as in 'zoo'", ("z", "s")),
        Contrast("final-devoicing", "final devoicing", ("Hund", "Tag", "Kind", "gelb", "Weg"),
                 ("t", "k", "p"), "with a voiced ending, as in English 'hund', 'tag'",
                 ("d", "ɡ", "b"), position="last"),
        Contrast("vowel-length", "vowel length", ("Staat", "Ofen", "Bahn", "Miete", "Sohn"),
                 ("aː", "oː", "iː"), "with a short vowel ('Stadt', 'offen', 'Mitte')",
                 ("a", "ɔ", "o", "ɪ", "i"), length=True),
    ),
    "fr": (
        Contrast("u", "u /y/", ("tu", "rue", "vu", "lune", "sucre"),
                 ("y",), "with an English 'oo', as in 'too'", ("u", "uː", "ʊ")),
        Contrast("nasal", "nasal vowels", ("bon", "vin", "blanc", "pont", "main"),
                 ("ɔ̃", "ɛ̃", "ɑ̃"), "as a plain vowel plus a real n ('bonn', 'vann')",
                 ("ɔ", "o", "ɛ", "a", "ɑ", "e")),
        Contrast("r", "French r", ("rouge", "rire", "riz", "route", "robe"),
                 ("ʁ",), "with an English r, as in 'road'", ("ɹ", "r", "ɻ"),
                 position="first"),
        Contrast("e-acute", "é", ("café", "été", "thé", "nez", "parler"),
                 ("e",), "with an English 'ay', as in 'say'", ("eɪ", "ɛ", "ɪ")),
    ),
}


@dataclass(frozen=True)
class Take:
    lang: str
    contrast: str
    word: str
    kind: str      # "best" or "error"
    take: int

    @property
    def filename(self) -> str:
        return f"{_safe(self.word)}__{self.kind}__{self.take}.wav"


def _safe(word: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", word)


def default_root() -> Path:
    env = os.environ.get("MDD_TESTSET")
    return Path(env) if env else Path.home() / ".mdd" / "testset"


def contrasts(lang: str) -> tuple[Contrast, ...]:
    return CONTRASTS.get(lang, ())


def contrast(lang: str, cid: str) -> Contrast:
    return next(c for c in contrasts(lang) if c.id == cid)


def plan(lang: str) -> list[Take]:
    """Every take, in recording order: per word, best then error, twice.

    Alternating keeps the two versions distinct in your mind, and repeating the
    pair catches a flag that depends on the take rather than the version.
    """
    return [Take(lang, c.id, w, kind, n)
            for c in contrasts(lang) for w in c.words
            for n in range(1, TAKES + 1) for kind in KINDS]


def path_for(root: Path, t: Take) -> Path:
    return Path(root) / t.lang / t.contrast / t.filename


def recorded(root: Path, lang: str) -> list[tuple[Take, Path]]:
    return [(t, path_for(root, t)) for t in plan(lang) if path_for(root, t).exists()]


def pending(root: Path, lang: str, skip: set[Take] = frozenset()) -> list[Take]:
    return [t for t in plan(lang) if t not in skip and not path_for(root, t).exists()]


def save(root: Path, t: Take, source: str | Path) -> Path:
    """Store one take as 16-bit wav, refusing silent or truncated recordings."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(source), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if len(data) < MIN_SECONDS * sr:
        raise ValueError("That recording is too short — say the word again.")
    if float(np.max(np.abs(data))) < MIN_PEAK:
        raise ValueError("That recording is silent — check the microphone.")
    dest = path_for(root, t)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), (np.clip(data, -1, 1) * 32767).astype("int16"), sr)
    return dest


def instruction(t: Take) -> str:
    c = contrast(t.lang, t.contrast)
    if t.kind == "best":
        return f"Say **{t.word}** as well as you can."
    return f"Now say **{t.word}** {c.error}."
