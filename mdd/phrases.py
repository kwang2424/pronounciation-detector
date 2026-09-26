"""Practice phrases: short sentences, each aimed at one or two sounds.

Typing your own sentences works, but a good practice sentence packs the target
sound in several times and avoids sounds the recogniser is known to mishear, and
that is hard to do on the spot. Every phrase here lists the phones it targets,
and `tests/test_phrases.py` checks that the pronunciation dictionary (espeak)
really produces them. That check is not a formality: candidates were dropped
because espeak gives French *rose* as [ʁɔz] and *jeunes* as [ʒøn], and a
reference voice reading those would teach the wrong vowel.

`contrasts` links a phrase to the perception tab's contrast ids, so the loop the
research supports — learn to hear a contrast, then produce it — is explicit.

Phrases marked `by_ear` target something the scorer cannot judge (vowel length is
a measured blind spot), so they are for listening and imitating with the
reference audio, not for scoring.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phrase:
    text: str
    #: What to listen for, in plain words.
    focus: str
    group: str
    #: Phones that must appear in the phrase's expected pronunciation.
    targets: tuple[str, ...]
    #: Perception-tab contrast ids this phrase exercises.
    contrasts: tuple[str, ...] = ()
    #: Why the score will not help, when it will not.
    by_ear: str | None = None

    @property
    def label(self) -> str:
        mark = "👂 " if self.by_ear else ""
        return f"{mark}{self.group} · {self.text} — {self.focus}"


_LENGTH = "vowel length is a blind spot: the score won't flag it, so compare by ear"

PHRASES: dict[str, tuple[Phrase, ...]] = {
    "de": (
        Phrase("Die Tür ist grün.", "long ü twice", "ü and ö",
               ("yː",), ("front-rounded",)),
        Phrase("Schöne Grüße aus Köln.", "long ö, long ü, short ö", "ü and ö",
               ("øː", "yː", "œ"), ("front-rounded",)),
        Phrase("Ich möchte fünf Brötchen.", "short ö, short ü, long ö, and ich",
               "ü and ö", ("œ", "y", "øː", "ç"), ("front-rounded", "ich-ach")),
        Phrase("Können Sie mir helfen?", "short ö", "ü and ö",
               ("œ",), ("front-rounded",)),
        Phrase("Über die Brücke.", "long ü, then short ü", "ü and ö",
               ("yː", "y"), ("front-rounded",)),
        Phrase("Ich mache das auch noch.", "front ich, then three back ach",
               "ich and ach", ("ç", "x"), ("ich-ach",)),
        Phrase("Milch und Kuchen.", "front after l, back after u", "ich and ach",
               ("ç", "x"), ("ich-ach",)),
        Phrase("Das Buch ist nicht schlecht.", "back, front, sch, front",
               "ich and ach", ("x", "ç", "ʃ"), ("ich-ach",)),
        Phrase("Heute ist ein neuer Tag.", "eu = 'oy', ei = 'eye'", "diphthongs",
               ("ɔʏ", "aɪ")),
        Phrase("Meine Freunde sind weit weg.", "ei and eu, plus w = v", "diphthongs",
               ("aɪ", "ɔʏ", "v")),
        Phrase("Zehn Katzen sitzen zu Hause.", "z = 'ts' four times", "z",
               ("ts",)),
        Phrase("Wir wohnen in Wien.", "w = English v", "w",
               ("v",)),
        Phrase("Der Student spielt Sport.", "'sht' and 'shp'", "st and sp",
               ("ʃ",)),
        Phrase("Rote Rosen riechen gut.", "uvular r at the start of a word", "r",
               ("ʁ",), ("r-uvular",)),
        Phrase("Der Staat und die Stadt.", "long a, then short a", "vowel length",
               ("aː", "a"), ("vowel-length",), by_ear=_LENGTH),
        Phrase("Der Ofen ist offen.", "long o, then short o", "vowel length",
               ("oː", "ɔ"), ("vowel-length",), by_ear=_LENGTH),
        Phrase("Der Hund ist halb wach.", "d and b said as t and p at the end",
               "final devoicing", ("t", "p"),
               by_ear="final devoicing is not yet measured, so treat a flag as a hint"),
    ),
    "fr": (
        Phrase("Tu as vu la rue ?", "u /y/ three times", "u and ou",
               ("y",), ("u-vs-ou",)),
        Phrase("Où est la rue du Louvre ?", "ou /u/ against u /y/", "u and ou",
               ("u", "y"), ("u-vs-ou",)),
        Phrase("Nous voulons du jus.", "ou, ou, then u, u", "u and ou",
               ("u", "y"), ("u-vs-ou",)),
        Phrase("Un bon vin blanc.", "all three nasal vowels", "nasal vowels",
               ("ɛ̃", "ɔ̃", "ɑ̃"), ("nasal-vowels",)),
        Phrase("Mon oncle Jean a faim.", "on, an, in", "nasal vowels",
               ("ɔ̃", "ɑ̃", "ɛ̃"), ("nasal-vowels",)),
        Phrase("Elle vend du pain.", "an, then in", "nasal vowels",
               ("ɑ̃", "ɛ̃"), ("nasal-vowels",)),
        Phrase("Cent cinquante.", "an, in, an", "nasal vowels",
               ("ɑ̃", "ɛ̃"), ("nasal-vowels",)),
        Phrase("C'est beau, c'est bon.", "oral o, then nasal on", "nasal vs oral",
               ("o", "ɔ̃"), ("nasal-vs-oral",)),
        Phrase("Le pain et la peine.", "nasal in, then oral è + n", "nasal vs oral",
               ("ɛ̃", "ɛ"), ("nasal-vs-oral",)),
        Phrase("Une grosse pomme.", "close o, then open o", "o and ɔ",
               ("o", "ɔ"), ("mid-vowels",)),
        Phrase("La robe jaune.", "open o, then close o", "o and ɔ",
               ("ɔ", "o"), ("mid-vowels",)),
        Phrase("Il fait chaud dehors.", "close o, then open o", "o and ɔ",
               ("o", "ɔ"), ("mid-vowels",)),
        Phrase("J'ai été très fatigué.", "é four times, è once", "é and è",
               ("e", "ɛ"), ("e-vs-e-grave",)),
        Phrase("Mon père préfère le café.", "è, é + è, é", "é and è",
               ("ɛ", "e"), ("e-vs-e-grave",)),
        Phrase("Un peu de beurre.", "closed eu, then open eu", "eu",
               ("ø", "œ")),
        Phrase("Il pleut, j'ai peur.", "closed eu, then open eu", "eu",
               ("ø", "œ")),
        Phrase("Le train rouge arrive à Paris.", "French r, four times", "r",
               ("ʁ",)),
    ),
}

#: Shown whenever a phrase in that language is picked.
CAVEATS = {
    "fr": ("French scoring is still unreliable for nasal vowels and u /y/ (it "
           "flags native speakers too), so lean on 🔊 for those."),
}


def phrases(lang: str) -> tuple[Phrase, ...]:
    return PHRASES.get(lang, ())


def find(lang: str, text: str) -> Phrase | None:
    text = (text or "").strip()
    return next((p for p in phrases(lang) if p.text == text), None)


def for_sound(lang: str, phone: str) -> list[Phrase]:
    """Scoreable phrases that target a phone, for suggesting what to practise."""
    return [p for p in phrases(lang) if phone in p.targets and not p.by_ear]
