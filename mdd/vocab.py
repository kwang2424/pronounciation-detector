"""Is this text at the right level? Coverage of a transcript by the words you know.

Comprehension depends heavily on the share of running words you already know.
Reading research puts comfortable independent reading near 98% (Hu & Nation
2000) and a workable minimum near 95% (Laufer 1989); listening tolerates a little
less, around 90–95% (van Zeeland & Schmitt 2013). Below that, material is
better used for intensive study with a transcript than for the volume of easy
input that drives acquisition.

"Known" is approximated by: the words in your Anki deck (from a plain-text
export), words you mark as known here, and optionally the N most frequent words
of the language (function words like *der, und, le, et* are rarely on cards but
are certainly known). Words are compared as lemmas (simplemma), so *ging*
counts as *gehen*. Separable German verbs (*fängt ... an*) and idioms are
counted word by word, so treat the figure as an estimate.

Unknown words are listed most useful first — by how common they are in the
language (wordfreq's Zipf scale, where 7 is *the* and 3 is once per million
words) — each with its sentence from the text, ready to export for Anki.
"""
from __future__ import annotations

import csv
import html
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Capitalised words rarer than this are probably names. In French a capital
#: mid-sentence is also a name; German capitalises every noun, so German names
#: are left for the learner to mark as known (names count as known in coverage
#: research, since a name rarely blocks understanding).
NAME_ZIPF = 1.5
FRENCH_ELISION = re.compile(r"^(?:[ljmtnsdc]|qu|jusqu|lorsqu|puisqu)['’]", re.IGNORECASE)
WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)
ARTICLES = {
    "de": {"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer"},
    "fr": {"le", "la", "les", "l", "un", "une", "des", "du", "de"},
}


def default_root() -> Path:
    env = os.environ.get("MDD_KNOWN")
    return Path(env) if env else Path.home() / ".mdd" / "known"


# ------------------------------------------------------------------ text
def strip_markup(text: str) -> str:
    """Plain text from pasted transcripts, subtitle files (.srt/.vtt) or HTML."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if (not s or s == "WEBVTT" or s.isdigit() or "-->" in s
                or re.match(r"^(NOTE|STYLE|REGION)\b", s)):
            continue
        lines.append(re.sub(r"\[[^\]]*\]|\{[^}]*\}", " ", s))   # [Musik], {\an8}
    return "\n".join(lines)


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def tokens(text: str, lang: str) -> list[str]:
    """Words, with French elisions split off (l'homme -> l', homme)."""
    out = []
    for m in WORD.finditer(text):
        w = m.group(0)
        if lang == "fr":
            el = FRENCH_ELISION.match(w)
            if el:
                out.append(el.group(0)[:-1])
                w = w[el.end():]
        # German hyphenated compounds count as their parts (Kinder-Garten).
        out.extend(p for p in (re.split(r"-", w) if lang == "de" else [w]) if p)
    return out


def dictionary_form(word: str, lang: str) -> str:
    """simplemma's lemma with its capitalisation (German nouns keep theirs)."""
    import simplemma

    return simplemma.lemmatize(word, lang=lang)


def lemma(word: str, lang: str) -> str:
    """The lemma as a lookup key."""
    return dictionary_form(word, lang).casefold()


def keys(word: str, lang: str) -> set[str]:
    """Every form under which a word can be known: its lemma, itself, and for a
    German past participle the stem without ge- (simplemma leaves 'gefahren' as
    is and turns 'gegessen' into 'weiteressen', so the lemma alone is not enough)."""
    low = word.casefold()
    out = {lemma(word, lang), low}
    if lang == "de" and low.startswith("ge") and len(low) > 5:
        out.add(lemma(low[2:], lang))
        if low.endswith("t"):
            out.add(low[2:-1] + "en")            # gesagt -> sagen, gekauft -> kaufen
    return out


def zipf(word: str, lang: str) -> float:
    from wordfreq import zipf_frequency

    return zipf_frequency(word, lang)


# ------------------------------------------------------------------ known words
class Known:
    """Lemmas you know, kept in ~/.mdd/known/<lang>.txt (one per line)."""

    def __init__(self, lang: str, root: Path | None = None):
        self.lang = lang
        self.path = Path(root or default_root()) / f"{lang}.txt"
        self.words: set[str] = set()
        if self.path.exists():
            self.words = {w.strip() for w in self.path.read_text(encoding="utf-8").splitlines()
                          if w.strip()}

    def add(self, words) -> int:
        """Add words (or whole sentences) as both lemma and the form given."""
        before = len(self.words)
        for w in words:
            for t in tokens(w, self.lang):
                self.words |= {lemma(t, self.lang), t.casefold()}
        return len(self.words) - before

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("\n".join(sorted(self.words)) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)
        return self.path


def read_anki_export(text: str) -> list[list[str]]:
    """Rows of fields from Anki's 'Notes in Plain Text' export (tab-separated,
    '#'-prefixed header lines, fields possibly quoted and holding HTML)."""
    rows = []
    body = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
    for row in csv.reader(io.StringIO(body), delimiter="\t"):
        fields = [re.sub(r"\[sound:[^\]]*\]", " ", strip_markup(f)).strip() for f in row]
        if any(fields):
            rows.append(fields)
    return rows


def words_from_anki(rows: list[list[str]], column: int, mode: str, lang: str) -> list[str]:
    """The entries to learn from one field: the whole field as a word ('word'), or
    every word in it ('sentence'). A leading article is dropped (die Tür -> Tür)."""
    out = []
    for row in rows:
        if column >= len(row) or not row[column]:
            continue
        value = row[column]
        if mode == "word":
            parts = tokens(value, lang)
            while len(parts) > 1 and parts[0].casefold() in ARTICLES.get(lang, ()):
                parts = parts[1:]
            out.extend(parts[:1])
        else:
            out.extend(tokens(value, lang))
    return out


# ------------------------------------------------------------------ analysis
@dataclass
class Unknown:
    lemma: str
    form: str
    count: int
    zipf: float
    sentence: str


@dataclass
class Report:
    lang: str
    total: int                      # counted running words
    known: int
    names: list[str] = field(default_factory=list)
    unknown: list[Unknown] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.known / self.total if self.total else 0.0

    def verdict(self) -> str:
        c = self.coverage
        if c >= 0.98:
            return "comfortable: good for lots of easy reading or listening"
        if c >= 0.95:
            return "workable: understandable with a few lookups — the sweet spot for learning"
        if c >= 0.90:
            return ("hard: fine for listening with the transcript, or intensive study, "
                    "but not as easy input")
        return "too hard for now: pick something easier, or study it line by line"


def analyse(text: str, lang: str, known: set[str], assume_top: int = 0) -> Report:
    """Coverage of `text` by `known` lemmas (plus the `assume_top` most common words)."""
    from wordfreq import top_n_list

    common: set[str] = set()
    for w in top_n_list(lang, assume_top) if assume_top else ():
        common |= {lemma(w, lang), w.casefold()}
    text = strip_markup(text)
    rep = Report(lang, 0, 0)
    seen: dict[str, Unknown] = {}
    names: dict[str, None] = {}
    for sent in sentences(text):
        toks = tokens(sent, lang)
        for i, tok in enumerate(toks):
            low = tok.casefold()
            z = zipf(low, lang)
            if tok[:1].isupper() and (z < NAME_ZIPF or (lang == "fr" and i > 0)):
                names[tok] = None                       # probably a name: not counted
                continue
            ks = keys(tok, lang)
            lem = lemma(tok, lang)
            rep.total += 1
            if ks & known or ks & common:
                rep.known += 1
                continue
            u = seen.get(lem)
            if u is None:
                seen[lem] = Unknown(dictionary_form(tok, lang), tok, 1, z, sent)
            else:
                u.count += 1
    rep.names = list(names)
    rep.unknown = sorted(seen.values(), key=lambda u: (-u.zipf, -u.count, u.lemma))
    return rep


def export_tsv(unknown: list[Unknown]) -> str:
    """Tab-separated: the word as it appears, its dictionary form (a guess), and its
    sentence. Anki's File > Import reads it."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\n")
    for u in unknown:
        w.writerow([u.form, u.lemma, u.sentence])
    return buf.getvalue()
