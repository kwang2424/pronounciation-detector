"""Use recorded native talkers instead of a synthesiser.

Synthesis is settled as a dead end for Danish stød: espeak can only be forced
into a glottal stop, and neural voices do not render it either (one creaked on
three quarters of all words, the other on a non-stød word). Recording is the
route, and the amount needed is small — a minimal-pair list from six to eight
speakers is a few minutes of audio.

This module turns a directory of clips into the same `render` callable
`mdd.validate` and `mdd.hvpt` already take, so however the audio was obtained —
recorded to order, pulled from a word-level pronunciation source, excised from a
corpus — it drops in without touching anything else.

Layout, either shape (the first is easier to record into):

    recordings/da/anna/hund.wav          <root>/<lang>/<talker>/<word>.wav
    recordings/da/hund__anna.wav         <root>/<lang>/<word>__<talker>.wav

`coverage()` reports which contrast words actually have talkers, because a
half-covered contrast is the failure mode here: trials silently narrow to the
words that happen to exist, which reintroduces exactly the item-memorisation
HVPT exists to prevent.

A note on provenance, since it decides whether the audio is any good: **citation
form is what you want**. Stød is most reliably realised on a stressed syllable in
an isolated word, and is weakened or absent when the word is unstressed in
running speech. Clips excised from continuous audio are therefore the least
reliable source for the one contrast that needs them most.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3")
TALKER_SEP = "__"
#: Enough distinct talkers for the variability HVPT depends on. Single-talker
#: training produces gains that do not generalise (Lively et al. 1993).
MIN_TALKERS = 3


def default_root() -> Path:
    """`$MDD_STIMULI` if set, else ~/.mdd/stimuli — beside the progress file."""
    import os

    env = os.environ.get("MDD_STIMULI")
    return Path(env) if env else Path.home() / ".mdd" / "stimuli"


def _key(word: str) -> str:
    """Normalise for lookup: macOS stores filenames NFD, so 'bønder' off a Mac
    does not compare equal to the NFC string in the contrast tables."""
    return unicodedata.normalize("NFC", word).casefold()


@dataclass
class Clip:
    word: str
    talker: str
    path: Path


class RecordedTalkers:
    """A directory of recordings, indexed by (word, talker)."""

    def __init__(self, root: Path | str, lang: str):
        self.root = Path(root)
        self.lang = lang
        self.clips: dict[str, dict[str, Path]] = {}
        self._cache: dict[tuple[str, str], list[int]] = {}
        self._scan()

    def _scan(self) -> None:
        base = self.root / self.lang
        if not base.is_dir():
            base = self.root if self.root.is_dir() else None
        if base is None:
            return
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() not in AUDIO_SUFFIXES or not path.is_file():
                continue
            stem = unicodedata.normalize("NFC", path.stem)
            if TALKER_SEP in stem:
                word, _, talker = stem.partition(TALKER_SEP)
            elif path.parent != base:
                word, talker = stem, path.parent.name
            else:
                continue                      # a bare word with no talker: unusable
            self.clips.setdefault(_key(word), {})[talker] = path

    # ------------------------------------------------------------------ query
    @property
    def talkers(self) -> list[str]:
        return sorted({t for by_talker in self.clips.values() for t in by_talker})

    def talkers_for(self, word: str) -> list[str]:
        return sorted(self.clips.get(_key(word), {}))

    def has(self, word: str, talker: str) -> bool:
        return talker in self.clips.get(_key(word), {})

    def __len__(self) -> int:
        return sum(len(v) for v in self.clips.values())

    # ----------------------------------------------------------------- render
    def samples(self, word: str, talker: str) -> list[int]:
        import soundfile as sf

        key = (_key(word), talker)
        if key in self._cache:
            return self._cache[key]
        path = self.clips.get(_key(word), {}).get(talker)
        if path is None:
            raise KeyError(f"no recording of {word!r} by {talker!r} under {self.root}")
        data, _ = sf.read(str(path), dtype="int16")
        if getattr(data, "ndim", 1) > 1:
            data = data[:, 0]
        self._cache[key] = list(data)
        return self._cache[key]

    def audio(self, word: str, talker: str) -> tuple[int, list[int]]:
        """(sample_rate, samples), matching `mdd.synth.synthesize`'s contract."""
        import soundfile as sf

        path = self.clips.get(_key(word), {}).get(talker)
        if path is None:
            raise KeyError(f"no recording of {word!r} by {talker!r} under {self.root}")
        info = sf.info(str(path))
        return info.samplerate, self.samples(word, talker)

    def renderer(self):
        """A `render(word, lang, talker)` callable for `mdd.validate`/`mdd.hvpt`."""

        def render(word: str, _lang: str, talker):
            return self.samples(word, str(talker))

        return render


@dataclass
class WordCoverage:
    word: str
    talkers: list[str]

    @property
    def ok(self) -> bool:
        return len(self.talkers) >= MIN_TALKERS


@dataclass
class ContrastCoverage:
    contrast_id: str
    label: str
    words: list[WordCoverage]

    @property
    def usable(self) -> bool:
        """Every word needs talkers: a partly-covered set silently narrows the
        answer choices, which is item memorisation rather than training."""
        return bool(self.words) and all(w.ok for w in self.words)

    def summary(self) -> str:
        missing = [w.word for w in self.words if not w.talkers]
        thin = [f"{w.word}({len(w.talkers)})" for w in self.words
                if w.talkers and not w.ok]
        if self.usable:
            n = min(len(w.talkers) for w in self.words)
            return f"{self.contrast_id}: ready — every word has {n}+ talkers"
        parts = []
        if missing:
            parts.append("no recordings for " + ", ".join(missing))
        if thin:
            parts.append(f"fewer than {MIN_TALKERS} talkers for " + ", ".join(thin))
        return f"{self.contrast_id}: not ready — " + "; ".join(parts)


def coverage(store: RecordedTalkers, profile) -> list[ContrastCoverage]:
    out = []
    for contrast in profile.contrasts:
        words = [WordCoverage(w, store.talkers_for(w)) for w in contrast.words()]
        out.append(ContrastCoverage(contrast.id, contrast.label, words))
    return out


def wordlist(profile) -> list[str]:
    """Every word a full recording session for this language would need."""
    seen: dict[str, None] = {}
    for contrast in profile.contrasts:
        for word in contrast.words():
            seen.setdefault(word, None)
    return list(seen)


def main():
    import argparse

    ap = argparse.ArgumentParser(
        prog="mdd.recorded",
        description="Print the word list to record, or check what a directory covers.")
    ap.add_argument("lang", nargs="?", default="da")
    ap.add_argument("--root", help="directory of recordings to check")
    ap.add_argument("--script", action="store_true",
                    help="print a recording script to hand to a speaker")
    args = ap.parse_args()

    from .languages import get

    profile = get(args.lang)

    if args.script or not args.root:
        words = wordlist(profile)
        print(f"# {profile.name} recording script — {len(words)} words, "
              f"about 10 minutes per speaker\n")
        print("Say each word ONCE, in isolation, at a natural pace, as if reading a list.")
        print("Citation form matters: an isolated stressed word is where stød is")
        print("reliably realised. Leave a clear gap between words.\n")
        print(f"Save as: <root>/{profile.code}/<your-name>/<word>.wav "
              f"(or <word>__<your-name>.wav)\n")
        for contrast in profile.contrasts:
            print(f"## {contrast.label}")
            for group in contrast.pairs:
                print("   " + "   ".join(group))
            print()
        print(f"Then: python -m mdd.recorded {profile.code} --root <root>")
        return

    store = RecordedTalkers(args.root, profile.code)
    print(f"{len(store)} clips · {len(store.talkers)} talkers: "
          f"{', '.join(store.talkers) or '(none found)'}\n")
    reports = coverage(store, profile)
    for rep in reports:
        print(f"  [{'OK  ' if rep.usable else 'GAPS'}] {rep.summary()}")
    ready = [r.contrast_id for r in reports if r.usable]
    print(f"\n{len(ready)}/{len(reports)} contrasts trainable from these recordings.")
    if ready:
        print("Use them:  Session(lang, recordings=RecordedTalkers(root, lang))")


if __name__ == "__main__":
    main()
