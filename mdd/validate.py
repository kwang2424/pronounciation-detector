"""Does the synthesiser actually render this contrast? Gate trials on the answer.

A perception trial is only answerable if the stimuli genuinely differ. espeak's
coverage is uneven: it renders the Danish soft d faithfully but places stød
unreliably, and it collapses the Korean lenis/fortis contrast for some places of
articulation. Shipping trials over pairs it cannot render would train the learner
to guess and would score them as if they had failed to perceive a real contrast.

Two independent checks:

1. **Transcription** — do the words tokenise to different phone sequences? Cheap
   and deterministic. A pair that fails this is definitely unusable.
2. **Acoustic** — do the rendered waveforms differ by more than the synthesiser's
   own run-to-run jitter? espeak is not deterministic (repeat renders differ in
   length), so the comparison is against a measured same-word noise floor rather
   than a fixed constant. This catches pairs whose transcriptions differ in a
   symbol the voice does not actually realise.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations

from .g2p import text_to_ipa_words
from .languages import Contrast, LanguageProfile, get
from .normalize import tokenize

#: Between-word separation must exceed the same-word noise floor by this factor.
SEPARATION_THRESHOLD = 1.35
#: Floor used when a backend is deterministic (recorded clips, most neural TTS),
#: where repeat renders are identical and the measured jitter is exactly zero.
#: Spectra are unit-normalised, so distances live on a fixed scale: espeak's own
#: jitter measures ~0.09-0.12 and clearly different words ~0.25-0.35. A floor well
#: below the jitter band means "numerically identical", which lets the same ratio
#: and the same threshold describe both kinds of backend.
MIN_FLOOR = 0.02
_FRAMES = 40
_NFFT = 512
_HOP = 128


#: Weight on relative duration in the distance. Spectra are resampled to a fixed
#: frame count so their shape can be compared, which throws duration away — and
#: for a length contrast (Stadt/Staat) duration IS the contrast, so the gate was
#: structurally unable to see a talker who rendered both the same length. A
#: listener found one that did. 0.6 puts a 50% duration difference on a par with
#: a clear spectral difference.
DURATION_WEIGHT = 0.6


def _duration(samples, sr: int = 22050) -> float:
    return len(samples) / sr


def _spectrum(samples):
    import numpy as np

    x = np.asarray(samples, dtype=float)
    if x.size < _NFFT:
        return None
    win = np.hanning(_NFFT)
    frames = [
        np.abs(np.fft.rfft(x[i:i + _NFFT] * win))
        for i in range(0, x.size - _NFFT, _HOP)
    ]
    if len(frames) < 2:
        return None
    S = np.log1p(np.asarray(frames))
    # Resample onto a fixed frame count so different durations are comparable.
    idx = np.linspace(0, len(S) - 1, _FRAMES).astype(int)
    S = S[idx]
    norm = np.linalg.norm(S)
    return S / norm if norm else S


def _distance(a, b, dur_a: float = 0.0, dur_b: float = 0.0) -> float:
    """Spectral shape plus relative duration, so length contrasts are visible."""
    import numpy as np

    spectral = float(np.linalg.norm(a - b))
    longest = max(dur_a, dur_b)
    if longest <= 0:
        return spectral
    relative = abs(dur_a - dur_b) / longest
    return float(np.hypot(spectral, DURATION_WEIGHT * relative))


#: A render backend: (word, lang, talker) -> int16 samples. Swappable so the same
#: honest gate can be pointed at neural TTS or recorded talkers, not just espeak.
Renderer = Callable[[str, str, object], Sequence[int]]


def espeak_renderer(word: str, lang: str, talker) -> Sequence[int]:
    from . import synth

    return synth.synthesize(word, lang, talker)[1]


def _renders(word: str, lang: str, talker, reps: int, render: Renderer):
    """[(spectrum, duration), ...] — duration kept because the spectrum drops it."""
    out = []
    for _ in range(reps):
        samples = render(word, lang, talker)
        spec = _spectrum(samples)
        if spec is not None:
            out.append((spec, _duration(samples)))
    return out


def acoustic_separation(word_a: str, word_b: str, lang: str,
                        reps: int = 3, n_talkers: int = 3,
                        render: Renderer | None = None,
                        talkers: Sequence | None = None,
                        per_talker: bool = False):
    """Between-word distance / same-word distance, averaged over talkers.

    ~1.0 means the pair is indistinguishable from the renderer's own jitter. A
    deterministic backend (recorded clips, most neural TTS) has no jitter, so it
    is scored against `MIN_FLOOR` instead and a real difference yields a much
    larger ratio than espeak ever produces -- compare against the threshold, not
    against espeak's numbers. Returns None if audio is unavailable or the words
    are too short to analyse.

    `render` and `talkers` override the espeak backend; pass a renderer that
    returns samples for a word and a list of whatever your talkers are keyed by.

    `per_talker` returns {talker: ratio} instead of the mean. Whether a pair is
    separable is often a property of one voice rather than of the language: a
    talker who gives Stadt and Staat the same vowel length makes that pair
    unanswerable for that talker alone.
    """
    from . import synth

    if render is None:
        if not synth.available():
            return None
        render = espeak_renderer
    pool = list(talkers) if talkers is not None else list(synth.TALKERS[:n_talkers])
    ratios = []
    for talker in pool:
        ra = _renders(word_a, lang, talker, reps, render)
        rb = _renders(word_b, lang, talker, reps, render)
        if not ra or not rb:
            continue
        within = [_distance(x[0], y[0], x[1], y[1]) for x, y in combinations(ra, 2)]
        within += [_distance(x[0], y[0], x[1], y[1]) for x, y in combinations(rb, 2)]
        between = [_distance(x[0], y[0], x[1], y[1]) for x in ra for y in rb]
        # max() so a deterministic backend (zero jitter) is scored against an
        # absolute floor instead of dividing by zero and reporting "unknown".
        floor = max(sum(within) / len(within) if within else 0.0, MIN_FLOOR)
        ratios.append(((sum(between) / len(between)) / floor, talker))
    if per_talker:
        return {t: r for r, t in ratios}
    return sum(r for r, _ in ratios) / len(ratios) if ratios else None


@dataclass
class PairCheck:
    word_a: str
    word_b: str
    ipa_a: str
    ipa_b: str
    transcription_distinct: bool
    separation: float | None
    #: Whether the transcription is evidence about *this* backend. It is espeak's
    #: G2P, so it settles what espeak will synthesise -- identical phonemes give
    #: identical audio. It says nothing about another backend, and for the pairs
    #: that matter most it is the very thing known to be wrong: espeak collapses
    #: Danish stod, so vetoing a neural voice on that basis would reject audio it
    #: never measured.
    transcription_authoritative: bool = True

    @property
    def usable(self) -> bool:
        if self.separation is None:
            # No audio measured: the transcription is all the evidence there is.
            return self.transcription_distinct
        if self.transcription_authoritative and not self.transcription_distinct:
            return False
        return self.separation >= SEPARATION_THRESHOLD

    def reason(self) -> str:
        if self.separation is None:
            return "ok" if self.transcription_distinct else f"identical transcription /{self.ipa_a}/"
        if self.transcription_authoritative and not self.transcription_distinct:
            return f"identical transcription /{self.ipa_a}/"
        if self.separation < SEPARATION_THRESHOLD:
            return f"audio separation {self.separation:.2f}x is within renderer jitter"
        if not self.transcription_distinct:
            return (f"ok on audio ({self.separation:.2f}x); espeak transcribes both as "
                    f"/{self.ipa_a}/, which says nothing about this backend")
        return "ok"


@dataclass
class ContrastReport:
    contrast_id: str
    label: str
    checks: list[PairCheck]

    @property
    def usable_pairs(self) -> list[PairCheck]:
        return [c for c in self.checks if c.usable]

    @property
    def usable(self) -> bool:
        return len(self.usable_pairs) > 0

    def summary(self) -> str:
        good, total = len(self.usable_pairs), len(self.checks)
        if good == total:
            return f"{self.contrast_id}: all {total} pairs usable"
        bad = {c.reason() for c in self.checks if not c.usable}
        return f"{self.contrast_id}: {good}/{total} pairs usable — " + "; ".join(sorted(bad))


def check_contrast(contrast: Contrast, profile: LanguageProfile,
                   audio: bool = True, render: Renderer | None = None,
                   talkers: Sequence | None = None) -> ContrastReport:
    checks: list[PairCheck] = []
    for group in contrast.pairs:
        ipa = {w: i for w, i in text_to_ipa_words(" ".join(group), profile)}
        for a, b in combinations(group, 2):
            ia, ib = ipa.get(a, ""), ipa.get(b, "")
            distinct = tokenize(ia, profile) != tokenize(ib, profile)
            # Measured even when the transcription already differs: a pair can
            # transcribe distinctly and still render as the same audio.
            sep = (acoustic_separation(a, b, profile.synth_voice,
                                       render=render, talkers=talkers)
                   if audio else None)
            checks.append(PairCheck(a, b, ia, ib, distinct, sep,
                                    transcription_authoritative=render is None))
    return ContrastReport(contrast.id, contrast.label, checks)


def check_language(profile: LanguageProfile | str | None = None,
                   audio: bool = True, render: Renderer | None = None,
                   talkers: Sequence | None = None) -> list[ContrastReport]:
    if not isinstance(profile, LanguageProfile):
        profile = get(profile)
    return [check_contrast(c, profile, audio, render, talkers) for c in profile.contrasts]


def main():
    import argparse

    ap = argparse.ArgumentParser(prog="mdd.validate",
                                 description="Check which contrasts espeak can actually render.")
    ap.add_argument("langs", nargs="*", default=["de", "da", "ko"])
    ap.add_argument("--no-audio", action="store_true", help="transcription check only (fast)")
    args = ap.parse_args()
    for code in args.langs:
        profile = get(code)
        print(f"\n=== {profile.name} ({code}) ===")
        if profile.g2p_caveat:
            print(f"  caveat: {profile.g2p_caveat}\n")
        for rep in check_language(profile, audio=not args.no_audio):
            mark = "OK  " if rep.usable else "SKIP"
            print(f"  [{mark}] {rep.summary()}")
            for c in rep.checks:
                flag = " " if c.usable else "x"
                sep = "  n/a" if c.separation is None else f"{c.separation:5.2f}x"
                print(f"        {flag} {c.word_a:8s}/{c.word_b:8s} "
                      f"/{c.ipa_a}/ vs /{c.ipa_b}/  sep={sep}")


if __name__ == "__main__":
    main()
