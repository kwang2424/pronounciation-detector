"""Can a better voice render the contrasts espeak cannot? Point the gate at any TTS.

espeak is a formant synthesiser following hand-written rules, so it fails on
anything its rules do not encode -- Danish stod above all. A *neural* voice is
trained on real recordings of the language, so it may reproduce such features
without anyone having modelled them. That is a hypothesis, not a promise, and
this script is how you test it rather than assume either way.

  python -m eval.tts_probe da                       # Danish, default neural voices
  python -m eval.tts_probe da --contrast stod       # just the gated one
  python -m eval.tts_probe da --voices da-DK-ChristelNeural da-DK-JeppeNeural

Needs internet (edge-tts calls Microsoft's endpoint). What to look for:

* A separation ratio well above 1.35 on a contrast espeak gates means that voice
  renders a difference espeak could not, and the contrast becomes trainable with
  that backend.
* Ratios near 1.0 mean the voice collapses the pair too -- the same verdict
  espeak got, now with better evidence.

One caveat carried over from the espeak work, and it is the important one: this
measures whether the audio *differs*, never whether it differs *correctly*.
espeak can be forced to "distinguish" Danish stod by emitting a full glottal stop
-- 4.8x separation, and phonetically wrong, since stod is creaky voice with
phonation continuing, not a silent gap. A high ratio here is a reason to *listen*
to the clips (they are kept, and the path is printed), not to flip `hvpt_ready`.
"""
import argparse
import tempfile
from pathlib import Path

from mdd.languages import get
from mdd.validate import SEPARATION_THRESHOLD, check_contrast

from .tts import synth_missing

DEFAULT_VOICES = {
    "da": ["da-DK-ChristelNeural", "da-DK-JeppeNeural"],
    "de": ["de-DE-KatjaNeural", "de-DE-ConradNeural"],
    "ko": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"],
}


def make_renderer(outdir: Path):
    """Render a word with a TTS voice, caching clips on disk so they can be listened to.

    Voices are `eval.tts` specs, so anything that module speaks works here:
    `edge:<name>` (neural, needs internet), `sapi:<name>` (Windows), `espeak`.
    Reusing it also inherits its retry-and-backoff on flaky network calls.
    """
    import soundfile as sf

    cache: dict[tuple[str, str], list[int]] = {}

    def render(word: str, lang: str, talker):
        voice = str(talker) if ":" in str(talker) else f"edge:{talker}"
        key = (word, voice)
        if key in cache:
            return cache[key]
        safe = voice.replace(":", "-")
        path = outdir / f"{lang}-{safe}-{word}.wav"
        if not path.exists():
            synth_missing(voice, [(word, path)])
        data, _ = sf.read(str(path), dtype="int16")
        if data.ndim > 1:
            data = data[:, 0]
        cache[key] = list(data)
        return cache[key]

    return render


def main():
    ap = argparse.ArgumentParser(prog="eval.tts_probe", description=__doc__.split("\n")[0])
    ap.add_argument("lang", nargs="?", default="da")
    ap.add_argument("--contrast", help="only this contrast id (default: all)")
    ap.add_argument("--voices", nargs="*",
                    help="voice specs, e.g. da-DK-JeppeNeural or edge:<name> / sapi:<name> / espeak")
    ap.add_argument("--outdir", default=None, help="where to keep the clips (default: a temp dir)")
    args = ap.parse_args()

    profile = get(args.lang)
    voices = args.voices or DEFAULT_VOICES.get(args.lang)
    if not voices:
        ap.error(f"no default voices for {args.lang!r}; pass --voices")
    outdir = Path(args.outdir) if args.outdir else Path(tempfile.mkdtemp(prefix="tts-probe-"))
    outdir.mkdir(parents=True, exist_ok=True)

    contrasts = [profile.contrast(args.contrast)] if args.contrast else list(profile.contrasts)
    render = make_renderer(outdir)

    print(f"{profile.name} · voices: {', '.join(voices)}")
    print(f"clips kept in {outdir} — listen before trusting any number below\n")
    for contrast in contrasts:
        # reps=1: neural TTS is deterministic, so the same-word floor is ~0 and
        # ratios come out far larger than espeak's. Compare against the threshold,
        # not against espeak's numbers.
        report = check_contrast(contrast, profile, audio=True, render=render, talkers=voices)
        good = len(report.usable_pairs)
        print(f"[{'OK  ' if report.usable else 'SKIP'}] {contrast.id}: {good}/{len(report.checks)} pairs separable")
        for c in report.checks:
            sep = "  n/a" if c.separation is None else f"{c.separation:6.2f}x"
            mark = " " if c.usable else "x"
            print(f"      {mark} {c.word_a:8s}/{c.word_b:8s} sep={sep}  (gate {SEPARATION_THRESHOLD})")
    print("\nA high ratio says the audio differs, not that it differs correctly.")
    print("Listen to the clips before enabling a contrast espeak had gated.")


if __name__ == "__main__":
    main()
