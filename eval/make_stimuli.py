"""Render perception stimuli with neural voices instead of formant synthesis.

The HVPT tab has been driven by espeak, which is a rule-based formant
synthesiser: intelligible enough to validate a contrast mechanically, but thin
and robotic to actually train on. Asking someone to learn /y/ vs /u/ from a voice
that barely has vowels is asking a lot, and it was the first thing a real user
noticed.

Neural TTS is trained on recordings of the language, so its output is far closer
to speech. It is still synthetic — recorded native talkers remain the goal, and
`mdd.recorded` reads whatever ends up in the directory either way — but it is a
large step up for no recording effort.

Talker variability is what makes HVPT work (Lively et al. 1993), and there are
only two to four neural voices per language. Each is therefore rendered at
several speaking rates and pitches, giving six to twelve distinct talkers that
all sound like people.

  python -m eval.make_stimuli fr
  python -m eval.make_stimuli de --root ./stimuli

Writes <root>/<lang>/<talker>/<word>.wav, the layout `mdd.recorded` expects, so
the app picks them up with no further configuration. Needs internet.
"""
if __name__ == "__main__":
    from mdd._utf8 import ensure_utf8_mode
    ensure_utf8_mode()

import argparse  # noqa: E402
import asyncio  # noqa: E402
import io  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

from mdd.languages import get  # noqa: E402
from mdd.recorded import default_root, wordlist  # noqa: E402

from .tts import EDGE_DEFAULT_BY_LANG  # noqa: E402


@dataclass(frozen=True)
class Setting:
    """A prosodic variant of one voice — a distinct 'talker' for training."""

    suffix: str
    rate: str = "+0%"
    pitch: str = "+0Hz"


#: Kept modest: past roughly ±20% rate a neural voice starts to sound processed,
#: which would trade the naturalness this exists to gain. These are also single
#: short words — at +18% "rue" was reported as sounding clipped, closer to "re" —
#: so the fast end is gentler than the slow.
SETTINGS = (
    Setting("", "+0%", "+0Hz"),
    Setting("-slow", "-14%", "-8Hz"),
    Setting("-fast", "+12%", "+10Hz"),
)

#: Silence added around each clip. Neural TTS trims hard on isolated words, and
#: the tail is where French carries nasality — "sans" and "son" were both
#: reported as cut off at the end, which removes the very cue being trained.
PAD_HEAD_MS = 60
PAD_TAIL_MS = 220
#: A raised edge reads as a plosive burst: "haute" was heard as starting with a
#: p or b, though its h is silent and the clip opens on a vowel. Fading the ends
#: removes the transient without touching the speech.
FADE_MS = 12


def talker_name(voice: str, setting: Setting) -> str:
    """A short, filesystem-safe talker id: 'fr-FR-DeniseNeural' -> 'Denise-fast'."""
    core = voice.split("-")[-1].replace("Neural", "") or voice
    return f"{core}{setting.suffix}"


def polish(data, sr: int):
    """Pad and fade a rendered clip.

    Three separate complaints from one training session traced back to the clip
    edges: a word heard as starting with a plosive that has no consonant, a short
    word heard as truncated, and nasal vowels heard as cut off at the end. Padding
    stops the tail being lost and fades stop an abrupt edge reading as a burst.
    """
    import numpy as np

    x = np.asarray(data)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float64)

    fade = max(1, int(sr * FADE_MS / 1000))
    if x.size > 2 * fade:
        x[:fade] *= np.linspace(0.0, 1.0, fade)
        x[-fade:] *= np.linspace(1.0, 0.0, fade)
    head = np.zeros(int(sr * PAD_HEAD_MS / 1000))
    tail = np.zeros(int(sr * PAD_TAIL_MS / 1000))
    return np.concatenate([head, x, tail]).astype(np.int16)


async def _render(text: str, voice: str, setting: Setting, path: Path) -> None:
    import edge_tts
    import soundfile as sf

    # A trailing period makes the voice finish the word as a complete utterance
    # rather than cutting the final phone, which is where French nasality lives.
    spoken = text if text.endswith((".", "!", "?")) else f"{text}."
    for attempt in range(4):
        try:
            buf = io.BytesIO()
            stream = edge_tts.Communicate(spoken, voice, rate=setting.rate,
                                          pitch=setting.pitch).stream()
            async for chunk in stream:
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            data, sr = sf.read(buf, dtype="int16")
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), polish(data, sr), sr)
            return
        except Exception:
            if attempt == 3:
                raise
            await asyncio.sleep(3 * (attempt + 1))


def build(lang: str, root: Path, voices: list[str] | None = None,
          settings=SETTINGS, force: bool = False) -> tuple[int, int]:
    profile = get(lang)
    voices = voices or EDGE_DEFAULT_BY_LANG.get(lang)
    if not voices:
        raise ValueError(f"no neural voices configured for {lang!r}; pass --voices")
    words = wordlist(profile)

    jobs = []
    for voice in voices:
        for setting in settings:
            talker = talker_name(voice, setting)
            for word in words:
                path = root / lang / talker / f"{word}.wav"
                if force or not path.exists():
                    jobs.append((word, voice, setting, path))

    total = len(words) * len(voices) * len(settings)
    if jobs:
        async def run():
            sem = asyncio.Semaphore(4)

            async def one(job):
                async with sem:
                    await _render(job[0], job[1], job[2], job[3])
            await asyncio.gather(*(one(j) for j in jobs))

        asyncio.run(run())
    return len(jobs), total


def main():
    ap = argparse.ArgumentParser(prog="eval.make_stimuli",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("lang", nargs="?", default="fr")
    ap.add_argument("--root", default=None,
                    help=f"where to write clips (default: {default_root()})")
    ap.add_argument("--voices", nargs="*", help="edge-tts voice names")
    ap.add_argument("--force", action="store_true", help="re-render existing clips")
    args = ap.parse_args()

    profile = get(args.lang)
    root = Path(args.root) if args.root else default_root()
    print(f"{profile.name}: rendering {len(wordlist(profile))} contrast words "
          f"into {root}")
    made, total = build(args.lang, root, args.voices, force=args.force)
    print(f"  {made} new clips ({total - made} already present)\n")

    # Neural voices are not exempt from the gate: a contrast is only trainable if
    # THIS audio separates it, which is a different question from whether espeak's did.
    from mdd.recorded import RecordedTalkers, coverage

    store = RecordedTalkers(root, args.lang)
    print(f"{len(store)} clips · {len(store.talkers)} talkers: {', '.join(store.talkers)}\n")
    for rep in coverage(store, profile):
        print(f"  [{'OK  ' if rep.usable else 'GAPS'}] {rep.summary()}")
    print(f"\nThe app uses these automatically for {profile.name} perception training.")


if __name__ == "__main__":
    main()
