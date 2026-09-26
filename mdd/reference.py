"""Hear the target: the expected sentence or word, spoken correctly.

The production report says what was expected and what was heard, but a learner
cannot fix a sound from an IPA symbol alone. This renders the expected text as
audio so the two can be compared by ear.

A neural voice is used when the network allows — espeak's formant synthesis was
the first thing a learner complained about in the perception tab — with espeak
as the offline fallback, and the result says which one it was. Clips are cached
on disk, so replaying a sentence costs nothing after the first time.

One voice per language, chosen from the evaluation where there was evidence: for
French, fr-FR-HenriNeural had the lowest disagreement with the canonical
transcription of the voices tested. It is still a single synthetic talker; for a
contrast that one voice renders poorly (a German voice was caught merging Staat
and Stadt) the perception tab's many talkers are the better reference.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path

from .audio import polish
from .languages import get

REFERENCE_VOICES = {
    "de": "de-DE-KatjaNeural",
    "fr": "fr-FR-HenriNeural",     # lowest disagreement in the French native control
    "da": "da-DK-JeppeNeural",     # the other Danish voice creaks on most words
    "ko": "ko-KR-SunHiNeural",
}
#: Slow playback, for hearing a sound you cannot yet pick out at full speed.
SLOW_RATE = "-25%"
NORMAL_WPM, SLOW_WPM = 150, 110
#: Give up on the network after this long and fall back to espeak.
NEURAL_TIMEOUT_S = 12.0

#: After a neural failure, use espeak for this long before trying again — so an
#: offline session does not wait out the timeout on every click, but one network
#: hiccup does not condemn the rest of the session to the robotic voice.
RETRY_AFTER_S = 300.0
#: time.monotonic() of the last neural failure, or None.
_neural_failed_at: float | None = None


def _neural_available() -> bool:
    import time

    return _neural_failed_at is None or time.monotonic() - _neural_failed_at > RETRY_AFTER_S


def default_root() -> Path:
    env = os.environ.get("MDD_REFERENCE")
    return Path(env) if env else Path.home() / ".mdd" / "reference"


@dataclass(frozen=True)
class Reference:
    path: Path
    #: "neural" or "espeak" — shown to the learner, since they sound very different.
    source: str
    voice: str
    slow: bool


def _spoken(text: str) -> str:
    """A complete utterance, so the voice finishes the last sound rather than
    clipping it — the same reason perception stimuli end in a period."""
    text = " ".join(text.split())
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _key(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


async def _edge(text: str, voice: str, rate: str):
    import edge_tts
    import soundfile as sf

    buf = io.BytesIO()
    async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    if not buf.getbuffer().nbytes:
        raise RuntimeError("neural voice returned no audio")
    buf.seek(0)
    return sf.read(buf, dtype="int16")


def _neural_default(text: str, voice: str, rate: str):
    return asyncio.run(asyncio.wait_for(_edge(text, voice, rate), NEURAL_TIMEOUT_S))


def speak(text: str, lang: str = "de", slow: bool = False, root: Path | None = None,
          neural=_neural_default) -> Reference:
    """Render `text` as it should sound, returning a cached wav.

    `neural` is the neural renderer — (text, voice, rate) -> (samples, rate) — or
    None to skip straight to espeak.
    """
    global _neural_failed_at
    import time

    import soundfile as sf

    from . import synth

    profile = get(lang)
    text = _spoken(text)
    root = Path(root) if root is not None else default_root()
    folder = root / profile.code
    folder.mkdir(parents=True, exist_ok=True)

    voice = REFERENCE_VOICES.get(profile.code)
    rate = SLOW_RATE if slow else "+0%"
    if voice and neural is not None and _neural_available():
        path = folder / f"{_key(voice, rate, text)}.wav"
        if path.exists():
            return Reference(path, "neural", voice, slow)
        try:
            data, sr = neural(text, voice, rate)
            sf.write(str(path), polish(data, sr), sr)
            _neural_failed_at = None
            return Reference(path, "neural", voice, slow)
        except Exception:
            _neural_failed_at = time.monotonic()   # offline or blocked: back off

    talker = synth.Talker("", SLOW_WPM if slow else NORMAL_WPM, 50)
    path = folder / f"{_key('espeak', str(talker.wpm), text)}-espeak.wav"
    if not path.exists():
        sr, samples = synth.synthesize(text, profile.synth_voice, talker)
        sf.write(str(path), polish(samples, sr), sr)
    return Reference(path, "espeak", profile.synth_voice, slow)


def describe(ref: Reference) -> str:
    speed = " (slowed down)" if ref.slow else ""
    if ref.source == "neural":
        return f"Correct pronunciation — neural voice {ref.voice}{speed}."
    return (f"Correct pronunciation — espeak{speed}, because the neural voice could not "
            f"be reached. It is robotic; connect to the internet for a natural voice.")
