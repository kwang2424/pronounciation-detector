"""Multi-talker stimulus synthesis via the bundled espeak-ng shared library.

HVPT works because the learner hears a contrast from *many* talkers, which
forces a category robust to speaker variation rather than memorisation of one
voice (Logan, Lively & Pisoni 1991). espeak-ng ships ~100 voice variants, and
varying variant + rate + pitch gives cheap talker variability with no corpus.

Caveat worth stating plainly: formant-synthesised voices are thinner than the
natural multi-talker recordings used in the HVPT literature, and the training
effects reported there were measured on natural speech. This is a usable
bootstrap, not a replication of those studies. Swap in recorded talkers (e.g.
Common Voice clips) when available — `Talker` and `synthesize` are the only
things a recorded-audio backend has to replace.
"""
from __future__ import annotations

import ctypes
import struct
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

_AUDIO_OUTPUT_SYNCHRONOUS = 0x02
_CHARS_UTF8 = 1

_lib = None
_rate = 0
_lock = threading.Lock()          # espeak's C API is not re-entrant
_buffer: list[int] = []

_SYNTH_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short),
                             ctypes.c_int, ctypes.c_void_p)


def _callback(wav, numsamples, events):
    if wav and numsamples > 0:
        _buffer.extend(wav[i] for i in range(numsamples))
    return 0


_c_callback = _SYNTH_CB(_callback)


class SynthUnavailable(RuntimeError):
    """espeak-ng could not be loaded; perception training can't generate audio."""


def _library_path() -> str:
    import os
    if "PHONEMIZER_ESPEAK_LIBRARY" in os.environ:
        return os.environ["PHONEMIZER_ESPEAK_LIBRARY"]
    import espeakng_loader
    return espeakng_loader.get_library_path()


def _init():
    global _lib, _rate
    if _lib is not None:
        return
    try:
        import espeakng_loader
        lib = ctypes.CDLL(_library_path())
        rate = lib.espeak_Initialize(_AUDIO_OUTPUT_SYNCHRONOUS, 0,
                                     espeakng_loader.get_data_path().encode(), 0)
    except Exception as exc:                                  # pragma: no cover
        raise SynthUnavailable(f"could not load espeak-ng: {exc}") from exc
    if rate <= 0:                                             # pragma: no cover
        raise SynthUnavailable("espeak_Initialize failed")
    lib.espeak_SetSynthCallback(_c_callback)
    _lib, _rate = lib, rate


def available() -> bool:
    try:
        _init()
        return True
    except SynthUnavailable:
        return False


def sample_rate() -> int:
    _init()
    return _rate


@dataclass(frozen=True)
class Talker:
    """One synthetic voice: an espeak variant plus rate and pitch settings."""

    variant: str
    wpm: int = 165
    pitch: int = 50

    def voice_name(self, lang: str) -> str:
        return f"{lang}+{self.variant}" if self.variant else lang


#: A spread of variants, speaking rates and pitches. Male/female variants and a
#: ±25% rate range give the between-talker variation HVPT depends on.
TALKERS: tuple[Talker, ...] = (
    Talker("m1", 150, 40), Talker("f2", 175, 65), Talker("m3", 190, 35),
    Talker("f5", 145, 70), Talker("m7", 165, 45), Talker("f1", 200, 60),
    Talker("m5", 135, 30), Talker("f3", 160, 75),
)


def synthesize(text: str, lang: str, talker: Talker | None = None) -> tuple[int, list[int]]:
    """Render `text` to (sample_rate, int16 samples). Thread-safe, serialised."""
    _init()
    talker = talker or TALKERS[0]
    payload = text.encode("utf-8")
    with _lock:
        _buffer.clear()
        name = talker.voice_name(lang).encode("utf-8")
        if _lib.espeak_SetVoiceByName(name) != 0:
            # Unknown variant: fall back to the plain language voice.
            if _lib.espeak_SetVoiceByName(lang.encode("utf-8")) != 0:
                raise SynthUnavailable(f"espeak has no voice for {lang!r}")
        _lib.espeak_SetParameter(1, ctypes.c_int(talker.wpm), 0)     # espeakRATE
        _lib.espeak_SetParameter(3, ctypes.c_int(talker.pitch), 0)   # espeakPITCH
        rc = _lib.espeak_Synth(payload, len(payload) + 1, 0, 0, 0, _CHARS_UTF8, None, None)
        if rc != 0:                                                  # pragma: no cover
            raise SynthUnavailable(f"espeak_Synth returned {rc}")
        return _rate, list(_buffer)


def write_wav(path: str | Path, rate: int, samples: list[int]) -> str:
    path = str(path)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


def render_to_file(text: str, lang: str, path: str | Path, talker: Talker | None = None) -> str:
    rate, samples = synthesize(text, lang, talker)
    return write_wav(path, rate, samples)
