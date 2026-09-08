"""Synthesis backends for the evaluation harness.

Voice specs:
  edge:<name>    Microsoft neural voices via edge-tts (needs internet), e.g. edge:de-DE-KatjaNeural
  sapi:<name>    Windows built-in voices via System.Speech, e.g. sapi:Microsoft Hedda Desktop
  espeak[:<v>]   espeak-ng text synthesis (offline, robotic), e.g. espeak:fr

`Espeak` also exposes phoneme-level access, which the synthetic-error tier uses to
inject a specific mispronunciation and synthesise the result.
"""
import asyncio
import ctypes
import io
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

EDGE_DEFAULT_BY_LANG = {
    "de": ["de-DE-KatjaNeural", "de-DE-ConradNeural", "de-DE-AmalaNeural", "de-DE-KillianNeural"],
    # Metropolitan French only: the canonical transcription is espeak's fr-fr, and
    # Quebec French differs systematically (diphthongised long vowels, affricated
    # /t d/ before /i y/). A fr-CA voice measures as a 20% disagreement rate that
    # is a dialect mismatch, not a pipeline error.
    "fr": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural", "fr-FR-EloiseNeural"],
    "da": ["da-DK-ChristelNeural", "da-DK-JeppeNeural"],
}
EDGE_DEFAULT = EDGE_DEFAULT_BY_LANG["de"]
SAPI_DEFAULT = "Microsoft Hedda Desktop"
SEP = "\t"   # phoneme separator we ask espeak for; never occurs in its mnemonics


def default_voices(lang: str = "de") -> list[str]:
    names = EDGE_DEFAULT_BY_LANG.get(lang)
    if not names:
        raise ValueError(f"no default TTS voices for {lang!r}; pass --voices")
    v = [f"edge:{n}" for n in names]
    if platform.system() == "Windows" and lang == "de":
        v.append(f"sapi:{SAPI_DEFAULT}")     # the bundled SAPI voice is German
    if lang == "de":
        return v + ["espeak"]        # unchanged spelling keeps German's clip cache valid
    from mdd.languages import get

    return v + [f"espeak:{get(lang).synth_voice}"]


def synth_missing(voice: str, items: list[tuple[str, Path]]) -> None:
    """items: (text, out_path). Synthesise the ones not already on disk."""
    items = [(t, Path(p)) for t, p in items if not Path(p).exists()]
    if not items:
        return
    for _, p in items:
        p.parent.mkdir(parents=True, exist_ok=True)
    kind, _, name = voice.partition(":")
    if kind == "edge":
        _edge(name, items)
    elif kind == "sapi":
        _sapi(name, items)
    elif kind == "espeak":
        # "espeak" alone keeps the German default; "espeak:fr" picks a voice.
        esp = Espeak(name or "de")
        for t, p in items:
            esp.synth_text(t, p)
    else:
        raise ValueError(f"unknown voice spec {voice!r}")


def _edge(name: str, items, concurrency: int = 4) -> None:
    import edge_tts

    async def one(sem, text, path):
        async with sem:
            for attempt in range(4):
                try:
                    buf = io.BytesIO()
                    async for chunk in edge_tts.Communicate(text, name).stream():
                        if chunk["type"] == "audio":
                            buf.write(chunk["data"])
                    buf.seek(0)
                    wav, sr = sf.read(buf, dtype="float32")
                    sf.write(str(path), wav, sr)
                    return
                except Exception:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(3 * (attempt + 1))

    async def main():
        sem = asyncio.Semaphore(concurrency)
        await asyncio.gather(*(one(sem, t, p) for t, p in items))

    asyncio.run(main())


def _sapi(name: str, items) -> None:
    if platform.system() != "Windows":
        raise RuntimeError("sapi voices are Windows-only")
    with tempfile.TemporaryDirectory() as d:
        manifest = Path(d) / "manifest.txt"
        manifest.write_text("".join(f"{p}\t{t}\n" for t, p in items), encoding="utf-8")
        script = Path(d) / "synth.ps1"
        script.write_text(
            "Add-Type -AssemblyName System.Speech\n"
            "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
            "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)\n"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
            f'$s.SelectVoice("{name}")\n'
            "$s.Rate = -1\n"
            f'foreach ($line in Get-Content -Encoding UTF8 "{manifest}") {{\n'
            '  $parts = $line -split "`t", 2\n'
            "  $s.SetOutputToWaveFile($parts[0], $fmt)\n"
            "  $s.Speak($parts[1])\n"
            "}\n"
            "$s.Dispose()\n",
            encoding="utf-8",
        )
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)], check=True)


_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)


class Espeak:
    """espeak-ng via ctypes, using the library bundled in espeakng-loader."""

    def __init__(self, voice: str = "de"):
        import espeakng_loader
        self.lib = ctypes.CDLL(espeakng_loader.get_library_path())
        self.lib.espeak_Initialize.restype = ctypes.c_int
        self.sr = self.lib.espeak_Initialize(2, 0, espeakng_loader.get_data_path().encode(), 0)  # SYNCHRONOUS
        if self.lib.espeak_SetVoiceByName(voice.encode()) != 0:
            raise RuntimeError(f"espeak voice {voice!r} not found")
        self.lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        self.lib.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int]
        self._buf: list[int] = []
        self._cb = _CB(self._collect)
        self.lib.espeak_SetSynthCallback(self._cb)

    def _collect(self, wav, n, events):
        if n > 0:
            self._buf.extend(wav[:n])
        return 0

    def _synth(self, text: str, path: Path, phoneme_input: bool) -> None:
        self._buf.clear()
        flags = 0x01 | 0x1000 | (0x100 if phoneme_input else 0)   # UTF8 | ENDPAUSE | PHONEMES
        t = text.encode("utf-8")
        if self.lib.espeak_Synth(t, len(t) + 1, 0, 1, 0, flags, None, None) != 0:
            raise RuntimeError("espeak_Synth failed")
        x = np.array(self._buf, dtype=np.float32) / 32768.0
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), x, self.sr)

    def synth_text(self, text: str, path: Path) -> None:
        self._synth(text, path, False)

    def synth_phonemes(self, mnemonics: str, path: Path) -> None:
        """`mnemonics` is espeak's own phoneme notation, e.g. "_|IC m'WCt@"."""
        self._synth(f"[[{mnemonics}]]", path, True)

    def _phoneme_strings(self, text: str, ipa: bool) -> list[str]:
        out = []
        p = ctypes.c_char_p(text.encode("utf-8"))
        mode = (0x02 if ipa else 0) | (ord(SEP) << 8)
        while p.value:
            out.append(self.lib.espeak_TextToPhonemes(ctypes.byref(p), 1, mode).decode("utf-8"))
        return out

    def phoneme_grid(self, text: str) -> "PhonemeGrid":
        mn = [[w.split(SEP) for w in c.split(" ") if w] for c in self._phoneme_strings(text, False)]
        ipa = [[w.split(SEP) for w in c.split(" ") if w] for c in self._phoneme_strings(text, True)]
        return PhonemeGrid(mn, ipa)


class PhonemeGrid:
    """espeak's phoneme output for a sentence: clauses -> words -> mnemonic tokens, with the
    parallel IPA for every real phoneme. Pause/glottal markers (mnemonics starting with '_')
    are kept for re-synthesis but have no IPA counterpart."""

    def __init__(self, mn, ipa):
        self.mn = mn
        self.words: list[list[tuple[tuple[int, int, int], str]]] = []   # per word: [(pos, ipa_token)]
        if [len(c) for c in mn] != [len(c) for c in ipa]:
            raise ValueError("clause/word structure differs between mnemonic and IPA output")
        for ci, (cm, cp) in enumerate(zip(mn, ipa)):
            for wi, (wm, wp) in enumerate(zip(cm, cp)):
                real_m = [(ti, t) for ti, t in enumerate(wm) if not t.startswith("_")]
                real_p = [t for t in wp if t]
                if len(real_m) != len(real_p):
                    raise ValueError(f"token count differs: {wm} vs {wp}")
                if real_m:
                    self.words.append([((ci, wi, ti), p) for (ti, _), p in zip(real_m, real_p)])

    @staticmethod
    def split_stress(tok: str) -> tuple[str, str]:
        core = tok.lstrip("',")
        return tok[: len(tok) - len(core)], core

    def mnemonic_at(self, pos) -> str:
        ci, wi, ti = pos
        return self.mn[ci][wi][ti]

    def with_replacement(self, pos, new_mnemonics: str | None) -> str:
        """Render the sentence in espeak phoneme notation with the phoneme at `pos` replaced
        (or deleted when None). Stress on the original phoneme is kept."""
        ci, wi, ti = pos
        out = []
        for c_i, clause in enumerate(self.mn):
            words = []
            for w_i, word in enumerate(clause):
                toks = list(word)
                if (c_i, w_i) == (ci, wi):
                    stress, _ = self.split_stress(toks[ti])
                    if new_mnemonics is None:
                        toks.pop(ti)
                    else:
                        toks[ti] = stress + new_mnemonics
                words.append("".join(toks))
            out.append(" ".join(words))
        return " ".join(out)

    def render(self) -> str:
        return " ".join(" ".join("".join(w) for w in c) for c in self.mn)
