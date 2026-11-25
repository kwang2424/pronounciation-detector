"""Tests for Phase 2: audio hygiene, acoustic vowel duration/formants, German allophony, and HVPT pairs."""
import numpy as np
import pytest

from mdd import acoustics, audio, pipeline, recognizer
from mdd.diagnose import tip_for
from mdd.languages import get
from mdd.validate import check_language


def _harmonic_vowel(f1: float, f2: float, f3: float, dur_s: float, sr: int = 16000, f0: float = 130.0) -> np.ndarray:
    n = max(1, int(round(sr * dur_s)))
    t = np.arange(n) / sr
    # Pulse train from harmonics up to Nyquist
    pulse = np.zeros(n, dtype=np.float64)
    k = 1
    while k * f0 < sr * 0.45:
        pulse += np.sin(2.0 * np.pi * k * f0 * t)
        k += 1

    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    z_inv = np.exp(-2j * np.pi * freqs / sr)
    h = np.ones_like(z_inv, dtype=complex)
    for f, bw in ((f1, 60.0), (f2, 80.0), (f3, 120.0)):
        r = np.exp(-np.pi * bw / sr)
        w = 2.0 * np.pi * f / sr
        a1 = -2.0 * r * np.cos(w)
        a2 = r * r
        h = h / (1.0 + a1 * z_inv + a2 * (z_inv ** 2))
    out = np.fft.irfft(np.fft.rfft(pulse) * h, n=n)
    peak = np.abs(out).max()
    return (0.28 * out / peak).astype(np.float32) if peak > 0 else out.astype(np.float32)


def _high_freq_consonant(dur_s: float, sr: int = 16000, seed: int = 0) -> np.ndarray:
    n = max(1, int(round(sr * dur_s)))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spec = np.fft.rfft(noise)
    spec[freqs < 3200.0] = 0.0
    out = np.fft.irfft(spec, n=n)
    peak = np.abs(out).max()
    return (0.20 * out / peak).astype(np.float32) if peak > 0 else out.astype(np.float32)


# ------------------------------------------------------------ Audio hygiene
def test_preprocess_recording_rejects_empty_silent_short_and_noisy():
    sr = 16000
    assert audio.preprocess_recording(np.array([], dtype=np.float32), sr).reject_reason == "empty"
    assert audio.preprocess_recording(np.zeros(sr, dtype=np.float32), sr).reject_reason == "silent"
    tiny = _harmonic_vowel(500, 1500, 2500, 0.04, sr)
    assert audio.preprocess_recording(tiny, sr).reject_reason == "too-short"

    # Stationary broadband noise with low SNR
    rng = np.random.default_rng(7)
    noisy = (0.12 * rng.standard_normal(sr)).astype(np.float32)
    assert audio.preprocess_recording(noisy, sr).reject_reason == "low-snr"


def test_preprocess_recording_trims_silence_keeps_cropped_and_warns_clipping():
    sr = 16000
    speech = _harmonic_vowel(700, 1200, 2500, 0.35, sr)

    # Leading/trailing 1.5 s dead air (> 1.2 s) gets trimmed cleanly
    sil_long = np.zeros(int(sr * 1.5), dtype=np.float32)
    padded = np.concatenate([sil_long, speech, sil_long])
    res = audio.preprocess_recording(padded, sr)
    assert res.ok
    assert res.duration_s < 1.20
    assert res.voiced_s >= 0.30

    # Normal 0.85 s trailing silence (<= 1.2 s) is preserved untouched
    normal = np.concatenate([speech, np.zeros(int(sr * 0.85), dtype=np.float32)])
    keep = audio.preprocess_recording(normal, sr)
    assert keep.ok
    assert len(keep.wav) == len(normal)

    # Tightly cropped clean harmonic speech must NOT be rejected as low-SNR
    cropped = audio.preprocess_recording(speech, sr)
    assert cropped.ok

    # Clipped recording stays usable (if voiced) and emits a clipping warning
    clipped = np.clip(speech * 10.0, -1.0, 1.0).astype(np.float32)
    clip_res = audio.preprocess_recording(clipped, sr)
    assert clip_res.ok
    assert any("clipping" in w.lower() for w in clip_res.warnings)


def test_app_run_raises_gr_error_on_silent_recording(monkeypatch, tmp_path):
    import gradio as gr
    import soundfile as sf

    import app

    class _StubLoader:
        load_audio = staticmethod(recognizer.PhoneRecognizer.load_audio)

    monkeypatch.setattr(app, "get_recognizer", lambda: _StubLoader())
    wav_path = tmp_path / "silent.wav"
    sf.write(str(wav_path), np.zeros(16000, dtype=np.float32), 16000)

    with pytest.raises(gr.Error, match="No speech detected"):
        app.run("German", "Ich bin gut", str(wav_path), "", -1.0)


# ---------------------------------------------- Acoustic formants & duration
def test_estimate_formants_tracks_synthetic_a_and_y():
    sr = 16000
    a_wav = _harmonic_vowel(700.0, 1200.0, 2500.0, 0.12, sr)
    y_wav = _harmonic_vowel(300.0, 2000.0, 2600.0, 0.12, sr)

    f1_a, f2_a = acoustics.estimate_formants(a_wav, sr)
    f1_y, f2_y = acoustics.estimate_formants(y_wav, sr)

    assert f1_a is not None and 580.0 <= f1_a <= 820.0
    assert f2_a is not None and 1050.0 <= f2_a <= 1350.0
    assert f1_y is not None and 200.0 <= f1_y <= 420.0
    assert f2_y is not None and 1800.0 <= f2_y <= 2200.0


def test_measure_vowel_measures_short_vs_long_cvc_durations():
    sr = 16000
    c1 = _high_freq_consonant(0.10, sr, seed=1)
    c2 = _high_freq_consonant(0.10, sr, seed=2)
    short_v = _harmonic_vowel(700, 1200, 2500, 0.075, sr)
    long_v = _harmonic_vowel(700, 1200, 2500, 0.210, sr)

    short_wav = np.concatenate([c1, short_v, c2])
    long_wav = np.concatenate([c1, long_v, c2])

    segs_short = [
        recognizer.Segment("ʃ", 0.02, 0.04, 0.0),
        recognizer.Segment("t", 0.05, 0.07, 0.0),
        recognizer.Segment("aː", 0.13, 0.15, 0.0),
        recognizer.Segment("t", 0.21, 0.23, 0.0),
    ]
    segs_long = [
        recognizer.Segment("ʃ", 0.02, 0.04, 0.0),
        recognizer.Segment("t", 0.05, 0.07, 0.0),
        recognizer.Segment("aː", 0.20, 0.22, 0.0),
        recognizer.Segment("t", 0.35, 0.37, 0.0),
    ]

    m_short = acoustics.measure_vowel(short_wav, sr, segs_short, 2)
    m_long = acoustics.measure_vowel(long_wav, sr, segs_long, 2)

    assert m_short is not None and m_short.bounded and m_short.duration_ms < 95.0
    assert m_long is not None and m_long.bounded and m_long.duration_ms > 180.0
    assert m_short.f1_hz is not None and m_short.f2_hz is not None


class _MockLengthRecognizer:
    def __init__(self, wav: np.ndarray, spikes: list[tuple[str, float]], segs: list[recognizer.Segment]):
        self._wav = wav
        self._spikes = spikes
        self._segs = segs

    def load_audio(self, path):
        return self._wav

    def log_probs(self, wav):
        return None

    def greedy_spikes(self, logp):
        return list(self._spikes)

    def gop(self, logp, canon_tokens):
        return list(self._segs)


def test_pipeline_recovers_german_staat_and_huete_vowel_length():
    from eval.common import is_flagged

    sr = 16000
    pad = np.zeros(int(sr * 0.15), dtype=np.float32)
    c1 = _high_freq_consonant(0.10, sr, seed=3)
    c2 = _high_freq_consonant(0.10, sr, seed=4)

    # 1) "Staat" spoken with a short 75 ms /a/ while recognizer emits length-folded "a"
    v_short_a = _harmonic_vowel(700, 1200, 2500, 0.075, sr)
    wav_staat_short = np.concatenate([pad, c1, v_short_a, c2, pad])
    segs_staat_short = [
        recognizer.Segment("ʃ", 0.16, 0.18, 0.0),
        recognizer.Segment("t", 0.20, 0.22, 0.0),
        recognizer.Segment("aː", 0.28, 0.30, 0.0),
        recognizer.Segment("t", 0.37, 0.39, 0.0),
    ]
    rec_short_a = _MockLengthRecognizer(
        wav_staat_short,
        [("ʃ", 0.99), ("t", 0.99), ("a", 0.99), ("t", 0.99)],
        segs_staat_short,
    )
    rep_short = pipeline.analyse("Staat", "unused.wav", rec_short_a, lang="de")
    flagged_a = [p for p in rep_short["phones"] if p["flagged"]]
    assert len(flagged_a) == 1
    assert flagged_a[0]["canonical"] == "aː"
    assert flagged_a[0]["realized"] == "a"
    assert "too short" in flagged_a[0]["tip"]
    assert flagged_a[0]["acoustics"]["duration_flagged"] is True
    assert is_flagged(flagged_a[0], -2.0) is True

    # 2) "Staat" spoken with a held 210 ms /aː/ passes cleanly
    v_long_a = _harmonic_vowel(700, 1200, 2500, 0.210, sr)
    wav_staat_long = np.concatenate([pad, c1, v_long_a, c2, pad])
    segs_staat_long = [
        recognizer.Segment("ʃ", 0.16, 0.18, 0.0),
        recognizer.Segment("t", 0.20, 0.22, 0.0),
        recognizer.Segment("aː", 0.34, 0.36, 0.0),
        recognizer.Segment("t", 0.50, 0.52, 0.0),
    ]
    rec_long_a = _MockLengthRecognizer(
        wav_staat_long,
        [("ʃ", 0.99), ("t", 0.99), ("a", 0.99), ("t", 0.99)],
        segs_staat_long,
    )
    rep_long = pipeline.analyse("Staat", "unused.wav", rec_long_a, lang="de")
    assert [p for p in rep_long["phones"] if p["flagged"]] == []

    # 3) "Hüte" spoken with a short 75 ms /y/ (despite ʏ->y fold) is recovered & flagged
    v_short_y = _harmonic_vowel(300, 2000, 2600, 0.075, sr)
    v_schwa = _harmonic_vowel(500, 1500, 2500, 0.070, sr)
    wav_huete_short = np.concatenate([pad, c1, v_short_y, c2, v_schwa, pad])
    segs_huete_short = [
        recognizer.Segment("h", 0.19, 0.21, 0.0),
        recognizer.Segment("yː", 0.28, 0.30, 0.0),
        recognizer.Segment("t", 0.37, 0.39, 0.0),
        recognizer.Segment("ə", 0.48, 0.50, 0.0),
    ]
    rec_huete = _MockLengthRecognizer(
        wav_huete_short,
        [("h", 0.99), ("y", 0.99), ("t", 0.99), ("ə", 0.99)],
        segs_huete_short,
    )
    rep_huete = pipeline.analyse("Hüte", "unused.wav", rec_huete, lang="de")
    flagged_y = [p for p in rep_huete["phones"] if p["flagged"]]
    assert len(flagged_y) == 1
    assert flagged_y[0]["canonical"] == "yː"
    assert flagged_y[0]["realized"] == "y"
    assert "rounded lips" in flagged_y[0]["tip"]

    # 4) Phrase-final short vowel ("da") held 250 ms is NOT flagged for over-holding
    v_da = _harmonic_vowel(700, 1200, 2500, 0.250, sr)
    wav_da = np.concatenate([pad, c1, v_da, pad])
    segs_da = [
        recognizer.Segment("d", 0.19, 0.21, 0.0),
        recognizer.Segment("a", 0.36, 0.38, 0.0),
    ]
    rec_da = _MockLengthRecognizer(wav_da, [("d", 0.99), ("a", 0.99)], segs_da)
    rep_da = pipeline.analyse("da", "unused.wav", rec_da, lang="de")
    assert [p for p in rep_da["phones"] if p["flagged"]] == []

    # 5) Short unstressed /iː/ ("die", 60 ms) records acoustics but is NOT duration-flagged
    v_die = _harmonic_vowel(300, 2200, 2800, 0.060, sr)
    wav_die = np.concatenate([pad, c1, v_die, pad])
    segs_die = [
        recognizer.Segment("d", 0.19, 0.21, 0.0),
        recognizer.Segment("iː", 0.27, 0.29, 0.0),
    ]
    rec_die = _MockLengthRecognizer(wav_die, [("d", 0.99), ("i", 0.99)], segs_die)
    rep_die = pipeline.analyse("die", "unused.wav", rec_die, lang="de")
    assert [p for p in rep_die["phones"] if p["flagged"]] == []
    assert rep_die["phones"][1]["acoustics"] is not None
    assert "duration_flagged" not in rep_die["phones"][1]["acoustics"]

    # 6) 60 ms /yː/ before coda /ʁ/ ("führt") is NOT duration-flagged
    v_fuehrt = _harmonic_vowel(300, 2000, 2600, 0.060, sr)
    wav_fuehrt = np.concatenate([pad, c1, v_fuehrt, c2, pad])
    segs_fuehrt = [
        recognizer.Segment("f", 0.16, 0.18, 0.0),
        recognizer.Segment("yː", 0.27, 0.29, 0.0),
        recognizer.Segment("ʁ", 0.34, 0.36, 0.0),
        recognizer.Segment("t", 0.39, 0.41, 0.0),
    ]
    rec_fuehrt = _MockLengthRecognizer(
        wav_fuehrt,
        [("f", 0.99), ("y", 0.99), ("ɐ", 0.99), ("t", 0.99)],
        segs_fuehrt,
    )
    rep_fuehrt = pipeline.analyse("führt", "unused.wav", rec_fuehrt, lang="de")
    assert [p for p in rep_fuehrt["phones"] if p["flagged"]] == []


# ----------------------------------------- German allophony & HVPT pairs
def test_german_contextual_allophony_and_tips():
    # [χ] allophone of /x/ accepted
    assert [p for p in pipeline.analyse("Bach", realized_ipa="baχ", lang="de")["phones"] if p["flagged"]] == []
    # Word-final -ig as [ɪk] accepted
    assert [p for p in pipeline.analyse("König", realized_ipa="køːnɪk", lang="de")["phones"] if p["flagged"]] == []
    assert [p for p in pipeline.analyse("richtig", realized_ipa="ʁɪçtɪk", lang="de")["phones"] if p["flagged"]] == []

    # Non-(-ig) ç -> k remains flagged!
    ich_flags = [p for p in pipeline.analyse("Ich", realized_ipa="ɪk", lang="de")["phones"] if p["flagged"]]
    assert [(p["canonical"], p["realized"]) for p in ich_flags] == [("ç", "k")]

    # Non-final -ig / inflection? Wait: möchte has no -ig and remains flagged
    moechte_flags = [p for p in pipeline.analyse("möchte", realized_ipa="mɔktə", lang="de")["phones"] if p["flagged"]]
    assert ("ç", "k") in [(p["canonical"], p["realized"]) for p in moechte_flags]

    # /yː/ unrounding tip
    tuer_flags = [p for p in pipeline.analyse("Tür", realized_ipa="tiːʁ", lang="de")["phones"] if p["flagged"]]
    assert len(tuer_flags) == 1 and "round your lips" in tuer_flags[0]["tip"]
    assert "voiceless /s/" in tip_for("s", "z", "de")


def test_german_hvpt_contrasts_are_true_usable_minimal_pairs():
    profile = get("de")
    all_pairs = {pair for c in profile.contrasts for pair in c.pairs}
    for bad in (("Mus", "müssen"), ("ich", "Asche"), ("Bücher", "Buch"), ("dich", "Dach"), ("rot", "Uhr"), ("Rand", "Vater")):
        assert bad not in all_pairs

    reports = check_language(profile, audio=True)
    for rep in reports:
        assert rep.usable, rep.summary()
        assert all(c.usable for c in rep.checks), rep.summary()
