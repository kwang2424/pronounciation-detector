"""Regression tests for Phase 1 core MDD bug fixes."""
import json
from collections import Counter

import numpy as np
import pytest
import soundfile as sf
import torch

from mdd import acoustics, g2p, hvpt, pipeline, progress, recognizer, reference
from mdd.languages import Contrast, LanguageProfile, get
from mdd.normalize import tokenize, tokenize_with_conf
from mdd.recorded import RecordedTalkers


class _StubTokenizer:
    def __init__(self, vocab: dict[str, int], unk_id: int = 1):
        self._vocab = vocab
        self.unk_token_id = unk_id

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class _StubProcessor:
    def __init__(self, vocab: dict[str, int], unk_id: int = 1):
        self.tokenizer = _StubTokenizer(vocab, unk_id=unk_id)


def _stub_recognizer(vocab: dict[str, int], blank: int = 0, unk_id: int = 1):
    rec = object.__new__(recognizer.PhoneRecognizer)
    rec.processor = _StubProcessor(vocab, unk_id=unk_id)
    rec.blank = blank
    return rec


def test_gop_expands_multichar_tokens_and_avoids_tail_drop():
    vocab = {"<pad>": 0, "<unk>": 1, "t": 2, "s": 3, "p": 4, "f": 5, "a": 6, "ɪ": 7, "z": 8}
    rec = _stub_recognizer(vocab)

    # 12 frames matching: t(0..2) s(2..4) p(4..5) f(5..7) a(7..8) ɪ(8..10) a(10..12)
    # Deliberately make frame 2..4 favour "z" instead of "s" to prove "ts" pools "s".
    logits = torch.full((12, 9), -6.0)
    for t in (0, 1):
        logits[t, 2] = 4.0  # t
    for t in (2, 3):
        logits[t, 8] = 4.0  # z dominates over s
        logits[t, 3] = 0.0  # s is second-best -> -4.0 GOP on tail sub-token
    logits[4, 4] = 4.0      # p
    for t in (5, 6):
        logits[t, 5] = 4.0  # f
    logits[7, 6] = 4.0      # a
    for t in (8, 9):
        logits[t, 7] = 4.0  # ɪ
    for t in (10, 11):
        logits[t, 6] = 4.0  # a (from aː, skipping unmapped ː without <unk>)

    logp = torch.log_softmax(logits, dim=-1)
    segs = rec.gop(logp, ["ts", "pf", "aɪ", "aː"])

    assert [s.token for s in segs] == ["ts", "pf", "aɪ", "aː"]
    assert segs[0].gop < -1.0, "tail sub-token 's' error must affect pooled 'ts' GOP"
    assert segs[1].gop > -0.1
    assert segs[2].gop > -0.1
    assert segs[3].gop > -0.1
    assert all(s.t_end > s.t_start for s in segs)


def test_gop_handles_empty_and_too_short_inputs_without_crashing():
    vocab = {"<pad>": 0, "<unk>": 1, "t": 2, "s": 3}
    rec = _stub_recognizer(vocab)
    logp = torch.log_softmax(torch.zeros((2, 4)), dim=-1)

    assert rec.gop(logp, []) == []
    assert rec.gop(torch.empty((0, 4)), ["t"]) == []
    assert rec.gop(logp, ["t", "s", "t"]) == []
    assert rec.gop(logp, ["t", "t"]) == []  # requires 3 frames with blank between repeats


def test_tokenize_with_conf_and_token_confidences_handle_spikes():
    cases = [
        ("de", [("a", 0.9), ("ː", 0.4)], ["aː"], [0.4]),
        ("de", [("ɔ", 0.8), ("ɪ", 0.5)], ["ɔʏ"], [0.5]),
        ("ko", [("t", 0.9), ("ɕ", 0.7), ("h", 0.6)], ["tɕh"], [0.6]),
        ("fr", [("ɛ", 0.8), ("̃", 0.3)], ["ɛ̃"], [0.3]),
        ("fr", [("(en)", 0.9), ("d", 0.8), ("ɒ", 0.7), ("s", 0.6), ("(fr)", 0.9)],
         ["d", "ɔ", "s"], [0.8, 0.7, 0.6]),
    ]
    for lang, spikes, exp_toks, exp_confs in cases:
        profile = get(lang)
        toks, confs = tokenize_with_conf(spikes, profile)
        assert toks == exp_toks
        assert confs == pytest.approx(exp_confs)
        joined = "".join(t for t, _ in spikes)
        real_tokens = tokenize(joined, profile)
        assert pipeline._token_confidences(spikes, real_tokens, profile) == pytest.approx(exp_confs)


class _FakeRecognizer:
    def __init__(self, spikes, gop_values):
        self._spikes = spikes
        self._gops = gop_values

    def load_audio(self, path):
        return np.zeros(1600, dtype=np.float32)

    def log_probs(self, wav):
        return torch.zeros((10, 4))

    def greedy_spikes(self, logp):
        return list(self._spikes)

    def gop(self, logp, canon_tokens):
        return [
            recognizer.Segment(t, i * 0.02, (i + 1) * 0.02, g)
            for i, (t, g) in enumerate(zip(canon_tokens, self._gops))
        ]


def test_pipeline_insertion_gating_word_attribution_and_overall_clamp():
    # "Singen gut": canonical /z ɪ ŋ ə n | ɡ uː t/
    # Inject a low-conf spurious insertion at start (suppressed) and a high-conf
    # trailing "k" after "Singen" (must blame "Singen", word_idx=0, not "gut").
    spikes = [
        ("t", 0.3),                                           # < INS_MIN_PROB -> suppressed
        ("z", 0.95), ("ɪ", 0.95), ("ŋ", 0.95), ("ə", 0.95), ("n", 0.95),
        ("k", 0.9),                                           # trailing ins on "Singen"
        ("ɡ", 0.95), ("u", 0.95), ("ː", 0.95), ("t", 0.95),
    ]
    fake = _FakeRecognizer(spikes, [-0.1] * 9)
    rep = pipeline.analyse("Singen gut", wav_path="unused.wav", recognizer=fake, lang="de")

    flagged = [p for p in rep["phones"] if p["flagged"]]
    assert len(flagged) == 1
    assert flagged[0]["op"] == "ins"
    assert flagged[0]["realized"] == "k"
    assert flagged[0]["word"] == "Singen"
    assert flagged[0]["word_idx"] == 0

    # Flood with high-conf insertions so flagged count > n_canon -> overall clamps at 0.0
    flood = _FakeRecognizer(
        [("p", 0.95), ("t", 0.95), ("k", 0.95), ("f", 0.95), ("s", 0.95), ("ʃ", 0.95)],
        [-3.0, -3.0, -3.0],
    )
    rep2 = pipeline.analyse("ich", wav_path="unused.wav", recognizer=flood, lang="de")
    assert rep2["overall"] == 0.0


def test_duplicate_words_highlight_only_mispronounced_occurrence():
    import app

    # "Ich bin Ich" — first "Ich" is right, second "Ich" has /k/ instead of /ç/
    summary, highlighted, rows, _ = app.run(
        "German", "Ich bin Ich", None, "ɪç bɪn ɪk", -1.0
    )
    assert highlighted == [("Ich", None), ("bin", None), ("Ich", "check")]
    assert "across 1 word" in summary
    assert len(rows) == 1 and rows[0][0] == "Ich"


def test_acoustics_zero_signal_returns_too_short_not_typeerror():
    res = acoustics.analyse(np.zeros(4410, dtype=np.float32), 22050)
    assert res.kind == "too-short"


def test_g2p_handles_tabs_and_unicode_quotes_keeping_apostrophes():
    words = g2p.text_to_ipa_words("„Hallo!“\tWie geht's? «gut»", "de")
    surface = [w for w, _ in words]
    assert surface == ["Hallo", "Wie", "geht's", "gut"]


def test_reference_serves_cached_neural_clip_while_offline(monkeypatch, tmp_path):
    class _FailingNeural:
        def __call__(self, text, voice, rate):
            raise OSError("offline")

    class _OkNeural:
        def __call__(self, text, voice, rate):
            sr = 24000
            t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
            return (np.sin(2 * np.pi * 200 * t) * 12000).astype(np.int16), sr

    monkeypatch.setattr(reference, "_neural_failed_at", None)
    cached = reference.speak("Guten Morgen", "de", root=tmp_path, neural=_OkNeural())
    assert cached.source == "neural"

    # Trigger offline backoff on an uncached phrase.
    miss = reference.speak("Auf Wiedersehen", "de", root=tmp_path, neural=_FailingNeural())
    assert miss.source == "espeak"

    # Cached neural clip must still be served despite the offline cooldown.
    again = reference.speak("Guten Morgen", "de", root=tmp_path, neural=_FailingNeural())
    assert again.source == "neural"
    assert again.path == cached.path


def _write_tone(path, dur=0.4, freq=220, sr=22050):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    x = (np.sin(2 * np.pi * freq * t) * 20000).astype(np.int16)
    sf.write(str(path), x, sr)


def test_recorded_covers_requires_shared_talker(tmp_path):
    _write_tone(tmp_path / "de" / "anna" / "mad.wav")
    _write_tone(tmp_path / "de" / "bo" / "mat.wav")
    store = RecordedTalkers(tmp_path, "de")
    assert store.shared_talkers(["mad", "mat"]) == []
    assert not store.covers(["mad", "mat"])


def test_hvpt_narrowed_group_preserves_pair_talkers_and_recency_tiebreak(tmp_path):
    # 3-word set ("w1", "w2", "w3"): "w3" merges with "w1" for everyone so the group
    # narrows to ("w1", "w2"). For ("w1", "w2"), only "good" separates them while "flat"
    # merges them. The narrowed group must keep ["good"] and never fall back to ["flat"].
    _write_tone(tmp_path / "xx" / "good" / "w1.wav", dur=0.35, freq=180)
    _write_tone(tmp_path / "xx" / "good" / "w2.wav", dur=0.60, freq=360)
    _write_tone(tmp_path / "xx" / "good" / "w3.wav", dur=0.35, freq=180)
    _write_tone(tmp_path / "xx" / "flat" / "w1.wav", dur=0.45, freq=250)
    _write_tone(tmp_path / "xx" / "flat" / "w2.wav", dur=0.45, freq=250)
    _write_tone(tmp_path / "xx" / "flat" / "w3.wav", dur=0.45, freq=250)

    c1 = Contrast("c1", "C1", ("a", "b"), (("w1", "w2", "w3"),), "tip", "why")
    c2 = Contrast("c2", "C2", ("c", "d"), (("w1", "w2"),), "tip", "why")
    profile = LanguageProfile(code="xx", name="Test", contrasts=(c1, c2), tips={})

    prog = progress.Progress(tmp_path / "prog.json")
    prog.record_session("xx", {"contrasts": {
        "c1": {"label": "C1", "seen": 5, "correct": 5, "accuracy": 1.0, "tip": ""},
    }}, "s1", now=1000.0)
    prog.record_session("xx", {"contrasts": {
        "c2": {"label": "C2", "seen": 5, "correct": 5, "accuracy": 1.0, "tip": ""},
    }}, "s2", now=2000.0)

    session = hvpt.Session(profile, seed=42, progress=prog, recordings=RecordedTalkers(tmp_path, "xx"))
    assert session.pools["c1"] == [("w1", "w2")]
    assert session._pair_talkers[("w1", "w2")] == ["good"]
    for _ in range(30):
        assert session.next_trial("c1").talker == "good"

    # Both c1 and c2 have 100% accuracy, so _last_practiced must favour stale c1.
    picks = Counter(session.next_trial().contrast_id for _ in range(300))
    assert picks["c1"] > picks["c2"]


def test_reliability_uses_requested_threshold_and_complete_tables(tmp_path, monkeypatch):
    from eval.native_control import per_phone
    from mdd import reliability as rel

    reps = [(
        "s",
        {"phones": [
            {"canonical": "d", "realized": "d", "op": "match", "gop": 0.0, "flagged": False},
            {"canonical": "ɡ", "realized": "k", "op": "sub", "gop": -1.5, "flagged": True},
        ]},
    )]
    rows = per_phone(reps, -1.0, top=None)
    assert {r["canonical"] for r in rows} == {"d", "ɡ"}

    results = tmp_path / "results" / "xx"
    results.mkdir(parents=True)
    (results / "native_control.json").write_text(json.dumps({
        "per_phone_complete": True,
        "per_phone_by_tau": {
            "-2.0": [{"canonical": "d", "false_flags": 0, "occurrences": 100, "rate": 0.0, "heard_as": []}],
            "-1.0": [{"canonical": "d", "false_flags": 8, "occurrences": 100, "rate": 0.08, "heard_as": [["t", 8]]}],
        },
        "per_phone": [],
    }), encoding="utf-8")
    (results / "synthetic_errors.json").write_text(json.dumps({
        "rows": [
            {"name": "err_a", "recall": {"-2.0": 0.1, "-1.0": 0.6}, "rendered": 3.0},
            {"name": "err_b", "recall": {"-2.0": 0.0, "-1.0": 0.2}, "rendered": 3.0},
        ],
    }), encoding="utf-8")

    monkeypatch.setattr(rel, "RESULTS", tmp_path / "results")
    rel._native.cache_clear()
    rel.blind_spots.cache_clear()
    try:
        assert rel.reliability("d", "xx", "t", threshold=-2.0).level == "solid"
        assert rel.reliability("d", "xx", "t", threshold=-1.0).level == "noisy"
        assert rel.reliability("pf", "xx", "f", threshold=-1.0).level == "unknown"
        assert rel.blind_spots("xx", threshold=-2.0) == {"err_a": 0.1, "err_b": 0.0}
        assert rel.blind_spots("xx", threshold=-1.0) == {"err_b": 0.2}
    finally:
        rel._native.cache_clear()
        rel.blind_spots.cache_clear()


def test_canonical_signature_tracks_equiv_and_ins_min_prob(monkeypatch):
    from eval import common
    from mdd import languages, pipeline as pl

    before = common.canonical_signature("père", "fr")
    orig_fr = languages.PROFILES["fr"]
    try:
        languages.PROFILES["fr"] = LanguageProfile(
            **{**orig_fr.__dict__, "equiv": {**orig_fr.equiv, "ɛ̃": "ɛ"}}
        )
        assert common.canonical_signature("père", "fr") != before
    finally:
        languages.PROFILES["fr"] = orig_fr

    monkeypatch.setattr(pl, "INS_MIN_PROB", 0.75)
    assert common.canonical_signature("père", "fr") != before
