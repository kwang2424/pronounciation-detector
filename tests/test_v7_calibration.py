"""Tests for German v7 logit masking and per-phone GOP threshold offsets."""
from types import SimpleNamespace

import numpy as np
import torch

from eval.common import canonical_signature, is_flagged
from mdd import languages, pipeline, recognizer


class _StubTokenizer:
    def __init__(self, vocab: dict[str, int], unk_id: int = 1):
        self._vocab = vocab
        self.unk_token_id = unk_id

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class _StubProcessor:
    def __init__(self, vocab: dict[str, int]):
        self.tokenizer = _StubTokenizer(vocab)

    def __call__(self, wav, sampling_rate: int = 16000, return_tensors: str = "pt"):
        return SimpleNamespace(to=lambda device: {"input_values": torch.zeros((1, len(wav)))})


def test_logit_masking_blocks_out_of_inventory_tokens_and_falls_back_safely():
    vocab = {"<pad>": 0, "<unk>": 1, "ç": 2, "a": 3, "i5": 4, "ɬ": 5}
    logits = torch.full((1, 2, len(vocab)), -5.0)
    # Frame 0: Mandarin "i5" beats German "ç"; Frame 1: lateral fricative "ɬ" beats "a"
    logits[0, 0, vocab["i5"]] = 4.0
    logits[0, 0, vocab["ç"]] = 2.0
    logits[0, 1, vocab["ɬ"]] = 4.0
    logits[0, 1, vocab["a"]] = 2.0

    rec = object.__new__(recognizer.PhoneRecognizer)
    rec.device = "cpu"
    rec.processor = _StubProcessor(vocab)
    rec.vocab = {v: k for k, v in vocab.items()}
    rec.blank = 0
    rec._mask_cache = {}
    rec.model = lambda **inputs: SimpleNamespace(logits=logits)

    wav = np.zeros(320, dtype=np.float32)

    # Unmasked: noisy tokens win greedy decode
    lp_unmasked = rec.log_probs(wav)
    assert rec.greedy_decode(lp_unmasked) == "i5ɬ"

    # Masked to German inventory: "i5" and "ɬ" become -inf, blank and allowed tokens stay finite
    lp_masked = rec.log_probs(wav, allowed_tokens=frozenset({"ç", "a"}))
    assert lp_masked[0, vocab["i5"]] == float("-inf")
    assert lp_masked[1, vocab["ɬ"]] == float("-inf")
    assert torch.isfinite(lp_masked[:, 0]).all()
    assert torch.isfinite(lp_masked[:, vocab["ç"]]).all()
    assert rec.greedy_decode(lp_masked) == "ça"

    # Disjoint vocab falls back gracefully without -inf/NaN poisoning
    lp_disjoint = rec.log_probs(wav, allowed_tokens=frozenset({"not_in_vocab"}))
    assert not torch.isnan(lp_disjoint).any()
    assert rec.greedy_decode(lp_disjoint) == "i5ɬ"


class _MockOffsetRecognizer:
    supports_logit_mask = True

    def __init__(self, spikes: list[tuple[str, float]], gops: list[float], allowed_log: list):
        self._spikes = spikes
        self._gops = gops
        self._allowed_log = allowed_log

    def load_audio(self, path):
        t = np.linspace(0, 0.3, 4800, endpoint=False)
        return (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    def log_probs(self, wav, allowed_tokens=None):
        self._allowed_log.append(allowed_tokens)
        return torch.zeros((10, 4))

    def greedy_spikes(self, logp):
        return list(self._spikes)

    def gop(self, logp, canon_tokens):
        return [
            recognizer.Segment(tok, i * 0.02, (i + 1) * 0.02, g)
            for i, (tok, g) in enumerate(zip(canon_tokens, self._gops))
        ]


def test_per_phone_gop_offsets_and_is_flagged_parity():
    seen_allowed: list = []
    # German "ich": canonical /ɪ ç/, realized /ɪ x/, GOP(ç) = -0.4 under tau = -1.0 -> flagged via +0.75 offset
    rec_ich = _MockOffsetRecognizer([("ɪ", 0.99), ("x", 0.99)], [0.0, -0.4], seen_allowed)
    rep_ich = pipeline.analyse("ich", "unused.wav", rec_ich, threshold=-1.0, lang="de", check_vowel_duration=False)
    assert seen_allowed[-1] == languages.GERMAN.allowed_vocab
    c_phone = rep_ich["phones"][1]
    assert c_phone["canonical"] == "ç"
    assert c_phone["gop_offset"] == 0.75
    assert c_phone["flagged"] is True
    assert is_flagged(c_phone, -1.0) is True

    # German "heute": canonical /h ɔʏ t ə/, realized /h ɔ t ə/, GOP(ɔʏ) = -2.0 under tau = -1.0 -> forgiven via -2.0 offset
    rec_oy_mild = _MockOffsetRecognizer(
        [("h", 0.99), ("ɔ", 0.99), ("t", 0.99), ("ə", 0.99)],
        [0.0, -2.0, 0.0, 0.0],
        seen_allowed,
    )
    rep_oy_mild = pipeline.analyse("heute", "unused.wav", rec_oy_mild, threshold=-1.0, lang="de", check_vowel_duration=False)
    oy_mild = rep_oy_mild["phones"][1]
    assert oy_mild["canonical"] == "ɔʏ"
    assert oy_mild["gop_offset"] == -2.0
    assert oy_mild["flagged"] is False
    assert is_flagged(oy_mild, -1.0) is False

    # Same substitution with GOP = -4.0 (< -3.0) IS flagged
    rec_oy_bad = _MockOffsetRecognizer(
        [("h", 0.99), ("ɔ", 0.99), ("t", 0.99), ("ə", 0.99)],
        [0.0, -4.0, 0.0, 0.0],
        seen_allowed,
    )
    rep_oy_bad = pipeline.analyse("heute", "unused.wav", rec_oy_bad, threshold=-1.0, lang="de", check_vowel_duration=False)
    oy_bad = rep_oy_bad["phones"][1]
    assert oy_bad["flagged"] is True
    assert is_flagged(oy_bad, -1.0) is True


def test_canonical_signature_tracks_allowed_vocab_and_gop_offsets():
    orig_de = languages.PROFILES["de"]
    base_sig = canonical_signature("Ich heute", "de")
    try:
        languages.PROFILES["de"] = languages.LanguageProfile(
            **{**orig_de.__dict__, "allowed_vocab": orig_de.allowed_vocab | {"ʔ"}}
        )
        assert canonical_signature("Ich heute", "de") != base_sig

        languages.PROFILES["de"] = languages.LanguageProfile(
            **{**orig_de.__dict__, "gop_offsets": {**orig_de.gop_offsets, "ç": 0.5}}
        )
        assert canonical_signature("Ich heute", "de") != base_sig
    finally:
        languages.PROFILES["de"] = orig_de
