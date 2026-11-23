"""Tests for Phase 3: standard MDD metrics, expanded German error catalog, and Tier 3 human benchmark."""
import json
import math

import numpy as np
import pytest
import soundfile as sf

from eval import compare, errors, human_eval, metrics, synthetic_errors
from eval.tts import Espeak
from mdd.validate import SEPARATION_THRESHOLD


# ------------------------------------------------------------ eval/metrics.py
def test_compute_metrics_and_recommend_threshold():
    c1 = metrics.MDDCounts(ta=90, fr=2, tr=8, fa=2, cd=6, de=2)
    c2 = metrics.MDDCounts(ta=10, fr=0, tr=0, fa=0, cd=0, de=0)
    m = metrics.compute_metrics(c1 + c2)

    assert m["ta"] == 100 and m["fr"] == 2 and m["tr"] == 8 and m["fa"] == 2
    assert m["precision"] == pytest.approx(8 / 10)
    assert m["recall"] == pytest.approx(8 / 10)
    assert m["f1"] == pytest.approx(0.8)
    assert m["far"] == pytest.approx(2 / 10)
    assert m["frr"] == pytest.approx(2 / 102)
    assert m["cond_diag_acc"] == pytest.approx(6 / 8)
    assert m["der"] == pytest.approx(2 / 8)
    expected_mcc = (8 * 100 - 2 * 2) / math.sqrt(10 * 10 * 102 * 102)
    assert m["mcc"] == pytest.approx(expected_mcc)

    empty = metrics.compute_metrics(metrics.MDDCounts())
    assert all(empty[k] is None for k in ("precision", "recall", "f1", "mcc", "far", "frr", "cond_diag_acc", "der"))

    # Threshold recommender picks highest F1 under FRR < 0.05
    by_tau = {
        "-2.0": {"frr": 0.01, "f1": 0.65},
        "-1.0": {"frr": 0.03, "f1": 0.80},
        "0.0": {"frr": 0.07, "f1": 0.90},   # violates FRR < 0.05
    }
    assert metrics.recommend_threshold(by_tau, max_frr=0.05) == -1.0

    # L1-only fallback (no f1): picks least restrictive threshold under max_frr
    l1_only = {
        "-2.0": {"frr": 0.02, "f1": None},
        "-1.0": {"frr": 0.04, "f1": None},
        "0.0": {"frr": 0.06, "f1": None},
    }
    assert metrics.recommend_threshold(l1_only, max_frr=0.05) == -1.0


# ------------------------------------------------------------ eval/errors.py
def test_expanded_catalog_has_mnemonics_context_and_audible_rhotic():
    for err in errors.CATALOG:
        if err.realized is not None:
            for tok in err.realized.split():
                assert tok in errors.MNEMONIC, f"missing MNEMONIC for {tok!r} in {err.name}"

    names = {e.name for e in errors.CATALOG}
    expected_new = {
        "ü_long→iː", "ü_short→ɪ", "ö_long→eː", "ö_short→ɛ",
        "i_long→ɪ", "u_long→ʊ", "o_long→ɔ", "e_long→ɛ",
        "short_a→long", "ich→ach", "ach→ich", "r_onset→english_r", "final_s→z",
    }
    assert expected_new <= names

    ich_err = next(e for e in errors.CATALOG if e.name == "ich→k")
    assert not ich_err.ok(["k", "øː", "n", "ɪ", "ç"], 4), "word-final -ig must be excluded"
    assert ich_err.ok(["ɪ", "ç"], 1)


def test_r_onset_english_r_is_audibly_rendered_by_espeak(tmp_path):
    esp = Espeak("de")
    ctl = tmp_path / "ctl.wav"
    inj = tmp_path / "inj.wav"
    esp.synth_phonemes("r'o:t", ctl)
    esp.synth_phonemes(f"{errors.MNEMONIC['ɹ']}'o:t", inj)
    ratio = synthetic_errors._rendered(ctl, inj)
    assert ratio is not None and ratio >= SEPARATION_THRESHOLD


# ---------------------------------------------------- eval/synthetic_errors.py
def test_synthetic_errors_conditional_der_clean_recall_and_duration_rendered(tmp_path):
    from eval.common import THRESHOLDS

    err = errors.ErrorType("ach→h", "x", "h")
    ctl_rep = {
        "phones": [
            {"op": "match", "canonical": "b", "realized": "b", "gop": 0.0},
            {"op": "sub", "canonical": "uː", "realized": "ʊ", "gop": -1.5},  # background false reject at >= -2
            {"op": "match", "canonical": "x", "realized": "x", "gop": 0.0},
        ]
    }
    rep_wrong_diag = {
        "phones": [
            {"op": "match", "canonical": "b", "realized": "b", "gop": 0.0},
            {"op": "match", "canonical": "uː", "realized": "uː", "gop": 0.0},
            {"op": "sub", "canonical": "x", "realized": "f", "gop": -2.5},   # flagged at >= -2, wrong diag
        ]
    }
    rep_del = {
        "phones": [
            {"op": "match", "canonical": "b", "realized": "b", "gop": 0.0},
            {"op": "match", "canonical": "uː", "realized": "uː", "gop": 0.0},
            {"op": "del", "canonical": "x", "realized": None, "gop": -3.0},
        ]
    }

    c1 = synthetic_errors.evaluate(err, rep_wrong_diag, ctl_rep, 2)
    c2 = synthetic_errors.evaluate(err, rep_del, ctl_rep, 2)
    assert c2["heard"] == "∅", "deletion inside sub error must report consistent '∅'"

    summary = synthetic_errors.summarise({"ach→h": {
        "kind": "sub", "canonical": "x", "realized": "h", "note": "",
        "cases": [c1, c2],
    }})
    row = summary["rows"][0]
    assert row["recall"]["-2.0"] == 1.0
    assert row["cond_diag_acc"]["-2.0"] == 0.0
    assert row["der"]["-2.0"] == 1.0
    assert row["clean_recall"]["-2.0"] == 1.0
    assert summary["overall"]["metrics"]["-1.0"]["fr"] == 2
    assert summary["overall"]["metrics"]["-1.0"]["ta"] == 2
    assert summary["overall"]["metrics"]["-1.0"]["precision"] == pytest.approx(0.5)
    md = synthetic_errors.markdown(summary, [])
    assert "Overall MDD metrics" in md

    # Pure duration change between two equal-spectrum tones must be detected by _rendered()
    sr = 16000
    t1 = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    t2 = np.linspace(0, 0.60, int(sr * 0.60), endpoint=False)
    w1, w2 = tmp_path / "short.wav", tmp_path / "long.wav"
    sf.write(str(w1), (0.3 * np.sin(2 * np.pi * 220 * t1)).astype(np.float32), sr)
    sf.write(str(w2), (0.3 * np.sin(2 * np.pi * 220 * t2)).astype(np.float32), sr)
    ratio = synthetic_errors._rendered(w1, w2)
    assert ratio is not None and ratio > 5.0
    assert set(THRESHOLDS)


# ----------------------------------------------------------- eval/compare.py
def test_compare_prints_removed_error_types(tmp_path, capsys):
    d1, d2 = tmp_path / "before", tmp_path / "after"
    d1.mkdir()
    d2.mkdir()
    (d1 / "synthetic_errors.json").write_text(json.dumps({
        "rows": [
            {"name": "kept", "n": 8, "recall": {"-2.0": 0.5}, "diag_acc": 0.5, "error": None},
            {"name": "old_err", "n": 8, "recall": {"-2.0": 0.625}, "diag_acc": 0.5, "error": None},
        ],
        "overall": {"n": 16, "recall": {"-2.0": 0.56}, "diag_acc": 0.5},
    }), encoding="utf-8")
    (d2 / "synthetic_errors.json").write_text(json.dumps({
        "rows": [{"name": "kept", "n": 8, "recall": {"-2.0": 0.75}, "diag_acc": 0.625, "error": None}],
        "overall": {"n": 8, "recall": {"-2.0": 0.75}, "diag_acc": 0.625},
    }), encoding="utf-8")

    compare.main(d1, d2)
    out = capsys.readouterr().out
    assert "old_err (removed)" in out


# -------------------------------------------------------- eval/human_eval.py
def test_human_eval_manifest_init_import_and_mixed_scoring(tmp_path):
    init_path = tmp_path / "starter.jsonl"
    starter = human_eval.init_manifest(init_path, lang="de", n=10, speaker="me", speaker_type="l2")
    assert len(starter) == 10
    assert human_eval.load_manifest(init_path) == starter

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    sf.write(str(clips_dir / "u1.wav"), np.zeros(1600, dtype=np.float32), 16000)
    (clips_dir / "u1.txt").write_text("Ich bin gut.", encoding="utf-8")
    imported_path = tmp_path / "imported.jsonl"
    imported = human_eval.import_dir(clips_dir, imported_path, speaker="native1", speaker_type="l1")
    assert len(imported) == 1 and imported[0]["text"] == "Ich bin gut."

    # Mixed L1 + L2 dry-run scoring with substitution, deletion, insertion, duplicate word
    entries = [
        # L1 native speaker, zero errors -> clean utterance ("das Haus": d a s h aʊ s -> 6 phones)
        {"id": "l1-1", "audio": None, "text": "das Haus", "speaker": "n1", "speaker_type": "l1",
         "errors": [], "realized_ipa": "das haʊs"},
        # L2 speaker, duplicate word "Ich bin Ich": second "Ich" says /k/ for /ç/,
        # plus an unannotated false flag? No, clean elsewhere.
        {"id": "l2-1", "audio": None, "text": "Ich bin Ich", "speaker": "l1", "speaker_type": "l2",
         "errors": [{"word": "Ich", "word_idx": 2, "canonical": "ç", "realized": "k", "op": "sub"}],
         "realized_ipa": "ɪç bɪn ɪk"},
        # L2 speaker drops word-final ə in "bitte" (b ɪ t ə -> b ɪ t)
        {"id": "l2-2", "audio": None, "text": "bitte", "speaker": "l1", "speaker_type": "l2",
         "errors": [{"word": "bitte", "canonical": "ə", "realized": None, "op": "del"}],
         "realized_ipa": "bɪt"},
        # L2 speaker inserts /k/ in "singen" (z ɪ ŋ ə n -> z ɪ ŋ k ə n)
        {"id": "l2-3", "audio": None, "text": "singen", "speaker": "l1", "speaker_type": "l2",
         "errors": [{"word": "singen", "canonical": None, "realized": "k", "op": "ins"}],
         "realized_ipa": "zɪŋkən"},
    ]
    summary = human_eval.evaluate_manifest(entries, lang="de", dry_run=True)
    m = summary["metrics_by_tau"]["-1.0"]

    assert m["tr"] == 3
    assert m["fa"] == 0
    assert m["fr"] == 0
    assert m["cd"] == 3
    assert m["de"] == 0
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["frr"] == pytest.approx(0.0)
    assert m["der"] == pytest.approx(0.0)
    assert set(summary["by_speaker_type"]) == {"l1", "l2"}
    assert summary["recommended_tau"] is not None

    md = human_eval.markdown(summary)
    assert "Overall MDD metrics by threshold" in md
    assert "l1" in md and "l2" in md
