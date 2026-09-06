# German Mispronunciation Detection & Diagnosis (MDD) — Design Doc v0.1

## 1. Goal

Given a German reference sentence and a learner recording, produce:

1. **Detection** — which phones (and words) were mispronounced.
2. **Diagnosis** — what was produced instead, and a human-readable correction tip.
3. **Confidence** — a per-phone score so weak evidence can be suppressed.

Primary user: an L1-English learner at A1–A2. Runs locally on CPU, sub-5s latency per sentence. No task-specific training required for v1.

Non-goals for v1: prosody/stress scoring, fluency scoring, real-time streaming, speech therapy use.

## 2. Why training-free first

There is no public, phone-level-annotated German L2 corpus comparable to L2-ARCTIC (English). Training a German MDD model from scratch would mean building the dataset first. The 2025 retrieval-based result (~70% F1 on L2-ARCTIC with zero MDD training) shows that a strong pretrained multilingual phone recognizer plus careful alignment gets most of the way. So v1 is a pipeline, not a model; v2 adds learned components once we have data and a baseline to beat.

## 3. Pipeline

```
audio (16 kHz mono)
   │
   ▼
[A] Phone recognizer  ── wav2vec2-xlsr-53-espeak-cv-ft (CTC over IPA)
   │  → recognized IPA sequence + per-frame log-posteriors
   ▼
[B] Reference G2P     ── phonemizer / espeak-ng, lang="de"
   │  → canonical IPA sequence for the reference text (+ word boundaries)
   ▼
[C] Inventory normalizer
   │  → both sequences mapped into one shared German phone set (see §5)
   ▼
[D] Feature-aware aligner ── Needleman-Wunsch, panphon feature-edit cost
   │  → (canonical, realized, op) triples: match / sub / del / ins
   ▼
[E] Confidence scorer ── GOP-style score from CTC posteriors per canonical phone
   │  → drop "errors" below threshold τ
   ▼
[F] Diagnosis mapper  ── rule table (canonical→realized → tip), optional LLM rewrite
   │
   ▼
JSON report + optional TTS of the correct form
```

### 3.1 Component notes

**[A] Recognizer.** `facebook/wav2vec2-xlsr-53-espeak-cv-ft` was fine-tuned on Common Voice with espeak-ng phone labels across many languages, so its output inventory *is already* espeak's IPA — which makes it line up with [B] almost for free. Keep the raw logits; [E] needs them. Fallback: `wav2vec2-lv-60-espeak-cv-ft` is English-only and not appropriate for German.

**[B] G2P.** `phonemizer` with the espeak-ng backend, `language="de"`, `with_stress=True`, word separator preserved. espeak's German is Standard German (roughly Duden norm) — that's our target accent. Keep stress marks for later prosody work but strip them before alignment in v1.

**[C] Normalizer.** espeak emits some diacritics/length marks inconsistently between G2P output and CTC labels. Normalize: strip stress, collapse `ː` into a length attribute on the vowel token rather than a separate symbol, map tie-bars in affricates (`t͡s` → `ts` as one token), unify `ʁ`/`r`/`ɐ` handling (§5).

**[D] Aligner.** Standard NW with gap penalty g and substitution cost `c(a,b) = panphon feature distance(a,b)` scaled to [0,1]. This is the single most important choice for useful output: with uniform costs, a learner saying `[u]` for `[y]` may align as delete+insert; with feature costs it becomes a clean substitution, which is what we want to diagnose.

**[E] Confidence.** For each canonical phone p spanning frames [s,e] (from CTC forced alignment of the *canonical* sequence — use `torchaudio.functional.forced_align`), compute
`GOP(p) = (1/(e−s)) Σ_t [ log P(p|x_t) − max_q log P(q|x_t) ]`.
Near 0 → the canonical phone was the best hypothesis; strongly negative → the model preferred something else. Flag an error only when the free-decode alignment [D] disagrees **and** GOP(p) < τ. This two-signal rule is the main defense against false positives on correct speech. Insertions have no canonical phone to score, so they are gated instead on the recognizer's own confidence in the inserted phone (its peak CTC posterior; native-speech blips sit around 0.4, real phones around 0.95).

**[F] Diagnosis.** A lookup keyed on `(canonical, realized)` → tip. Seed it with the L1-English error table in §6. Unknown pairs fall back to a generic message with the two IPA symbols. Optionally pass the structured diff to an LLM to phrase feedback; the LLM never sees audio and never decides what the error is.

## 4. Data model

```python
@dataclass
class PhoneResult:
    word: str
    canonical: str       # IPA
    realized: str | None # IPA, None on deletion
    op: Literal["match","sub","del","ins"]
    gop: float
    t_start: float; t_end: float
    flagged: bool
    tip: str | None

@dataclass
class UtteranceResult:
    text: str
    phones: list[PhoneResult]
    word_scores: dict[str, float]   # min GOP per word
    overall: float                  # fraction of unflagged phones
```

## 5. German phone inventory decisions

Places where a naive pipeline goes wrong on German specifically:

| Issue | Decision |
|---|---|
| Vowel length/tenseness (`Stadt` /ʃtat/ vs `Staat` /ʃtaːt/) | Treat length as a feature on the vowel, not a separate token; substitution cost between long/short same-quality vowel is low but nonzero so it *is* flagged, with its own tip. |
| `ch` allophones `[ç]` (ich) vs `[x]` (ach) | espeak G2P handles the rule; keep both distinct. Confusing them is a real learner error. |
| `r` realizations `[ʁ] [r] [ʀ] [ɐ]` | Accept any of `ʁ/r/ʀ` as a match for canonical `ʁ` (regional variation, not error). espeak's G2P writes coda r as consonantal `ʁ` (Bier → biːʁ) even though natives vocalize it, so the pipeline accepts `[ɐ]`, `[ɜ]`, `[ə]`, `[a]` or nothing for coda `ʁ` (found by the native-control eval: coda r was the top false positive). Onset r stays strict. English `[ɹ]` is always a flag. |
| Final devoicing (`Hund` → /hʊnt/) | espeak already devoices in G2P; a learner saying `[d]` gets flagged with a devoicing tip. |
| Glottal stop `[ʔ]` before vowel-initial syllables | espeak may or may not emit it; strip `ʔ` from both sides in v1. |
| Schwa / syllabic consonants (`-en` → `[n̩]`) | Map `ən` ≈ `n̩` as equivalent in v1. |
| Affricates `pf`, `ts`, `tʃ` | Single tokens so a learner dropping the `p` in `Pferd` shows as a substitution `pf→f`, not a deletion. |

## 6. Expected L1-English error table (seed for the diagnosis mapper)

| Canonical | Common realization | Example | Tip |
|---|---|---|---|
| `y`, `yː` (ü) | `u`, `uː`, `ju` | über, müde | Say /i/ then round your lips without moving the tongue. |
| `ø`, `øː`, `œ` (ö) | `o`, `ɛ`, `ɜ` | schön, können | Say /e/ then round your lips. |
| `ç` | `k`, `ʃ`, `x` | ich, nicht | Whisper the "h" in "huge" — tongue high, front. |
| `x` | `k`, `h`, `ç` | Buch, machen | Friction at the back, like clearing your throat gently. |
| `ʁ` | `ɹ` | rot, Brot | Uvular — vibrate at the very back, not curled tongue. |
| `ts` (z) | `z` | Zeit, zehn | Start with a sharp /t/: "tsait". |
| `v` (w) | `w` | wir, was | Top teeth on lower lip — it's English "v". |
| `f` (v) | `v` | Vater, viel | Written v, said /f/. |
| `ʃt`/`ʃp` initial | `st`/`sp` | Straße, Sport | "sht", "shp" at word start. |
| Final `t/k/p` | `d/g/b` | Hund, Tag, halb | Devoice at the end of the word. |
| `aː` vs `a` | length collapsed | Staat/Stadt | Hold the long vowel roughly twice as long. |
| `ə` | full vowel | bitte, Name | Unstressed final -e is a short schwa, never silent, never "ay". |
| `ai`, `ɔy` (ei, eu) | `iː`, `juː` | ein, neu | ei = "eye", eu = "oy". |
| `ŋ` before `k`/`g` | `ŋg` | singen | No hard g after ng. |

## 7. Evaluation plan (no labeled German L2 corpus exists)

Three complementary test sets, cheapest first:

1. **Native negative control** — 500 Common Voice German clips from high-rated speakers. Any flag is a false positive. Target: phone-level FPR < 5%. This calibrates τ.
2. **Synthetic positives** — take native sentences, apply an error from §6 to the G2P string, synthesize with a German TTS (Piper/Coqui) that accepts phoneme input, run the pipeline. Measures recall & diagnosis accuracy per error type. Caveat: TTS errors are "clean"; real ones are messier.
3. **Real learner set** — 100–200 sentences you record yourself (plus anyone you can recruit), annotated at the phone level with `praat`/`ELAN` or a simple CLI. This is the only set that reflects reality and becomes v2 training/finetuning data.

Metrics (standard MDD convention): detection precision/recall/F1 at phone level; diagnosis accuracy = fraction of correctly detected errors whose realized phone matches the annotation; per-error-type recall from set 2.

Baseline to beat: GOP-only with a single threshold (drop [D], use only [E]).

## 8. Roadmap

| Phase | Deliverable | Effort |
|---|---|---|
| 0 | Repo skeleton, load recognizer, G2P a sentence, print both IPA strings | 1 evening |
| 1 | Normalizer + feature-aware NW aligner, JSON report | 1–2 evenings |
| 2 | CTC forced alignment + GOP, threshold τ from Common Voice control | 1–2 evenings |
| 3 | Diagnosis table + CLI (`mdd "Ich möchte ein Bier" rec.wav`) | 1 evening |
| 4 | Synthetic eval harness, per-error-type report | weekend |
| 5 | Self-recorded set, annotation tool, first real F1 number | ongoing |
| v2 | Fine-tune recognizer head on (native CV + synthetic errors) → adapt on real set, per the two-stage recipe from the 2026 Arabic MDD work; consider from-scratch GOP scorer over frozen wav2vec2 features | later |

## 9. Risks & mitigations

- **Recognizer bias toward canonical.** CTC phone recognizers trained on native speech tend to "hear" the expected phone. Mitigation: the two-signal rule (§3 [E]); in v2, decode with no LM and consider N-best.
- **G2P disagreements aren't learner errors.** espeak may pick a pronunciation for loanwords/names that differs from the norm. Mitigation: allow per-word override dictionary; skip proper nouns.
- **Regional native variation flagged as error.** Handled partly in §5; add an "accept alternates" set per phone.
- **Audio quality.** Require 16 kHz mono, apply VAD trim; reject clips with SNR below a floor rather than produce garbage.
- **Over-flagging kills motivation.** Default to conservative τ; expose a "strictness" setting.

## 10. Stack

Python 3.11, `torch`, `torchaudio`, `transformers`, `phonemizer` (+ system `espeak-ng`), `panphon`, `numpy`, `soundfile`. Optional: `piper-tts` for synthetic data, `librosa` for VAD/SNR. Everything CPU-viable; a GPU only matters for v2 fine-tuning.
