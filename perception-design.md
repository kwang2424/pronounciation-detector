# Perception Training (HVPT) & Multi-Language Support — Design Doc v0.1

## 1. Goal

Add the half of pronunciation learning the MDD pipeline does not cover: hearing
the contrast. Then generalise both halves beyond German.

The production pipeline assumes the learner can already hear what they are being
asked to fix. Often they cannot — that is why the error persists. Flege's Speech
Learning Model predicts this directly: an L2 sound that is *similar* to an L1
sound gets assimilated into the L1 category, and no amount of production practice
separates them until the perceptual category exists.

## 2. Why HVPT

High Variability Phonetic Training is the best-evidenced intervention here.

- Logan, Lively & Pisoni (1991) trained Japanese speakers on English /r/-/l/ with
  forced-choice identification over multiple talkers and phonetic environments.
  Gains generalised to new words and new talkers.
- Lively et al. (1993) showed the *variability* is what does it: single-talker
  training produces gains that do not generalise.
- Bradlow et al. (1997) showed perception training improves **production** with
  no production practice at all, and the gains persisted at three months.
- Lee, Jang & Plonsky's (2015) meta-analysis puts explicit pronunciation
  instruction at a large effect size overall.

So: forced-choice identification, many talkers, immediate feedback, short
sessions. That is the whole paradigm, and it is cheap to implement.

## 3. Pipeline

```
contrast table (per language)
      |
      v
validate: does espeak render this pair distinctly?   <-- gate
      |
      v
trial: pick minimal set -> pick target -> pick talker
      |
      v
synthesize (espeak ctypes, variant + rate + pitch)
      |
      v
learner answers -> score -> staircase adjusts difficulty
      |
      v
per-contrast accuracy + confusion tracking
```

### 3.1 Component notes

- **`mdd/languages.py`** — all per-language data: tokenizer rules, tip tables,
  contrast inventory, and two honesty flags (`g2p_caveat`, `hvpt_ready`).
- **`mdd/synth.py`** — ctypes wrapper over the espeak-ng shared library that
  `espeakng-loader` already ships. 8 talkers built from variant × rate × pitch.
  espeak's C API is not re-entrant, so calls are serialised under a lock.
- **`mdd/validate.py`** — the gate. Two checks, described in §5.
- **`mdd/hvpt.py`** — trial construction, talker rotation, staircase, stats.

## 4. Adaptive difficulty

A 2-down-1-up staircase over the **number of answer choices** (2→4): widen after
two consecutive correct, narrow after one wrong. This converges near ~70%
accuracy, which is the "desirable difficulty" region — hard enough to force
discrimination, not so hard the learner is guessing.

Trials are steered to the weakest contrast: unseen contrasts first, then lowest
accuracy. Talker never repeats on consecutive trials.

## 5. The gate: only train contrasts the synthesiser can render

This is the part that matters most, and the part it would be easiest to skip.

If the synthesiser renders `hun` and `hund` identically, a stød trial is
unanswerable. The learner scores 50%, the app reports a perception deficit, and
the deficit is the app's. So every pair is checked before use:

1. **Transcription check** — do the words tokenise differently? Deterministic,
   cheap, authoritative. A pair failing this is definitely unusable.
2. **Acoustic check** — is the between-word spectral distance larger than the
   synthesiser's own run-to-run jitter? espeak is *not* deterministic (repeat
   renders of the same word differ in length), so the threshold is a measured
   same-word noise floor, not a constant. Ratio ≥ 1.35 to pass.

Measured separation ratios, Danish (one run; they vary by a few percent because
the jitter they are measured against is itself random):

| pair | contrast | ratio | verdict |
|---|---|---|---|
| mad / mat | soft d | 2.81 | train |
| mor / mod | r-colouring | 1.85 | train |
| mile / mele | front vowels | 1.88 | train |
| ven / vand | front vowels | 1.40 | train |
| hun / hund | stød | 1.09 | **skip** |
| mor / mord | stød | 1.00 | **skip** |
| ven / vend | stød | 0.96 | **skip** |

The stød ratios sit at the noise floor: espeak's output for those pairs is
identical audio plus jitter. Note the margin above threshold is thin for some
usable pairs (ven/vand at 1.40 against a 1.35 gate), which is a reason to prefer
recorded talkers over tuning the threshold.

### 5.1 What the gate cannot do

It checks pairs are rendered *distinctly*, not *correctly*. espeak renders Korean
꽁 as `qoŋ` — a uvular stop. That is acoustically distinct from 공 `ɡoŋ` and sails
through both checks, but it is not the Korean fortis/lenis contrast, and training
on it would build a category that does not exist in the language.

Danish stød is the same trap, and a more tempting one. espeak writes `hund` as
the mnemonic `h'?un` and `hun` as `h?un` — identical but for the stress mark, so
the pair is gated. But feed espeak an explicit glottal stop in the rhyme
(`h'?u?n`) via its phoneme input and the audio separates at **4.8x**, far past the
1.35 gate. It looks like stød became trainable.

It did not. Measuring the amplitude envelope, the forced version drops to **0% of
peak** — phonation stops dead. That is a glottal stop: a silent gap. Danish stød
is creaky voice with phonation *continuing* through the rhyme, irregular but never
silent. A learner trained on it would learn to listen for a gap that real Danish
does not contain.

This matters because of where it sits. Once a correct lexicon lands and the
pipeline knows `hund` carries stød, "just emit a glottal stop" is the obvious next
step, and it passes every automated check in this repo. **Do not.** espeak's
Danish inventory has no creaky-voice unit and a formant synthesiser of this kind
does not model irregular phonation at all.

No automatic check catches any of this. Hence `hvpt_ready` on the profile: a human
judgement, set False for Korean, with the reason recorded next to it.

### 5.2 Which check runs when

A `Session` gates on the **transcription check only** — it is deterministic, so
the same contrasts are available on every run, and it already catches the case
that matters (stød). The acoustic check is stochastic and slow, so it is an audit
tool: `python -m mdd.validate` runs both and prints the ratios.

## 6. Language notes

### German — works
espeak's German G2P matches Duden for ordinary vocabulary. All four contrasts
(front rounded, ich/ach, vowel length, uvular r) validate cleanly.

### Danish — the best fit
Danish difficulty is concentrated in perception, which is what this module
addresses. The classic evidence: Bleses et al. (2008, 2011) found Danish children
acquire receptive vocabulary more slowly than Swedish and Norwegian children,
attributed to Danish's heavy consonant lenition and vowel density making word
boundaries hard to recover. Gooskens' intelligibility work finds the same
asymmetry between adult Scandinavian speakers. If native children are slowed by
the sound structure, an adult L2 learner is not going to hear it by exposure alone.

Four contrasts train: soft d, the front vowel crowd, front rounded vowels, and
r-colouring. Two normalisation fixes were needed:

- espeak's Danish voice emits **Greek** small epsilon (U+03B5) where IPA open e
  (U+025B) belongs. panphon does not recognise the Greek character, so every
  feature distance involving it silently degraded.
- The glottal stop must **not** be stripped for Danish. It is how espeak spells
  stød, which is phonemic; the German tokenizer strips it because German's
  glottal stop is automatic before initial vowels.

Stød is excluded from perception training but still diagnosed in production,
where the learner's own audio is the evidence rather than espeak's.

### Korean — not yet
espeak's Korean G2P fails on the two things that matter most:

- Collapses the three-way laryngeal contrast for some places of articulation
  (불/뿔 both `puɫ`, 자다/짜다 both `tɕɐdɐ`) and misrenders it as uvular for others.
- Does not apply obligatory sandhi: 신라 → `sinɾɐ` rather than /silla/,
  국물 → `ɡuqmuɫ` rather than /kuŋmul/. A spurious `q` appears in the inventory.

Both diagnosis and stimulus generation are untrustworthy. Fix is a real G2P
(g2pK or KoG2P) plus recorded talkers, not a tweak to the tables.

## 7. Honest limits

- **Synthetic stimuli.** The HVPT literature measured its effects on natural
  multi-talker recordings. Formant synthesis gives talker variability cheaply but
  is spectrally thinner than real speech, and the reported effect sizes should
  not be assumed to carry over. `Talker` and `synthesize` are the seam for a
  recorded-audio backend (Common Voice clips would do).
- **Minimal pair coverage is small** — 3-4 sets per contrast, hand-written. Real
  HVPT uses far more items and more phonetic environments to prevent item-level
  memorisation.
- **Retention scheduling is coarse.** Progress persists across sittings
  (`mdd/progress.py`) and steering uses lifetime accuracy with least-recently-
  practised as the tiebreak, which spaces practice within a training block. That
  is not a real spaced-repetition schedule: there is no per-contrast interval
  growth and no due date, so the app cannot yet tell you *when* to come back.
- **L1 is assumed to be English.** `Contrast.l1` records the assumption so other
  L1s can be added without rewriting the tables.
- **Unverified: recogniser vocab coverage for Danish.** The production side scores
  against `wav2vec2-xlsr-53-espeak-cv-ft`, whose CTC vocabulary is fixed. Whether
  it contains `ʔ`, `ð` and `ɐ̯` as distinct units was not checkable in the
  environment this was built in (the model host was unreachable). `PhoneRecognizer.gop`
  falls back to a token's first character when it is missing from the vocab, so
  the failure mode is a degraded score rather than a crash — but Danish stød
  scoring in particular should be treated as unproven until someone runs it.
  Perception training is unaffected: it uses no model at all.

## 7a. Persistence

Sessions are the source of truth, keyed by id; lifetime totals are *derived* by
summing them, never accumulated in place. That matters because the app saves
after every answer — a training session is normally closed, not formally ended —
and an accumulating write would count the same session once per answer. The first
version did exactly that and reported 21 trials for 6 answered; the regression
test is `test_saving_the_same_session_repeatedly_does_not_inflate_totals`.

Trimming to `MAX_SESSIONS` folds older sessions into an `archived` totals blob, so
bounding the file does not quietly shrink your history. A corrupt or
future-versioned file is reported and treated as empty, and left on disk untouched
rather than overwritten.

## 7b. Swapping the render backend

`mdd/validate.py` takes a `render` callable, so the same gate can be pointed at
neural TTS or recorded clips instead of espeak. Two backend kinds need different
floors and the code handles both: espeak is stochastic and is scored against its
own measured jitter; recorded clips and most neural TTS are deterministic, where
that jitter is exactly zero, so they are scored against `MIN_FLOOR` instead. The
first version of the pluggable backend divided by the measured floor
unconditionally and returned "unknown" for every deterministic pair — which
silently disabled the gate for precisely the backends it was added to support.

`python -m eval.tts_probe da` runs the gate over Danish neural voices and reports
which gated contrasts a better voice could rescue. It needs internet. It also
keeps the clips and prints the path, because after the glottal-stop finding above,
a high separation ratio is a reason to *listen*, never to flip `hvpt_ready`.

## 8. Roadmap

1. Recorded talkers, replacing synthesis. Note the shape needed is unusual: the
   *same* small minimal-pair list from *many* talkers, which general ASR corpora
   do not contain — it is ~40 words x 8 speakers, about five minutes of audio,
   so the scarcity is shape and labelling, not volume. The HVPT literature
   recorded its own stimuli for exactly this reason.
2. Spaced scheduling on top of the persisted history: per-contrast intervals and
   due dates, so the app can say when a contrast is due rather than only which is
   weakest. (Persistence itself landed in `mdd/progress.py`.)
3. Real Korean G2P, then re-enable Korean.
4. Wire perception results into production: flag the contrasts a learner cannot
   *hear* before scoring them on saying it.
5. More minimal pairs per contrast, and per-environment variation (onset, coda,
   stressed, unstressed).
