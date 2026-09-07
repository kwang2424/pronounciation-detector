# Pronunciation trainer

Training-free pronunciation work for German, Danish and Korean, in two halves:

- **Production** — mispronunciation detection and diagnosis. Say a sentence, get
  per-phone feedback. See `german-mdd-design.md`.
- **Perception** — High Variability Phonetic Training. Identify a word from a
  minimal set, heard from a different synthetic talker each trial. Perceptual
  gains transfer to production (Bradlow et al. 1997), so this is the half worth
  doing first. See `perception-design.md`.

```bash
pip install -r requirements.txt   # includes espeakng-loader, which bundles espeak-ng (no system install needed)
python -m pytest tests            # alignment/profile/HVPT tests, no model needed
python app.py                     # web UI: both tabs plus a coverage report

# production feedback
python -m mdd.pipeline "Ich möchte ein Bier" rec.wav
python -m mdd.pipeline "Ich möchte ein Bier" --ipa "ɪk mɔktə aɪn biːɾ"   # text-only dry run
python -m mdd.pipeline "mad gade" --lang da --ipa "mad ɡadə"

# perception progress is kept in ~/.mdd/progress.json ($MDD_PROGRESS to move it)

# which contrasts can espeak actually render?
python -m mdd.validate da
python -m mdd.validate --no-audio          # fast, transcription check only
python -m eval.tts_probe da                # could a neural voice rescue a gated one? (needs internet)
```
First real run of the production tab downloads `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (~1.2 GB).
The perception tab needs no model at all.

## Languages

Everything language-specific lives in `mdd/languages.py`: tokenizer rules, error
tips, coda-r leniency, and the contrast inventory that drives perception
training. The alignment, GOP and staircase code is language-neutral.

| | espeak G2P | Perception training | Evaluated |
|---|---|---|---|
| German | reliable | 4 / 4 contrasts | yes, tiers 1-2 below |
| Danish | good segments, unreliable stød | 4 / 5 contrasts | not yet |
| Korean | unreliable | disabled | not yet |

**Danish** is the best fit for this approach, because its difficulty is
concentrated in exactly what the tool addresses: the soft d, a very dense vowel
inventory, and r-colouring. espeak renders all of those faithfully. It does not
render **stød** reliably — it places the glottal stop on the onset consonant
rather than the rhyme, and `hun`/`hund` and `mor`/`mord` come out identical — so
stød is excluded from training rather than drilled with unanswerable trials. It
is still diagnosed on the production side, where the learner's own audio is the
evidence.

Neural TTS does not rescue it either: measured over six minimal pairs, one Danish
neural voice creaked on 9 of 12 words including three with no stød (creaky voice
quality, not phonology) and the other creaked only on a *non*-stød word. Stød
needs recorded native talkers; `python -m eval.tts_probe da --anatomy` is how that
was established and how to re-test any new backend.

Perception progress persists across sittings in `~/.mdd/progress.json` (set
`$MDD_PROGRESS` to move it). The app saves after every answer, so closing the tab
mid-session loses nothing, and practice is steered by *lifetime* accuracy with
least-recently-practised as the tiebreak — which spaces a training block instead
of grinding one contrast. Delete the file to start over.

**Korean** perception training is off. espeak collapses the three-way laryngeal
contrast (자다/짜다 both give `tɕɐdɐ`), skips obligatory sandhi (신라 → `sinɾɐ`
instead of /silla/), and renders fortis stops as uvulars — audibly distinct, but
not the Korean contrast, so training on it would build the wrong category.
Korean needs a real G2P (g2pK or KoG2P) and recorded talkers.

## Evaluation

Two automated tiers from section 7 of the design doc (the third, real learner recordings, is manual):

```bash
python -m eval.native_control      # tier 1: native TTS voices, every flag is a false positive -> FPR per threshold
python -m eval.synthetic_errors    # tier 2: inject each error from the catalogue via espeak phonemes -> recall + diagnosis
```

Reports land in `eval/results/*.md` (plus JSON). Synthesised audio and model outputs are cached under
`eval/cache/`, so re-running a threshold sweep is instant. `--n 20` limits the sentence count for a quick pass;
`--voices` picks TTS backends (`edge:<voice>` needs internet, `sapi:<voice>` is Windows-only, `espeak` is offline).

### Current numbers (pipeline v3)

Tier 1, 510 native clips from 5 natural voices (8830 phones), every flag a false positive:

| τ | -2 | -1 (default) | 0 (gate off) |
|---|---|---|---|
| phone-level FPR | 1.6% | 2.6% | 4.3% |
| sentences with ≥1 false flag | 24% | 33% | 49% |

Tier 2, 157 errors injected into espeak phoneme strings: recall 64% at the default τ
(61% at -2), exact diagnosis 50%. Reliable (≥85% recall):
ü_long→uː, ü_short→ʊ, ö_long→oː, ö_short→ɔ, ach→k, ach→h, z→voiced_z, ei→iː, eu→uː. Weak: ich→sch (25%), final_t→d (0%), final_k→ɡ (0%), final_p→b (0%), long_a→short (0%), schwa→eː (38%). Final devoicing is not detectable
with this recogniser at all: it hears a word-final voiced stop as its devoiced twin, the "bias toward canonical" risk
from section 9 of the design doc. Vowel length alone is not flagged by default (`flag_length=True` to enable): the
recogniser mis-hears native long vowels as short 37% of the time.

The baseline before the eval-driven fixes (coda-r acceptance, ʏ/ɛː inventory mapping, insertion confidence gate)
is kept in `eval/results/v1/`; `python -m eval.compare eval/results/v1 eval/results` prints the before/after.

The thresholds and inventory mappings the eval produced (`GOP_THRESHOLD`,
`INS_MIN_PROB`, the ʏ→y and ɛː→eː folds, coda-r acceptance) were tuned on
**German only**. Danish and Korean inherit the machinery but not the calibration:
the tiers above need re-running per language before their numbers mean anything.

## Honest limits

- Perception stimuli are **formant-synthesised**, not recorded. The HVPT
  literature measured its effects on natural multi-talker speech; synthetic
  voices are a usable bootstrap, not a replication. `Talker` and `synthesize` in
  `mdd/synth.py` are the only things a recorded-audio backend has to replace.
- `mdd/validate.py` checks that a pair is rendered *distinctly*. It cannot check
  that it is rendered *correctly*. Two demonstrated cases: espeak renders Korean
  fortis stops as uvulars, and it can be forced to "distinguish" Danish stød by
  emitting a full glottal stop (4.8x separation, phonation dropping to 0% of peak
  — a silent gap, where real stød is creaky voice that never goes silent). Both
  pass every automated check and would train a category the language does not
  have. That is why `hvpt_ready` is a human judgement, not a computed one, and
  why a high separation ratio is a reason to listen to the clips rather than to
  flip it. See `perception-design.md` §5.1.
- Whether the recogniser's CTC vocabulary covers the Danish-specific units
  (`ʔ`, `ð`, `ɐ̯`) is **unverified** — scoring falls back to the token's first
  character when one is missing, so treat Danish stød scores as unproven until
  someone runs the model. Perception training uses no model and is unaffected.

## Platform notes

On Windows, `app.py` and `python -m mdd.pipeline` re-launch themselves in UTF-8 mode
automatically (panphon's data files need it). For `pytest`, set `$env:PYTHONUTF8=1` first.

If `panphon` fails to install with `AttributeError: install_layout` (Debian/Ubuntu system
Python), run `SETUPTOOLS_USE_DISTUTILS=stdlib pip install unicodecsv` first, then retry.
