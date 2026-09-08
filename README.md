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

# better perception stimuli: neural voices instead of espeak (needs internet)
python -m eval.make_stimuli fr        # ~9 talkers from 3 voices x 3 speaking rates
python -m eval.make_stimuli fr --force   # re-render (after a clip-quality change)
python -m eval.make_stimuli de

# recorded native talkers — the only route for stød
python -m mdd.recorded da --script              # word list to hand a speaker
python -m mdd.recorded da --root ./recordings   # what those recordings cover

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
| French | reliable | 5 / 5 contrasts | tier 1 run — **production not ready**, see below |
| Danish | good segments, unreliable stød | 4 / 5 contrasts | not yet |
| Korean | unreliable | disabled | not yet |

**French** trains the five contrasts that matter most for English speakers: /y/ vs
/u/ (tu/tout), the three nasal vowels (sans/sain/son), nasal vs oral (paix/pain),
close vs open mid vowels (saute/sotte), and é vs è (les/lait). espeak needs two
different names for it — `fr-fr` to phonemise, `fr` to synthesise — which the
profile carries as `g2p_code` and `voice`.

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

`mdd/recorded.py` takes a directory of clips (`<root>/<lang>/<talker>/<word>.wav`)
and makes it the stimulus source, which unblocks stød and improves every other
contrast. It is 41 Danish words per speaker, roughly ten minutes each. Recordings
are not trusted blindly — they go through the same gate, so a set where two words
were recorded identically is still rejected. Prefer **citation form**: stød is
reliably realised on an isolated stressed word and weakens when the word is
unstressed in running speech, so clips excised from continuous audio are the
least reliable source for the contrast that needs them most.

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
python -m eval.preflight fr        # check a language is ready before a long run (no model needed)
python -m eval.native_control --lang fr   # same for French; results land in eval/results/fr/
python -m eval.synthetic_errors    # tier 2: inject each error from the catalogue via espeak phonemes -> recall + diagnosis
```

Reports land in `eval/results/*.md` (plus JSON). Synthesised audio and model outputs are cached under
`eval/cache/`, so re-running a threshold sweep is instant. The analysis cache key includes a hash of the
canonical phone sequence, so changing G2P, the tokeniser or a language profile invalidates the affected
entries automatically and leaves untouched languages cached — an earlier version keyed only on a
hand-maintained `VERSION`, which was forgotten once and silently replayed a whole run's stale numbers. `--n 20` limits the sentence count for a quick pass;
`--voices` picks TTS backends (`edge:<voice>` needs internet, `sapi:<voice>` is Windows-only, `espeak` is offline).

### Current numbers (pipeline v3)

Tier 1, 510 native clips from 5 natural voices (8830 phones), every flag a false positive:

| τ | -2 | -1 (default) | 0 (gate off) |
|---|---|---|---|
| phone-level FPR | 1.6% | 2.6% | 4.3% |
| sentences with ≥1 false flag | 24% | 33% | 49% |

Tier 2, 157 errors injected into espeak phoneme strings (**note**: rows whose
injection espeak does not render are untestable, see below): recall 64% at the default τ
(61% at -2), exact diagnosis 50%. Reliable (≥85% recall):
ü_long→uː, ü_short→ʊ, ö_long→oː, ö_short→ɔ, ach→k, ach→h, z→voiced_z, ei→iː, eu→uː. Weak: ich→sch (25%), final_t→d (0%), final_k→ɡ (0%), final_p→b (0%), long_a→short (0%), schwa→eː (38%). Final devoicing is not detectable
with this recogniser at all: it hears a word-final voiced stop as its devoiced twin, the "bias toward canonical" risk
from section 9 of the design doc. Vowel length alone is not flagged by default (`flag_length=True` to enable): the
recogniser mis-hears native long vowels as short 37% of the time.

The baseline before the eval-driven fixes (coda-r acceptance, ʏ/ɛː inventory mapping, insertion confidence gate)
is kept in `eval/results/v1/`; `python -m eval.compare eval/results/v1 eval/results` prints the before/after.

The thresholds and inventory mappings the eval produced (`GOP_THRESHOLD`,
`INS_MIN_PROB`, the ʏ→y and ɛː→eː folds, coda-r acceptance) were tuned on
**German only**. Every other language inherits the machinery but not the
calibration.

`python -m eval.preflight <lang>` checks a language is ready before you spend the
time: which dependencies are missing, whether the recogniser is already cached, a
runtime estimate, and a model-free look at the canonical side — espeak
language-switch artifacts and contrast phones too rare for a per-phone rate to
mean anything. It found two English loanwords (`pull`, `week-end`) in the first
draft of the French sentences that espeak phonemises *as English*, which would
have quietly corrupted those comparisons.

Tier 1 is now language-parameterised, so `python -m eval.native_control --lang fr`
produces a French false-positive rate and a recommended threshold (70 sentences
are included; it needs internet for the neural voices and downloads the 1.2 GB
recogniser on first run). German keeps its original paths, voice spellings and
clip cache so its committed baseline stays comparable. Tier 2 is still
German-only: it needs a French error catalogue in `eval/errors.py` before it can
report recall.

### French tier-1 result: improving, still not ready

The first French run gave a pooled natural-voice false-positive rate of **12.8%
at τ=-2**, against German's 1.6%. Three things were behind it, two of them the
harness's fault:

1. **Liaison.** Phonemising word by word, `les amis` is `le ami` while a speaker
   says `lezami`, so the recogniser hears a /z/ nobody predicted. Liaison was the
   single largest false-positive category (insertions: z×30, t×19). Fixed —
   `mdd/g2p.py` now phonemises a sentence at a time, which espeak applies liaison
   across. German is byte-identical either way, verified over all 102 of its
   sentences and all 62 error injections.
2. **A dialect mismatch.** `fr-CA-SylvieNeural` was in the default voice list
   against an `fr-fr` canonical transcription. Quebec French differs
   systematically, so its 20.3% disagreement measured dialect, not error. Removed.
3. **One outlier voice.** Per-voice disagreement ranged 8.5% to 47.1%, and
   pooling let the worst talker set the threshold for everyone — hence the
   nonsensical "recommended τ = -8". The report now also recommends from the
   **median voice** and prints the spread.

After the liaison fix the insertion row fell from 101 flags to 60 and lost its
liaison consonants entirely (was z×21 t×15, now ə×12 ŋ×8 l×6). Per voice:

| voice | disagreement | FPR at τ=-2 |
|---|---|---|
| fr-FR-Henri | 8.5% → **7.1%** | 4.0% → **2.6%** |
| fr-FR-Denise | 9.3% → **7.9%** | 5.6% → **4.2%** |
| fr-FR-Eloise | 47.1% → 45.4% | 30.1% → 28.8% |

Henri is at **2.6%** and Denise **4.1%** at τ=-2, against German's 1.6%.

**Eloise is now the dominant problem and is not a French one**: 26.8% against
their 2.6-4.1%, and 42.8% disagreement against their ~7%. She is roughly
two-thirds of the pooled figure. Without her the two remaining voices average
about 3.4% at τ=-2, which is a usable production scorer. The report now prints a
**per-voice phone breakdown** whenever one talker disagrees more than twice as
often as another: if the same phones dominate every voice it is a degree
difference (rate, recording), and if different phones dominate it is an accent
and that voice is measuring something other than the pipeline. Read the
median-voice recommendation, not the pooled one, until this is settled.

A second fix followed from the per-phone table and **has now been measured**:
/ʁ/ was the largest single-phone source at 16.3%, heard as ∅×16, h×11, x×6.
Those are allophones, not errors — French /ʁ/ devoices to [χ] next to voiceless
consonants and phrase-finally (the recogniser has no [χ] and spells it h or x),
and it drops from a final obstruent+liquid cluster in ordinary speech (quatre →
[kat]). Accepting them halved it: **16.3% → 8.0%**, from second in the table to
seventh, with h and x gone entirely and ∅ down 16 → 7. The seven that remain are
pre-vocalic, where dropping r really is an error, and the English rhotic [ɹ]
stays flagged.

What is left after all that is the recogniser itself, and tuning will not touch
it: **nasal vowels** (ɛ̃ 34.4%, heard as `a` ×22; ɔ̃ 25.8%; ɑ̃ 23.1%) and **front
rounded vowels** (y 31.0%, ø 40%, œ 40.7%) — exactly the contrasts French
learners need most. Perception training uses no model and is unaffected.
Production scoring for French stays unreliable, and probably needs a
French-specific acoustic model rather than a better threshold.

## Reading a production report

Not every flag is worth the same. The production tab annotates each with a
confidence read from the committed evaluation results (`mdd/reliability.py`):

- **noisy** — the recogniser does this to native speakers too. German `d`→`t` is
  the case to know: /d/ is falsely flagged on only 2.8% of its native
  occurrences, but 10 of those 11 flags were specifically `→t`, so seeing exactly
  that substitution is weak evidence.
- **fair** / **solid** — progressively less likely to be the recogniser's own error.
- **unknown** — that language has not been evaluated yet, stated rather than guessed.

It also names genuine **blind spots**, where absence from a report means nothing
— but only once they are shown to be blind spots. Tier 2 injects errors into
espeak phoneme strings, and an injection espeak does not render produces
identical audio, so the pipeline scores 0% recall on an error that is not in the
signal. Measured: `final_k→ɡ` separates at **0.93x** against synthesis jitter,
`final_t→d` at 1.01x and `final_p→b` at 0.99x — all silent, so three of the four
"undetectable" rows were **untestable, not undetectable**. `long_a→short`
separates at 7.91x and is a real blind spot.

`eval/synthetic_errors.py` now measures this per row and reports `n/a ⚠️ not
rendered` instead of a recall figure, and the app withholds any blind-spot claim
that predates the measurement rather than repeating one that may be an
artifact.

## Honest limits

- Perception stimuli default to espeak **formant synthesis**, which is
  intelligible enough to validate a contrast mechanically but thin and robotic to
  train on — the first thing a real user noticed. `python -m eval.make_stimuli
  <lang>` renders the contrast words with neural voices at several speaking rates
  (nine talkers from three voices for French) into `~/.mdd/stimuli`, and the app
  uses whatever is there automatically. That is still synthetic; recorded native
  talkers via `mdd/recorded.py` remain the goal, and drop into the same directory.
  Either way the audio goes through the same gate — better-sounding stimuli are
  not exempt from having to separate the pair, and separability is checked **per
  talker**, so one voice that merges a pair is dropped for that pair rather than
  disabling the contrast.
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
