# german-mdd

Training-free German mispronunciation detection & diagnosis. See `german-mdd-design.md`.

```bash
pip install -r requirements.txt   # includes espeakng-loader, which bundles espeak-ng (no system install needed)
python -m pytest tests            # alignment/diagnosis tests, no model needed
python -m mdd.pipeline "Ich möchte ein Bier" rec.wav       # full pipeline
python -m mdd.pipeline "Ich möchte ein Bier" --ipa "ɪk mɔktə aɪn biːɾ"   # text-only dry run
python app.py                     # web UI: record in the browser, see flagged sounds + tips
```
First real run downloads `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (~1.2 GB).

On Windows, `app.py` and `python -m mdd.pipeline` re-launch themselves in UTF-8 mode automatically
(panphon's data files need it). For `pytest`, set `$env:PYTHONUTF8=1` in your shell first.

If `panphon` fails to install with `AttributeError: install_layout` (Debian/Ubuntu system Python), run
`SETUPTOOLS_USE_DISTUTILS=stdlib pip install unicodecsv` first, then retry.

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
Caveat: TTS audio is clean and consistent; tier 3 (your own recordings, annotated) is still the only real ground truth.
