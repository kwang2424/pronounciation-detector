# Pronunciation trainer

Training-free pronunciation work for German, Danish and Korean, in two halves:

- **Production** — mispronunciation detection and diagnosis. Say a sentence, get
  per-phone feedback. See `german-mdd-design.md`.
- **Perception** — High Variability Phonetic Training. Identify a word from a
  minimal set, heard from a different synthetic talker each trial. Perceptual
  gains transfer to production (Bradlow et al. 1997), so this is the half worth
  doing first.

```bash
pip install -r requirements.txt   # includes espeakng-loader, which bundles espeak-ng (no system install needed)
python -m pytest tests            # alignment/profile/HVPT tests, no model needed
python app.py                     # web UI: both tabs plus a coverage report

# production feedback
python -m mdd.pipeline "Ich möchte ein Bier" rec.wav
python -m mdd.pipeline "Jeg vil gerne have en øl" --lang da --ipa "jaj vil ɡɛɐ̯nə hæ en øl"

# which contrasts can espeak actually render?
python -m mdd.validate da
python -m mdd.validate --no-audio          # fast, transcription check only
```
First real run of the production tab downloads `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (~1.2 GB).
The perception tab needs no model at all.

## Languages

Everything language-specific lives in `mdd/languages.py`: tokenizer rules, error
tips, and the contrast inventory that drives perception training. The alignment,
GOP and staircase code is language-neutral.

| | espeak G2P | Perception training |
|---|---|---|
| German | reliable | 4 / 4 contrasts |
| Danish | good segments, unreliable stød | 4 / 5 contrasts |
| Korean | unreliable | disabled |

**Danish** is the best fit for this approach, because its difficulty is
concentrated in exactly what the tool addresses: the soft d, a very dense vowel
inventory, and r-colouring. espeak renders all of those faithfully. It does not
render **stød** reliably — it places the glottal stop on the onset consonant
rather than the rhyme, and `hun`/`hund` and `mor`/`mord` come out identical — so
stød is excluded from training rather than drilled with unanswerable trials. It
is still diagnosed on the production side, where the learner's own audio is the
evidence.

**Korean** perception training is off. espeak collapses the three-way laryngeal
contrast (자다/짜다 both give `tɕɐdɐ`), skips obligatory sandhi (신라 → `sinɾɐ`
instead of /silla/), and renders fortis stops as uvulars — audibly distinct, but
not the Korean contrast, so training on it would build the wrong category.
Korean needs a real G2P (g2pK or KoG2P) and recorded talkers.

## Honest limits

- Stimuli are **formant-synthesised**, not recorded. The HVPT literature measured
  its effects on natural multi-talker speech; synthetic voices are a usable
  bootstrap, not a replication. `Talker` and `synthesize` in `mdd/synth.py` are
  the only things a recorded-audio backend has to replace.
- `mdd/validate.py` checks that a pair is rendered *distinctly*. It cannot check
  that it is rendered *correctly* — the Korean uvular case is exactly that gap,
  which is why `hvpt_ready` on the profile is a human judgement, not a computed one.
- The GOP threshold is still uncalibrated (§7 of the design doc).
- Whether the recogniser's CTC vocabulary covers the Danish-specific units
  (`ʔ`, `ð`, `ɐ̯`) is **unverified** — scoring falls back to the token's first
  character when one is missing, so treat Danish stød scores as unproven until
  someone runs the model. Perception training uses no model and is unaffected.

## Platform notes

On Windows, `app.py` and `python -m mdd.pipeline` re-launch themselves in UTF-8 mode
automatically (panphon's data files need it). For `pytest`, set `$env:PYTHONUTF8=1` first.

If `panphon` fails to install with `AttributeError: install_layout` (Debian/Ubuntu system
Python), run `SETUPTOOLS_USE_DISTUTILS=stdlib pip install unicodecsv` first, then retry.
