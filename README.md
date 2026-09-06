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
