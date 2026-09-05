# german-mdd

Training-free German mispronunciation detection & diagnosis. See `german-mdd-design.md`.

```bash
sudo apt install espeak-ng
pip install -r requirements.txt
python -m pytest tests            # alignment/diagnosis tests, no model needed
python -m mdd.pipeline "Ich möchte ein Bier" rec.wav       # full pipeline
python -m mdd.pipeline "Ich möchte ein Bier" --ipa "ɪk mɔktə aɪn biːɾ"   # text-only dry run
```
First real run downloads `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (~1.2 GB).

If `panphon` fails to install with `AttributeError: install_layout` (Debian/Ubuntu system Python), run
`SETUPTOOLS_USE_DISTUTILS=stdlib pip install unicodecsv` first, then retry.
