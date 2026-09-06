"""German mispronunciation detection & diagnosis (training-free v1)."""
import os
import shutil

# phonemizer (used by g2p and by the recogniser's tokenizer) needs espeak-ng. With no system
# install and no explicit library path, point it at the DLL/.so bundled in `espeakng-loader`.
if "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ and shutil.which("espeak-ng") is None:
    try:
        import espeakng_loader
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeakng_loader.get_library_path()
        os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
    except ImportError:
        pass
