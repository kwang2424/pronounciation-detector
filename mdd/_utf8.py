"""Make sure the interpreter runs in UTF-8 mode.

panphon opens its data files without an explicit encoding, which fails on
Windows where the default is the console codepage (cp1252). Rather than patch
the library, re-launch the same command with PYTHONUTF8=1 when needed.
"""
import locale
import os
import subprocess
import sys


def ensure_utf8_mode() -> None:
    if sys.flags.utf8_mode or locale.getencoding().lower().replace("-", "") == "utf8":
        return
    env = {**os.environ, "PYTHONUTF8": "1"}
    sys.exit(subprocess.call(sys.orig_argv, env=env))
