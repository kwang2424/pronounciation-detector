"""(canonical, realised) -> learner-facing tip. Seeded for L1-English learners."""

TIPS: dict[tuple[str, str], str] = {}


def _add(canons, reals, tip):
    for c in canons:
        for r in reals:
            TIPS[(c, r)] = tip


_add(["y", "yː"], ["u", "uː", "ʊ", "ju"], "ü: say /i/ (as in 'see') and round your lips without moving your tongue.")
_add(["ø", "øː", "œ"], ["o", "oː", "ɔ", "ɛ", "ɜ", "ɜː"], "ö: say /e/ (as in 'say') and round your lips.")
_add(["ç"], ["k", "ʃ", "x", "h"], "ich-Laut: a whispered 'h' as in 'huge' — tongue high and front, no contact.")
_add(["x"], ["k", "h", "ç"], "ach-Laut: friction at the back of the mouth, like gently clearing your throat.")
_add(["ʁ"], ["ɹ"], "German r is uvular — vibrate at the very back of the throat, don't curl the tongue.")
_add(["ts"], ["z", "s"], "z is /ts/: start with a sharp 't' — 'tsait' for Zeit.")
_add(["v"], ["w"], "German w is English 'v': top teeth on lower lip.")
_add(["f"], ["v"], "German v is usually /f/: Vater = 'fahter'.")
_add(["ʃ"], ["s"], "s before t/p at word start is 'sh': Straße = 'shtrahsse'.")
_add(["t"], ["d"], "Final devoicing: a written d at the end of a word is said /t/ (Hund = 'hunt').")
_add(["k"], ["ɡ"], "Final devoicing: a written g at the end of a word is said /k/ (Tag = 'tahk').")
_add(["p"], ["b"], "Final devoicing: a written b at the end of a word is said /p/ (halb = 'halp').")
_add(["ə"], ["eː", "e", "ɛ", "aɪ"], "Unstressed final -e is a short schwa — never 'ay', never silent.")
_add(["aɪ"], ["iː", "i"], "ei is pronounced 'eye'.")
_add(["ɔʏ"], ["juː", "u", "uː", "ɛʊ", "ɔ", "ʊ"], "eu/äu is pronounced 'oy'.")
_add(["pf"], ["f", "p"], "pf is one sound: close the lips for p, release straight into f.")


def tip_for(canonical: str | None, realized: str | None) -> str:
    if canonical is None:
        return f"Extra sound [{realized}] inserted."
    if realized is None:
        return f"Sound [{canonical}] was dropped."
    if canonical.rstrip("ː") == realized.rstrip("ː"):
        if canonical.endswith("ː"):
            return f"Vowel [{canonical}] should be long — hold it about twice as long."
        return f"Vowel [{canonical}] should be short."
    return TIPS.get((canonical, realized), f"Expected [{canonical}], heard [{realized}].")
