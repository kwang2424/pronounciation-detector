"""Per-language profiles: tokenizer rules, error tips, and perception contrasts.

Everything that varies by target language lives here. The alignment, GOP and
staircase machinery is language-neutral and reads a `LanguageProfile`.

A `Contrast` is a set of phones learners with a given L1 tend to merge, plus
minimal pairs that pull them apart. It drives High Variability Phonetic
Training (HVPT): forced-choice identification over many talkers, which is the
best-evidenced way to build a new perceptual category (Logan/Lively/Pisoni
1991-93; Bradlow et al. 1997 showed perception gains transfer to production).

Contrast sets are seeded for L1-English learners. `l1` on each contrast records
that assumption so other L1s can be added without rewriting the tables.
"""
from dataclasses import dataclass, field

# Marks that carry no contrast in ANY language and are always removed. Anything
# that is phonemic somewhere belongs in a profile's own `strip` instead: the
# combining tilde lived here once, inherited from the German tokeniser where ɛ̃ is
# only a recogniser artifact, and it silently erased every French nasal vowel —
# paix and pain both came out /pɛ/.
BASE_STRIP = frozenset({"ˈ", "ˌ", "͡", "‿", ".", " ", "|", "?"})


@dataclass(frozen=True)
class Contrast:
    """A confusable set plus the minimal pairs that separate it."""

    id: str
    label: str
    phones: tuple[str, ...]
    #: Each tuple is one minimal set: words differing only in the target phone.
    pairs: tuple[tuple[str, ...], ...]
    tip: str
    why: str
    l1: str = "en"

    def words(self) -> list[str]:
        return [w for group in self.pairs for w in group]


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    name: str
    #: espeak's two entry points do not always agree on a language's name. Its
    #: phonemizer wants "fr-fr" for French while its synthesiser only accepts
    #: "fr"; for German, Danish and Korean one name serves both. Both default to
    #: `code`, so only the languages that need the split carry it.
    g2p_code: str = ""
    voice: str = ""
    #: Multi-character tokens (diphthongs, affricates) kept whole by the tokenizer.
    multi: tuple[str, ...] = ()
    #: Rhotics that all count as a correct realisation of the language's /r/.
    r_variants: frozenset[str] = frozenset()
    r_canonical: str = "ʁ"
    #: Realisations of a *coda* r that count as native. espeak's G2P writes coda r
    #: as a consonant, but speakers vocalise it; flagging that was the top false
    #: positive on the German native-control eval. `None` means "dropped entirely".
    #: Empty means no leniency — onset r is strict in every language.
    coda_r_ok: frozenset[str | None] = frozenset()
    #: Recogniser/G2P spelling variants folded to one token.
    equiv: dict[str, str] = field(default_factory=dict)
    #: Diacritics stripped in addition to BASE_STRIP.
    strip: frozenset[str] = frozenset()
    tips: dict[tuple[str, str], str] = field(default_factory=dict)
    contrasts: tuple[Contrast, ...] = ()
    #: Honest note on how far espeak-ng's G2P for this language can be trusted.
    g2p_caveat: str = ""
    #: Editorial gate for perception training. `mdd.validate` catches pairs the
    #: synthesiser renders identically, but it cannot tell whether a pair that
    #: *does* differ differs in the right way — espeak renders Korean fortis
    #: stops as uvulars, which is audibly distinct but phonetically wrong, and
    #: would train a category that does not exist. Set False where the output is
    #: distinct but untrustworthy.
    hvpt_ready: bool = True
    hvpt_caveat: str = ""
    example: str = ""

    @property
    def phonemizer_language(self) -> str:
        return self.g2p_code or self.code

    @property
    def synth_voice(self) -> str:
        return self.voice or self.code

    def strip_set(self) -> frozenset[str]:
        return BASE_STRIP | self.strip

    def contrast(self, contrast_id: str) -> Contrast:
        for c in self.contrasts:
            if c.id == contrast_id:
                return c
        raise KeyError(contrast_id)


def _tips(spec: list[tuple[list[str], list[str], str]]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for canons, reals, tip in spec:
        for c in canons:
            for r in reals:
                out[(c, r)] = tip
    return out


GERMAN = LanguageProfile(
    code="de",
    name="German",
    multi=("pf", "ts", "tʃ", "dʒ", "aɪ", "aʊ", "ɔʏ", "ɔɪ", "ɔy", "ɔø"),
    r_variants=frozenset({"r", "ʀ", "ɾ", "ʁ"}),
    r_canonical="ʁ",
    coda_r_ok=frozenset({None, "ɐ", "ɜ", "ə", "a"}),
    equiv={
        "ɐ̯": "ɐ", "ɛ̃": "ɛ", "ɑ": "a", "ɑː": "aː",
        "ɔø": "ɔʏ", "ɔy": "ɔʏ", "ɔɪ": "ɔʏ",   # espeak writes eu/äu as ɔø
        "g": "ɡ",                              # ASCII g -> IPA ɡ (U+0261), used by espeak and panphon
        # inventory mismatches found by the native-control eval: the recogniser never emits ʏ or ɛː
        "ʏ": "y",                              # short ü: model says y for espeak's ʏ (100% false flags otherwise)
        "ɛː": "eː",                            # long ä: merged with eː by the model (and by most speakers)
    },
    # German has no phonemic glottal stop (it is automatic before initial vowels)
    # and no nasal vowels — the recogniser's ɛ̃ is an artifact.
    strip=frozenset({"ʔ", "̯", "̃"}),
    example="Ich möchte ein Bier",
    g2p_caveat="Reliable. espeak-ng's German G2P matches Duden for ordinary vocabulary.",
    tips=_tips([
        (["y", "yː"], ["u", "uː", "ʊ", "ju"],
         "ü: say /i/ (as in 'see') and round your lips without moving your tongue."),
        (["ø", "øː", "œ"], ["o", "oː", "ɔ", "ɛ", "ɜ", "ɜː"],
         "ö: say /e/ (as in 'say') and round your lips."),
        (["ç"], ["k", "ʃ", "x", "h"],
         "ich-Laut: a whispered 'h' as in 'huge' — tongue high and front, no contact."),
        (["x"], ["k", "h", "ç"],
         "ach-Laut: friction at the back of the mouth, like gently clearing your throat."),
        (["ʁ"], ["ɹ"],
         "German r is uvular — vibrate at the very back of the throat, don't curl the tongue."),
        (["ts"], ["z", "s"], "z is /ts/: start with a sharp 't' — 'tsait' for Zeit."),
        (["v"], ["w"], "German w is English 'v': top teeth on lower lip."),
        (["f"], ["v"], "German v is usually /f/: Vater = 'fahter'."),
        (["ʃ"], ["s"], "s before t/p at word start is 'sh': Straße = 'shtrahsse'."),
        (["t"], ["d"], "Final devoicing: a written d at the end of a word is said /t/ (Hund = 'hunt')."),
        (["k"], ["ɡ"], "Final devoicing: a written g at the end of a word is said /k/ (Tag = 'tahk')."),
        (["p"], ["b"], "Final devoicing: a written b at the end of a word is said /p/ (halb = 'halp')."),
        (["ə"], ["eː", "e", "ɛ", "aɪ"], "Unstressed final -e is a short schwa — never 'ay', never silent."),
        (["aɪ"], ["iː", "i"], "ei is pronounced 'eye'."),
        (["ɔʏ"], ["juː", "u", "uː", "ɛʊ", "ɔ", "ʊ"], "eu/äu is pronounced 'oy'."),
        (["pf"], ["f", "p"], "pf is one sound: close the lips for p, release straight into f."),
    ]),
    contrasts=(
        Contrast(
            id="front-rounded",
            label="ü / u and ö / o",
            phones=("yː", "uː", "øː", "oː"),
            pairs=(("Tier", "Tür"), ("Mus", "müssen"), ("Höhle", "Hohle"), ("schon", "schön")),
            tip="Front rounded vowels: tongue where it is for 'ee'/'ay', lips rounded as for 'oo'.",
            why="English has no front rounded vowels, so /y/ is heard as /u/ and /ø/ as /o/ or /ɜ/.",
        ),
        Contrast(
            id="ich-ach",
            label="ich-Laut / ach-Laut / sch",
            phones=("ç", "x", "ʃ"),
            pairs=(("ich", "Asche"), ("Bücher", "Buch"), ("dich", "Dach")),
            tip="ç is front (like 'huge'), x is back (throat-clearing), ʃ is 'sh' with rounded lips.",
            why="English maps all three onto 'sh' or 'k'; the ç/x split is positional in German.",
        ),
        Contrast(
            id="vowel-length",
            label="Long vs short vowels",
            phones=("aː", "a", "iː", "ɪ", "oː", "ɔ"),
            pairs=(("Stadt", "Staat"), ("Bitte", "biete"), ("offen", "Ofen"), ("Wall", "Wal")),
            tip="German vowel length is phonemic: hold the long one about twice as long.",
            why="English pairs length with quality; German contrasts duration on its own.",
        ),
        Contrast(
            id="r-uvular",
            label="Uvular r vs vocalised r",
            phones=("ʁ", "ɐ"),
            pairs=(("rot", "Uhr"), ("Rand", "Vater")),
            tip="Onset r is uvular friction; after a vowel it becomes an [ɐ] offglide, not a tongue curl.",
            why="English speakers substitute the retroflex approximant [ɹ] in every position.",
        ),
    ),
)


DANISH = LanguageProfile(
    code="da",
    name="Danish",
    multi=("aj", "ɑj", "aw", "ɑw", "ɔj", "ʌj", "ɐ̯"),
    # Danish /r/ is uvular in onset; postvocalically it is the vowel [ɐ̯], which
    # is a separate token rather than an r variant.
    r_variants=frozenset({"r", "ʀ", "ʁ"}),
    r_canonical="ʁ",
    # Danish goes further than German: postvocalic r *is* the vowel [ɐ̯], so a
    # vocalised realisation is the target, not a tolerated variant.
    coda_r_ok=frozenset({None, "ɐ", "ɐ̯", "ə", "ɔ"}),
    equiv={
        # espeak's Danish voice emits GREEK SMALL LETTER EPSILON (U+03B5) for
        # what should be LATIN SMALL LETTER OPEN E (U+025B). panphon does not
        # know the Greek character, so every distance against it degrades.
        "ε": "ɛ",
        "ɑ": "a",
        "ɒ": "ɔ",
    },
    # NOTE: ʔ is *not* stripped for Danish — it is how espeak spells stød.
    strip=frozenset({"̃"}),
    example="Jeg vil gerne have en øl",
    g2p_caveat=(
        "Segments are good; suprasegmentals are not. espeak-ng renders the soft d, "
        "the front vowel series and r-colouring faithfully, but it places stød on "
        "the onset consonant rather than the rhyme and drops it in many words "
        "(hun/hund and mor/mord come out identical). Treat stød output as unreliable."
    ),
    tips=_tips([
        (["ð"], ["d", "t", "θ"],
         "Blødt d: tongue tip *down* behind the lower teeth, blade bunched — closer to English "
         "'l' in 'full' than to 'th'. Never a hard [d]."),
        (["ʔ"], [""],
         "Stød: a creaky catch in the middle of the vowel, not a full stop. It is phonemic — "
         "hun (she) vs hund (dog)."),
        (["y", "yː"], ["u", "uː", "ju", "ʊ"],
         "y: say 'ee' and round your lips without moving your tongue."),
        (["ø", "øː", "œ"], ["o", "oː", "ɔ", "ɜ", "ɜː"],
         "ø: say 'ay' and round your lips."),
        (["ɐ̯"], ["ɹ", "r", "ʁ"],
         "A written r after a vowel is not a consonant — it colours the vowel into an [ɐ] "
         "offglide (mor ≈ 'moa'). Don't curl the tongue."),
        (["ʁ"], ["ɹ", "r"],
         "Danish r at the start of a word is uvular, made at the very back of the throat."),
        (["ɛ"], ["e", "eː", "æ"], "æ/e here is open — between English 'bed' and 'bad'."),
        (["e"], ["ɛ", "eɪ"], "This e is close and pure — no 'ay' glide."),
        (["ə"], ["e", "ɛ", "eː"],
         "Unstressed -e is a schwa, and in fast speech it merges into the neighbouring consonant."),
        (["ʋ"], ["v", "w"], "Danish v is an approximant [ʋ] — lighter than English v, no buzz."),
        (["t"], ["d"], "Danish t/d differ by aspiration, not voicing: t is strongly puffed (and affricated)."),
        (["p"], ["b"], "Danish p/b differ by aspiration, not voicing: p is strongly puffed."),
        (["k"], ["ɡ"], "Danish k/g differ by aspiration, not voicing: k is strongly puffed."),
    ]),
    contrasts=(
        Contrast(
            id="soft-d",
            label="Soft d [ð] vs hard [d] / [t]",
            phones=("ð", "d", "t"),
            pairs=(("mad", "mat"), ("bad", "bat"), ("lade", "latte"), ("hvide", "hvidt")),
            tip="Blødt d is a velarised approximant: tongue tip down, blade raised. "
                "It is not English 'th' and not [d].",
            why="English speakers hear it as /d/, /ð/ or /l/ and produce a stop, "
                "which is the single most recognisable foreign feature in Danish.",
        ),
        Contrast(
            id="front-vowels",
            label="The front vowel crowd i / e / ɛ / a",
            phones=("i", "e", "ɛ", "a"),
            pairs=(("mile", "mele", "male"), ("hile", "hele", "hale"), ("vin", "ven", "vand")),
            tip="Danish packs four or more unrounded front vowels where English has two or three. "
                "Aim for evenly spaced jaw positions.",
            why="Danish has one of the densest vowel inventories of any language; "
                "neighbouring qualities fall inside a single English category.",
        ),
        Contrast(
            id="front-rounded",
            label="Front rounded y / ø vs u / o",
            phones=("y", "ø", "œ", "u", "o"),
            pairs=(("ny", "nu"), ("lys", "lus"), ("søn", "son"), ("køre", "kore")),
            tip="Tongue forward as for 'ee'/'ay', lips rounded as for 'oo'.",
            why="English has no front rounded vowels, so they collapse onto /u/ and /o/.",
        ),
        Contrast(
            id="r-colour",
            label="Vocalised r vs consonantal r",
            phones=("ɐ̯", "ʁ", "ɐ"),
            pairs=(("mor", "mod"), ("her", "hed"), ("sur", "sud")),
            tip="After a vowel, r is a vowel: it pulls the preceding vowel toward [ɐ].",
            why="English speakers insert a rhotic [ɹ] where Danish has a vowel offglide.",
        ),
        Contrast(
            id="stod",
            label="Stød vs no stød",
            phones=("ʔ",),
            # Pure pairs first: segmentally identical, differing only by stød, with
            # the written final -d silent in Danish. Those are the ones that tell
            # you whether a voice renders stod at all -- if a backend makes them
            # differ by an audible [d], it is spelling-reading, not stod.
            # læser/læsser is kept but is *confounded*: it differs in vowel length
            # as well as stod, so hearing it apart proves nothing about stod.
            pairs=(("man", "mand"), ("hun", "hund"), ("ven", "vend"),
                   ("mor", "mord"), ("bønner", "bønder"), ("læser", "læsser")),
            tip="Stød is creaky voice partway through the syllable — a catch, not a stop. "
                "It is the only thing separating many word pairs.",
            why="English has no phonemic laryngealisation, so learners hear these as homophones "
                "and rely on context.",
        ),
    ),
)


KOREAN = LanguageProfile(
    code="ko",
    name="Korean",
    multi=("tɕ", "tɕh", "tʃ", "tʃh", "ph", "th", "kh", "ɕ"),
    r_variants=frozenset({"ɾ", "r", "l", "ɫ"}),
    r_canonical="ɾ",
    equiv={"ɐ": "a", "ɫ": "l", "q": "k", "ʌ": "ʌ"},
    strip=frozenset({"ʔ", "̃"}),
    example="안녕하세요",
    hvpt_ready=False,
    hvpt_caveat=(
        "Perception trials are disabled for Korean. Beyond the pairs the validator "
        "rejects outright, espeak renders the fortis stops as uvulars (꽁 -> qoŋ), "
        "which is acoustically distinct but is not the Korean contrast — training on "
        "it would build the wrong category. Needs recorded talkers or a Korean TTS voice."
    ),
    g2p_caveat=(
        "NOT production-ready. espeak-ng's Korean G2P collapses the three-way laryngeal "
        "contrast (자다/짜다 both give tɕɐdɐ, 불/뿔 both give puɫ) and does not apply "
        "obligatory sandhi: 신라 should be /silla/ but comes out sinɾɐ, 국물 should be "
        "/kuŋmul/ but comes out ɡuqmuɫ, and a spurious 'q' appears. Korean needs a real "
        "G2P (g2pK or KoG2P) before either diagnosis or stimulus generation can be trusted."
    ),
    tips=_tips([
        (["p", "t", "k", "tɕ"], ["ph", "th", "kh", "tɕh", "b", "d", "ɡ"],
         "Lenis stop: light, slightly breathy, and voiced between vowels — not English b/d/g."),
        (["ph", "th", "kh", "tɕh"], ["p", "t", "k", "tɕ"],
         "Aspirated stop: a strong puff of air, stronger than English p/t/k."),
        (["ʌ"], ["o", "oː", "ɔ", "ə"],
         "ㅓ is unrounded — spread the lips; ㅗ is rounded. Do not merge them."),
        (["ɯ"], ["u", "uː", "ʊ"],
         "ㅡ is unrounded — say 'oo' with the lips spread flat."),
        (["ɾ"], ["ɹ", "r", "l"],
         "ㄹ between vowels is a quick tap, like the tt in American 'butter'."),
        (["l"], ["ɹ", "ɾ"], "ㄹ at the end of a syllable is a clear [l], tongue tip to the ridge."),
        (["ŋ"], ["n", "ɡ"], "Syllable-final ㅇ is [ŋ] with no hard g release."),
    ]),
    contrasts=(
        Contrast(
            id="three-way-stops",
            label="Lenis / fortis / aspirated stops",
            phones=("p", "p*", "ph"),
            pairs=(("불", "뿔", "풀"), ("달", "딸", "탈"), ("자다", "짜다", "차다"), ("공", "꽁", "콩")),
            tip="Three categories where English has two: lenis (light, breathy), "
                "fortis (tense, no aspiration, high pitch onset), aspirated (strong puff).",
            why="English maps all three onto its two-way voicing contrast; this is the "
                "hardest and most persistent Korean perception problem for English speakers.",
        ),
        Contrast(
            id="o-vs-eo",
            label="ㅗ vs ㅓ",
            phones=("o", "ʌ"),
            pairs=(("보다", "버다"), ("공", "검"), ("소리", "서리")),
            tip="ㅗ is rounded and back; ㅓ is unrounded and lower.",
            why="Both land inside the English /ɔ/-/ʌ/ region and get merged.",
        ),
        Contrast(
            id="u-vs-eu",
            label="ㅜ vs ㅡ",
            phones=("u", "ɯ"),
            pairs=(("쿠키", "크키"), ("불", "블"), ("수", "스")),
            tip="ㅜ is rounded; ㅡ is the same height with the lips spread.",
            why="English has no unrounded high back vowel, so ㅡ is heard as ㅜ.",
        ),
    ),
)

FRENCH = LanguageProfile(
    code="fr",
    name="French",
    g2p_code="fr-fr",      # the phonemiser rejects "fr"
    voice="fr",            # the synthesiser rejects "fr-fr"
    multi=("ɑ̃", "ɛ̃", "ɔ̃", "œ̃", "wa", "wɛ̃", "ɥi", "tʃ", "dʒ"),
    r_variants=frozenset({"r", "ʀ", "ʁ", "ɾ"}),
    r_canonical="ʁ",
    equiv={
        "ɒ": "ɔ",        # espeak occasionally emits the English open-back vowel
        "a": "a", "ɑ": "a",
        "œ̃": "ɛ̃",       # brun/brin merger: standard in most of France today
    },
    # French has no phonemic glottal stop; length is not phonemic either, but
    # espeak marks it on some words (côte -> koːt), so length marks are kept as
    # part of the token and simply never contrast.
    strip=frozenset({"ʔ"}),
    example="Je voudrais une bière",
    g2p_caveat=(
        "Good. Nasal vowels, the front rounded series and the mid-vowel splits all "
        "come through. Two quirks: espeak switches language on a few words that look "
        "English (dos -> '(en)dɒs(fr)', stripped by the tokeniser), and it renders "
        "jeûne/jeune as a length difference (ʒøːn/ʒøn) rather than the ø/œ quality "
        "difference French actually has — so that pair is not used."
    ),
    tips=_tips([
        (["y"], ["u", "uː", "ʊ", "ju", "juː"],
         "u (as in tu): say 'ee' and round your lips without moving your tongue. "
         "It is not 'oo' — that is the spelling ou."),
        (["u"], ["y", "ʊ", "ʌ"],
         "ou is a true 'oo' with the tongue pulled back — keep it distinct from u."),
        (["ø", "œ"], ["o", "ɔ", "ɜ", "ɜː", "ʌ"],
         "eu: say 'ay' and round your lips. Not the English 'uh'."),
        (["ɑ̃"], ["a", "ɑ", "an", "ɔ̃", "ʌn"],
         "an/en is one nasal vowel — air through the nose, and no [n] at the end."),
        (["ɛ̃"], ["ɛ", "ɛn", "an", "æn"],
         "in/ain is one nasal vowel — no [n] consonant after it."),
        (["ɔ̃"], ["ɔ", "ɔn", "on", "ɑ̃"],
         "on is one nasal vowel, rounder and higher than an — no [n] at the end."),
        (["e"], ["ɛ", "eɪ"],
         "é is close and pure — no glide toward 'ay'."),
        (["ɛ"], ["e", "eɪ"],
         "è/ai is open — jaw lower than for é."),
        (["o"], ["ɔ", "oʊ"],
         "This o is close and pure (saute, beau) — no glide."),
        (["ɔ"], ["o", "oʊ", "ɑ"],
         "This o is open (sotte, pomme) — jaw lower, lips less rounded."),
        (["ʁ"], ["ɹ", "r"],
         "French r is uvular — friction at the very back of the throat, tongue tip down."),
        (["ʒ"], ["dʒ", "z"],
         "j/ge is a soft 'zh' as in 'measure' — never the hard English 'j'."),
        (["ʃ"], ["tʃ"],
         "ch is 'sh', never the English 'ch'."),
        (["p", "t", "k"], ["ph", "th", "kh"],
         "French p/t/k are unaspirated — no puff of air. Hold the sound back."),
    ]),
    contrasts=(
        Contrast(
            id="u-vs-ou",
            label="u /y/ vs ou /u/",
            phones=("y", "u"),
            pairs=(("tu", "tout"), ("rue", "roue"), ("pu", "pou"),
                   ("bu", "boue"), ("vu", "vous")),
            tip="/y/ is 'ee' with rounded lips; /u/ is a back 'oo'. The tongue moves, "
                "not just the lips.",
            why="English has no front rounded vowel, so /y/ is assimilated to /u/ — "
                "the single most persistent French perception problem for English speakers.",
        ),
        Contrast(
            id="nasal-vowels",
            label="The three nasal vowels an / in / on",
            phones=("ɑ̃", "ɛ̃", "ɔ̃"),
            pairs=(("sans", "sain", "son"), ("banc", "bain", "bon"),
                   ("vent", "vin", "vont"), ("lent", "lin", "long")),
            tip="Three distinct nasal vowels, differing in tongue height and rounding. "
                "None of them ends in an [n].",
            why="English has no phonemic nasal vowels; learners hear a vowel plus /n/ "
                "and merge the three into one or two categories.",
        ),
        Contrast(
            id="nasal-vs-oral",
            label="Nasal vs oral vowel",
            phones=("ɛ̃", "ɛ", "ɔ̃", "o"),
            pairs=(("paix", "pain"), ("fait", "faim"), ("beau", "bon"), ("sait", "saint")),
            tip="The nasal member sends air through the nose from the start of the vowel; "
                "the oral one does not. Neither has a consonant after it.",
            why="Without a nasal-vowel category, learners either miss the contrast or "
                "insert an [n] that French does not have.",
        ),
        Contrast(
            id="mid-vowels",
            label="Close /o/ vs open /ɔ/",
            phones=("o", "ɔ"),
            pairs=(("saute", "sotte"), ("paume", "pomme"), ("côte", "cotte"), ("haute", "hotte")),
            tip="Close /o/ has rounded, tense lips; open /ɔ/ drops the jaw and slackens them.",
            why="English 'o' is a diphthong, so both French vowels are heard as one gliding sound.",
        ),
        Contrast(
            id="e-vs-e-grave",
            label="é /e/ vs è /ɛ/",
            phones=("e", "ɛ"),
            pairs=(("les", "lait"), ("thé", "taie"), ("fée", "fait"), ("ces", "sait")),
            tip="é is close and pure; è is open. Neither glides the way English 'ay' does.",
            why="Both land inside the English /eɪ/ category, and the glide masks the difference.",
        ),
    ),
)

PROFILES: dict[str, LanguageProfile] = {p.code: p for p in (GERMAN, DANISH, KOREAN, FRENCH)}
DEFAULT = "de"


def get(code: str | None = None) -> LanguageProfile:
    code = (code or DEFAULT).lower()
    if code not in PROFILES:
        raise KeyError(f"no profile for {code!r}; have {sorted(PROFILES)}")
    return PROFILES[code]
