"""Text coverage by known words."""
from mdd import vocab as V


def known(lang, *words):
    out = set()
    for w in words:
        out |= {V.lemma(w, lang), w.casefold()}
    return out


def test_tokens_split_french_elision_and_german_compounds():
    assert V.tokens("J'ai vu l'homme aujourd'hui.", "fr") == ["J", "ai", "vu", "l", "homme",
                                                              "aujourd'hui"]
    assert V.tokens("Kinder-Garten 2024!", "de") == ["Kinder", "Garten"]


def test_subtitle_files_are_reduced_to_their_text():
    srt = "1\n00:00:01,000 --> 00:00:03,000\n<i>Hallo</i> [Musik]\n\n2\n00:00:04,000 --> 00:00:05,000\nTschüss"
    vtt = "WEBVTT\n\n00:01.000 --> 00:03.000\nBonjour\n"
    assert V.strip_markup(srt).split() == ["Hallo", "Tschüss"]
    assert V.strip_markup(vtt).split() == ["Bonjour"]


def test_inflected_forms_count_as_their_word():
    rep = V.analyse("Er ging gestern nach Hause. Die Häuser waren alt.", "de",
                    known("de", "er", "gehen", "gestern", "nach", "Haus", "die", "sein", "alt"))
    assert rep.coverage == 1.0, [u.form for u in rep.unknown]


def test_participles_the_lemmatiser_misses_still_match():
    """simplemma leaves 'gefahren' unchanged."""
    rep = V.analyse("Wir sind gefahren und haben gesagt.", "de",
                    known("de", "wir", "sein", "fahren", "und", "haben", "sagen"))
    assert rep.coverage == 1.0, [u.form for u in rep.unknown]


def test_unknown_words_most_common_first_with_their_sentence():
    rep = V.analyse("Das Wetter ist schön. Die Bahnsteigkante ist nass.", "de",
                    known("de", "der", "die", "das", "sein"))
    forms = [u.form for u in rep.unknown]
    assert forms.index("Wetter") < forms.index("Bahnsteigkante")
    wetter = next(u for u in rep.unknown if u.form == "Wetter")
    assert wetter.sentence == "Das Wetter ist schön." and wetter.count == 1


def test_assuming_the_most_common_words_covers_function_words():
    text = "Ich und du, wir sind hier."
    assert V.analyse(text, "de", set(), assume_top=0).coverage == 0
    assert V.analyse(text, "de", set(), assume_top=500).coverage == 1.0


def test_french_names_mid_sentence_are_not_counted():
    rep = V.analyse("Hier, nous avons vu Jean-Baptiste et Marie à Paris.", "fr", set(),
                    assume_top=1000)
    assert set(rep.names) == {"Jean-Baptiste", "Marie", "Paris"}
    assert rep.coverage == 1.0


def test_verdict_thresholds():
    r = V.Report("de", 100, 98)
    assert r.verdict().startswith("comfortable")
    assert V.Report("de", 100, 96).verdict().startswith("workable")
    assert V.Report("de", 100, 91).verdict().startswith("hard")
    assert V.Report("de", 100, 80).verdict().startswith("too hard")


def test_anki_plain_text_export():
    text = ('#separator:tab\n#html:true\ndie Tür\tthe door [sound:t.mp3]\n'
            '"<b>der</b> Bahnhof"\tstation\nIch möchte fünf Brötchen.\tI want five rolls.\n')
    rows = V.read_anki_export(text)
    assert rows[0] == ["die Tür", "the door"] and len(rows) == 3
    assert V.words_from_anki(rows, 0, "word", "de") == ["Tür", "Bahnhof", "Ich"]
    assert "Brötchen" in V.words_from_anki(rows, 0, "sentence", "de")
    assert V.words_from_anki(rows, 5, "word", "de") == []


def test_known_words_persist(tmp_path):
    k = V.Known("de", root=tmp_path)
    assert k.add(["die Häuser"]) > 0
    k.save()
    again = V.Known("de", root=tmp_path)
    assert {"haus", "häuser"} <= again.words


def test_export_for_anki():
    rep = V.analyse("Die Bahnsteigkante ist nass.", "de", known("de", "die", "sein"))
    rows = [line.split("\t") for line in V.export_tsv(rep.unknown).splitlines()]
    assert rows[0][0] in {"Bahnsteigkante", "nass"} and rows[0][2] == "Die Bahnsteigkante ist nass."
