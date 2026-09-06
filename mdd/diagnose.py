"""(canonical, realised) -> learner-facing tip, driven by the language profile.

Tip tables live in `mdd.languages`. Deletions are looked up as (canonical, "")
and insertions as ("", realised), so a profile can give a targeted message for a
dropped sound — Danish stød being the case that matters.
"""
from .languages import LanguageProfile, get

# Kept for backwards compatibility with callers that imported the German table.
TIPS = get("de").tips


def tip_for(canonical: str | None, realized: str | None,
            profile: LanguageProfile | str | None = None) -> str:
    if not isinstance(profile, LanguageProfile):
        profile = get(profile)
    tips = profile.tips

    if canonical is None:
        return tips.get(("", realized), f"Extra sound [{realized}] inserted.")
    if realized is None:
        return tips.get((canonical, ""), f"Sound [{canonical}] was dropped.")
    if canonical.rstrip("ː") == realized.rstrip("ː"):
        if canonical.endswith("ː"):
            return f"Vowel [{canonical}] should be long — hold it about twice as long."
        return f"Vowel [{canonical}] should be short."
    return tips.get((canonical, realized), f"Expected [{canonical}], heard [{realized}].")
