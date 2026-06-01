# louis is installed via apt (python3-louis) on Linux/Community Cloud.
# It is NOT available via pip and is NOT installed on Windows dev machines.
# We use a lazy import inside translate() so that:
#   - importing this module on Windows does NOT crash
#   - tests can monkeypatch sys.modules['louis'] before calling translate()
#   - on the deployed server (Linux), import louis works fine at call time

GRADE_TABLES: dict[str, list[str]] = {
    "grade2":   ["braille-patterns.cti", "en-ueb-g2.ctb"],
    "grade1":   ["braille-patterns.cti", "en-ueb-g1.ctb"],
    "nemeth":   ["braille-patterns.cti", "nemeth.ctb"],
    "computer": ["braille-patterns.cti", "en-us-comp8-ext.utb"],
}


def translate(braille_unicode: str, grade: str = "grade2") -> str:
    """
    Convert a Braille Unicode string to English text using liblouis.

    Uses backTranslateString() — the Braille→English direction.
    translateString() is the wrong direction (English→Braille).

    Grade 2 back-translation accuracy is ~95%+ for standard prose but
    degrades on proper nouns and isolated cells due to contraction ambiguity.

    Falls back to character-by-character translation on bulk failure;
    failed characters are replaced with [?].

    Returns "[liblouis unavailable]" if python3-louis is not installed
    (e.g. Windows dev machine without the apt package).
    """
    if not braille_unicode.strip():
        return ""

    try:
        import louis  # lazy: only imported at call time, not at module load
    except ImportError:
        return "[liblouis unavailable — install python3-louis on Linux]"

    tables = GRADE_TABLES.get(grade, GRADE_TABLES["grade2"])

    try:
        return louis.backTranslateString(tables, braille_unicode)
    except Exception:
        parts: list[str] = []
        for char in braille_unicode:
            if char == '\n':
                parts.append('\n')
                continue
            try:
                parts.append(louis.backTranslateString(tables, char))
            except Exception:
                parts.append('[?]')
        return "".join(parts)
