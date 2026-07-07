"""Unicode normalization helpers used before tokenization."""

from __future__ import annotations

import unicodedata

SUPPORTED_FORMS = {"NFC", "NFKC", "NFD", "NFKD"}


def validate_unicode(text: str) -> str:
    """Return *text* if it is a valid Python Unicode string.

    Python ``str`` values are Unicode, but they may contain isolated surrogate
    code points when constructed manually. Those values cannot be encoded as
    UTF-8 and are almost always accidental in tokenizer input.

    Examples:
        >>> validate_unicode("cafe")
        'cafe'
    """

    if not isinstance(text, str):
        raise TypeError("text must be a str")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("text contains invalid isolated surrogate code points") from exc
    return text


def normalize_text(text: str, form: str = "NFC") -> str:
    """Normalize Unicode text with one of NFC, NFKC, NFD, or NFKD.

    NFC is the default because it preserves ordinary visual text while making
    canonically equivalent strings, such as composed and decomposed ``e`` with
    acute accent, compare equal.
    """

    validate_unicode(text)
    normalized_form = form.upper()
    if normalized_form not in SUPPORTED_FORMS:
        raise ValueError(f"unsupported normalization form {form!r}; use one of {sorted(SUPPORTED_FORMS)}")
    return unicodedata.normalize(normalized_form, text)


def explain_codepoints(text: str) -> list[dict]:
    """Describe each Unicode code point in *text* for debugging.

    Returns dictionaries with the character, code point, Unicode name, category,
    combining class, and bidirectional class.
    """

    validate_unicode(text)
    rows = []
    for index, char in enumerate(text):
        rows.append(
            {
                "index": index,
                "char": char,
                "codepoint": f"U+{ord(char):04X}",
                "name": unicodedata.name(char, "<unassigned>"),
                "category": unicodedata.category(char),
                "combining": unicodedata.combining(char),
                "bidirectional": unicodedata.bidirectional(char),
            }
        )
    return rows
