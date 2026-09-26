"""Name and address normalisation.

Design rules:
- Never overwrite the raw text; every cleaned form is an extra column.
- Keep a full normalised name and a "core" name (legal forms and honorifics
  removed) side by side, so distinct businesses are not collapsed.
- Rules are general formatting conventions that work for any country label.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from indic_transliteration import sanscript
from unidecode import unidecode

from . import lexicon as lx

# --------------------------------------------------------------------------- #
# Indic transliteration
# --------------------------------------------------------------------------- #
_INDIC_BLOCKS = [
    (0x0900, 0x097F, sanscript.DEVANAGARI, True),
    (0x0980, 0x09FF, sanscript.BENGALI, True),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI, True),
    (0x0A80, 0x0AFF, sanscript.GUJARATI, True),
    (0x0B00, 0x0B7F, sanscript.ORIYA, True),
    (0x0B80, 0x0BFF, sanscript.TAMIL, False),
    (0x0C00, 0x0C7F, sanscript.TELUGU, False),
    (0x0C80, 0x0CFF, sanscript.KANNADA, False),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM, False),
]
_NUKTAS = dict.fromkeys(map(ord, "़়਼઼଼"), None)
_INDIC_RUN = re.compile(r"[ऀ-ൿ]+")
_VOWELS = set("aeiou")
_KEEP_SCHWA_AFTER = set("ryv")


def _script_of(ch: str):
    cp = ord(ch)
    for lo, hi, script, drops_schwa in _INDIC_BLOCKS:
        if lo <= cp <= hi:
            return script, drops_schwa
    return None, False


def _fix_romanised(word: str, drop_schwa: bool) -> str:
    word = re.sub(r"[^a-z]", "", word.lower())
    # Anusvara before a consonant is pronounced as the matching nasal.
    word = re.sub(r"m(?=[^aeioupbm])", "n", word)
    if drop_schwa and len(word) > 3 and word.endswith("a") and word[-2] not in _VOWELS:
        cluster = word[-3] not in _VOWELS and word[-2] in _KEEP_SCHWA_AFTER
        if not cluster:
            word = word[:-1]
    return word


@lru_cache(maxsize=200_000)
def _transliterate_run(run: str) -> str:
    script, drop_schwa = _script_of(run[0])
    if script is None:
        return run
    roman = unidecode(sanscript.transliterate(run.translate(_NUKTAS), script, sanscript.IAST))
    return _fix_romanised(roman, drop_schwa)


def transliterate_indic(text: str) -> tuple[str, bool]:
    """Romanise every run of Indic-script characters; report whether any was found."""
    if not _INDIC_RUN.search(text):
        return text, False
    return _INDIC_RUN.sub(lambda m: _transliterate_run(m.group(0)), text), True


def to_ascii_lower(text: str) -> tuple[str, bool]:
    text = unicodedata.normalize("NFKC", text or "")
    text, had_indic = transliterate_indic(text)
    return unidecode(text).lower(), had_indic


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")
_DOMAIN = re.compile(r"\b(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|biz|info|co\.in|in|co|fr|us|io)\b")
_BRACKETED = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
_ALIAS_SPLIT = re.compile(r"\s(?:" + "|".join(re.escape(m) for m in lx.ALIAS_MARKERS) + r")\s")
_NON_WORD = re.compile(r"[^a-z0-9]+")
_HASH_TAG = re.compile(r"#\s*\d+")


def _fix_leet(token: str) -> str:
    """'h0rizon' -> 'horizon', but leave real codes like 'cs1464' or '24x7' alone."""
    if token.isdigit() or token.isalpha() or _ORDINAL.match(token):
        return token
    letters = sum(c.isalpha() for c in token)
    digits = len(token) - letters
    if letters >= 2 and digits <= 2 and letters > digits:
        return token.translate(_LEET)
    return token


def _join_single_letters(tokens: list[str]) -> list[str]:
    """'l l c' -> 'llc', 'a b c' -> 'abc' (dotted initials after punctuation removal)."""
    out, buf = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if buf:
            out.append("".join(buf))
            buf = []
        out.append(t)
    if buf:
        out.append("".join(buf))
    return out


def _canonical_legal(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        pair = tuple(tokens[i : i + 2])
        if len(pair) == 2 and pair in lx.LEGAL_BIGRAMS:
            out.extend(lx.LEGAL_BIGRAMS[pair].split())
            i += 2
            continue
        out.extend(lx.LEGAL_FORMS.get(tokens[i], tokens[i]).split())
        i += 1
    return out


_LEGAL_CANON = set(" ".join(lx.LEGAL_FORMS.values()).split())


def _name_tokens(text: str) -> list[str]:
    text = text.replace("&", " and ").replace("@", " at ")
    text = _DOMAIN.sub(r" \1 ", text)
    tokens = [_fix_leet(t) for t in _NON_WORD.sub(" ", text).split()]
    tokens = [lx.SHARED_TOKENS.get(t, t) for t in tokens]
    return _canonical_legal(_join_single_letters(tokens))


def _dedupe_adjacent(tokens: list[str]) -> list[str]:
    return [t for i, t in enumerate(tokens) if i == 0 or t != tokens[i - 1]]


_EDGE_STOPWORDS = {"and", "et", "the", "of", "de", "la", "le", "du"}
# Matched on the text before initials are joined, so "M R Sun" is not read as "Mr".
_LEADING_HONORIFIC = re.compile(
    r"^\W*(?:" + "|".join(re.escape(h) for h in sorted(lx.HONORIFICS, key=len, reverse=True)) + r")\.?\s+"
)


def _core_tokens(text: str) -> list[str]:
    text = _LEADING_HONORIFIC.sub("", text.strip())
    toks = _name_tokens(_HASH_TAG.sub(" ", _BRACKETED.sub(" ", text)))
    toks = _dedupe_adjacent([t for t in toks if t not in _LEGAL_CANON])
    while toks and toks[0] in _EDGE_STOPWORDS:
        toks = toks[1:]
    while toks and toks[-1] in _EDGE_STOPWORDS:
        toks = toks[:-1]
    return toks


def _initials(text: str) -> str:
    """First letter of each word, counting dotted initials separately: 'M R & X Sun' -> 'mrxs'."""
    text = _LEADING_HONORIFIC.sub("", text.strip()).replace("&", " and ")
    words = _NON_WORD.sub(" ", _BRACKETED.sub(" ", text)).split()
    skip = _LEGAL_CANON | set(lx.LEGAL_FORMS) | _EDGE_STOPWORDS
    return "".join(w[0] for w in words if w[0].isalpha() and w not in skip)


def normalize_name(raw: str) -> dict:
    """Full, core and alias forms of a business name.

    "X dba Y": the trade name X and the alias Y are both kept. name_norm holds
    both parts so search can hit either; name_core / name_alias hold them apart.
    """
    text, had_indic = to_ascii_lower(raw)
    parts = [p for p in _ALIAS_SPLIT.split(f" {text} ") if p.strip()]
    main = parts[0] if parts else ""
    alias = " ".join(parts[1:])

    full = _name_tokens(" ".join(parts))
    core = _core_tokens(main) or _name_tokens(main)
    alias_core = _core_tokens(alias) if alias else []
    legal = sorted({t for t in full if t in _LEGAL_CANON})

    return {
        "name_norm": " ".join(full),
        "name_core": " ".join(core),
        "name_alias": " ".join(alias_core),
        "name_initials": _initials(main),
        "name_legal": " ".join(legal),
        "name_translit": had_indic,
    }


# --------------------------------------------------------------------------- #
# Addresses
# --------------------------------------------------------------------------- #
def _state_key(text: str) -> str:
    """Loose phonetic key so 'Maharashtra', 'maharastra', 'MAHARASHTRA' agree."""
    s = re.sub(r"[^a-z]", "", text.lower())
    s = re.sub(r"(?<=[^aeiou])h", "", s).replace("w", "v")
    s = re.sub(r"(.)\1+", r"\1", s)
    return s[:-1] if s.endswith("a") and len(s) > 4 else s


def _build_state_index():
    full_names: dict[str, str] = {}
    for table in (lx.US_STATES, lx.INDIA_STATES):
        for name in table.values():
            full_names[_state_key(name)] = name
    for alias, name in lx.STATE_NAME_ALIASES.items():
        full_names[_state_key(alias)] = name
    for region in lx.FRANCE_REGIONS:
        full_names[_state_key(region)] = region
    return full_names


_STATE_BY_KEY = _build_state_index()
_CODES_BY_COUNTRY = {"us": lx.US_STATES, "usa": lx.US_STATES, "india": lx.INDIA_STATES}

_ORDINAL_SUFFIX = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")


def _drop_number_markers(tokens: list[str]) -> list[str]:
    """Drop label words like 'h no', 'plot no' only when a number follows them."""
    keep = [True] * len(tokens)
    for i, t in enumerate(tokens):
        if t not in lx.NUMBER_MARKERS:
            continue
        j = i + 1
        while j < len(tokens) and tokens[j] in lx.NUMBER_MARKERS:
            j += 1
        if j < len(tokens) and tokens[j][0].isdigit():
            keep[i] = False
    return [t for t, k in zip(tokens, keep) if k]
_NUMBER = re.compile(r"\d+")


def _match_state(segment: str, country: str) -> str:
    seg = segment.strip()
    codes = _CODES_BY_COUNTRY.get(country)
    if codes and seg in codes:
        return codes[seg]
    return _STATE_BY_KEY.get(_state_key(seg), "") if len(seg) > 2 else ""


def normalize_address(raw: str, country: str) -> dict:
    """Return cleaned address, state, a city guess and the set of numbers it contains."""
    country = (country or "").strip().lower()
    text, had_indic = to_ascii_lower((raw or "").replace("°", " ").replace("º", " "))

    segments = [s for s in (" ".join(_NON_WORD.sub(" ", seg).split()) for seg in text.split(",")) if s]

    # The state is taken from the last matching segment, so a city such as
    # "new delhi" earlier in the address is not mistaken for the state.
    state, state_idx = "", -1
    for i in range(len(segments) - 1, -1, -1):
        state = _match_state(segments[i], country)
        if state:
            state_idx = i
            break
    kept = [s for i, s in enumerate(segments) if i != state_idx]

    numbers = sorted({n.lstrip("0") or "0" for n in _NUMBER.findall(" ".join(kept))}, key=lambda n: (len(n), n))

    tokens_by_seg: list[list[str]] = []
    for seg in kept:
        seg = _ORDINAL_SUFFIX.sub(r"\1", seg)
        toks = []
        for t in _drop_number_markers(seg.split()):
            if t in lx.ADDRESS_NOISE:
                continue
            if t.isdigit():
                t = t.lstrip("0") or "0"
            elif t == "r" and toks and toks[-1].isdigit():
                t = "rue"
            t = lx.ADDRESS_ABBREV.get(t, t)
            t = lx.SHARED_TOKENS.get(t, t)
            t = lx.CITY_ALIASES.get(t, t)
            toks.append(t)
        toks = _dedupe_adjacent(toks)
        if toks:
            tokens_by_seg.append(toks)

    city = ""
    for toks in reversed(tokens_by_seg):
        if not any(c.isdigit() for t in toks for c in t):
            city = " ".join(toks)
            break

    flat = [t for toks in tokens_by_seg for t in toks]
    return {
        "addr_norm": " ".join(flat),
        "addr_state": state,
        "addr_city": city,
        "addr_numbers": " ".join(numbers),
        "addr_missing": not flat,
        "addr_translit": had_indic,
    }


def normalize_record(name: str, address: str, country: str) -> dict:
    out = normalize_name(name)
    out.update(normalize_address(address, country))
    out["country_norm"] = (country or "").strip().lower()
    return out
