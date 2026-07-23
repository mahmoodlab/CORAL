"""Marker-name cleaning, normalization, and exact matching.

``clean_marker_name`` puts a raw marker name into a consistent form
(lower case; separators collapsed to ``_``) so that spelling and
punctuation variants of the same marker line up.

``normalize_key`` and ``match_marker`` are the exact-key matcher: they
resolve a raw name against a canonical key index by exact key only, so
``CD-8``, ``CD_8``, ``CD 8``, and ``CD8`` all match ``cd8``. On a miss
they return advisory suggestions for the user to confirm rather than
silently guessing — which avoids nonsensical matches like ``CD24`` to
``CD4``. Prefer these for resolving channel names.

``normalize_marker_name`` is an older resolver that adds an alias step
and a fuzzy fallback; it is still used in a few places. CORAL ships no
default alias map, so a per-cycle nuclear stain stays distinct and is
never silently collapsed onto another (for example, ``Hoechst`` is not
renamed to ``DAPI``).
"""

from __future__ import annotations

import difflib
from typing import Literal

__all__ = [
    "MatchLevel",
    "MatchStatus",
    "NOVEL_TOKEN",
    "RESOLVED_TOKEN",
    "REVIEW_TOKEN",
    "clean_marker_name",
    "match_marker",
    "normalize_key",
    "normalize_marker_name",
]

#: Marker-status tokens — the values of the marker_map ``status`` column
#: and of each marker's resolution level. A marker is RESOLVED (mapped to
#: a canonical/recognised name), REVIEW (still needs the user's eyes), or
#: NOVEL (a user-declared marker not in the registry).
RESOLVED_TOKEN = "RESOLVED"
REVIEW_TOKEN = "REVIEW"
NOVEL_TOKEN = "NOVEL"

MatchLevel = Literal["exact", "alias", "fuzzy", "none"]
MatchStatus = Literal["RESOLVED", "REVIEW", "NOVEL"]

# Name-cleaning translation table: separators collapse to ``_``, ``/``
# drops out, ``α`` -> ``a``.
_CLEAN_TABLE = str.maketrans(
    {"-": "_", " ": "_", ":": "_", "(": "_", ")": "_", "α": "a", "/": ""}
)


def clean_marker_name(name: str) -> str:
    """Normalize a raw marker name to a consistent form.

    Puts a marker name into one canonical spelling so that variants of
    the same marker line up against a single registry key. It lower-cases
    the name, collapses separators (spaces, dashes, colons, parentheses)
    to underscores, drops slashes, replaces the Greek ``α`` with ``a``,
    and strips surrounding whitespace. This is pure formatting — it never
    changes the biological identity of the marker.

    Args:
        name: Raw marker or channel name.

    Returns:
        The cleaned name (lower-case, underscore-separated).

    Example:
        ``HLA-DR`` becomes ``hla_dr`` and ``CD8α(Opal650)`` becomes
        ``cd8a_opal650_``; an already-clean name like ``Hoechst1`` simply
        lower-cases to ``hoechst1``.
    """
    return name.lower().translate(_CLEAN_TABLE).strip()


def normalize_key(name: str) -> str:
    """Collapse-separators match key — ``clean_marker_name`` then drop ``_``.

    ``clean_marker_name`` maps separators to ``_`` but keeps them, so
    ``CD-8`` (``cd_8``) would not match ``CD8`` (``cd8``). The match key
    removes the ``_`` so separator-only differences collapse:
    ``CD-8`` = ``CD_8`` = ``CD 8`` = ``CD8`` -> ``cd8``. Used only for
    matching; the stored canonical name stays the registry display form.

    Args:
        name: Raw marker or channel name.

    Returns:
        The separator-free, lower-case match key.

    Example:
        >>> normalize_key("CD-8")
        'cd8'
        >>> normalize_key("HLA-DR")
        'hladr'
        >>> normalize_key("CD-8") == normalize_key("CD8")
        True
    """
    return clean_marker_name(name).replace("_", "")


def match_marker(
    name: str,
    key_index: dict[str, str],
    *,
    n_suggestions: int = 2,
) -> tuple[str | None, MatchStatus, list[str]]:
    """Resolve a raw marker to a canonical name by **exact key only**.

    A canonical name is assigned only when ``normalize_key(name)`` is an
    exact key of ``key_index`` (built by
    :func:`coral.markers.registry.canonical_key_index`). There is **no**
    edit-distance fallback — that is what produced nonsensical matches
    such as ``CD24 -> CD4``. A miss returns up to ``n_suggestions``
    **advisory** candidates for the user to confirm or override; they are
    never applied automatically.

    Args:
        name: Raw marker or channel name.
        key_index: ``{normalize_key(canonical): canonical_display}``.
        n_suggestions: Max advisory candidates returned for a miss.

    Returns:
        ``(canonical | None, status, suggestions)`` — ``(display,
        RESOLVED_TOKEN, [])`` on a hit; ``(None, REVIEW_TOKEN, [c1, c2])``
        on a miss.

    Example:
        >>> idx = {"cd8": "CD8", "cd4": "CD4", "cd40": "CD40"}
        >>> match_marker("CD-8", idx)
        ('CD8', 'RESOLVED', [])
        >>> name, status, _ = match_marker("CD24", idx)
        >>> (name, status)
        (None, 'REVIEW')
    """
    key = normalize_key(name)
    canonical = key_index.get(key)
    if canonical is not None:
        return canonical, RESOLVED_TOKEN, []
    return None, REVIEW_TOKEN, _suggest(key, key_index, n_suggestions)


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the shared leading run of two strings."""
    n = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        n += 1
    return n


def _suggest(key: str, key_index: dict[str, str], n: int) -> list[str]:
    """Up to ``n`` **advisory** canonical candidates for an unmatched key.

    Ranked by substring containment and shared-prefix length first, with a
    ``difflib`` ratio only as a last-resort tiebreak. Advisory only — never
    auto-applied, so a coincidental near-match (``CD24``/``CD4``) is a hint
    to verify, not an assignment.
    """

    def score(cand_key: str) -> tuple[int, int, float]:
        contains = int(key in cand_key or cand_key in key)
        prefix = _common_prefix_len(key, cand_key)
        ratio = difflib.SequenceMatcher(None, key, cand_key).ratio()
        return contains, prefix, ratio

    ranked = sorted(key_index, key=score, reverse=True)
    return [key_index[k] for k in ranked[:n]]


def normalize_marker_name(
    name: str,
    canonical_names: list[str] | None = None,
    renaming_dict: dict[str, str] | None = None,
    fuzzy_threshold: float = 0.8,
) -> tuple[str, MatchLevel]:
    """Resolve a raw marker name against an optional canonical list.

    Resolution order, after ``clean_marker_name``:

    1. **exact** — cleaned name is in ``canonical_names``.
    2. **alias** — cleaned name is a key of the caller-supplied
       ``renaming_dict`` (CORAL ships no default; see module docstring).
    3. **fuzzy** — closest ``difflib`` match in ``canonical_names`` at
       or above ``fuzzy_threshold`` (only for names >= 3 chars).
    4. **none** — no match; the cleaned name is returned unchanged.

    Prefer :func:`match_marker` for new code — it drops the fuzzy step
    (the source of ``CD24 -> CD4``-style errors) for explicit review.
    With neither ``canonical_names`` nor ``renaming_dict`` the result is
    always ``(cleaned, "none")`` — it never invents a mapping.

    Args:
        name: Raw marker or channel name.
        canonical_names: Cleaned canonical names to match against
            (e.g. from the registry). ``None`` skips exact + fuzzy.
        renaming_dict: Optional caller alias map keyed by cleaned
            name. ``None`` skips the alias step.
        fuzzy_threshold: ``difflib`` cutoff in ``[0, 1]`` for fuzzy.

    Returns:
        A ``(resolved_name, match_level)`` pair.

    Example:
        >>> normalize_marker_name("CD8a", canonical_names=["cd8a"])
        ('cd8a', 'exact')
        >>> normalize_marker_name("Hoechst1")
        ('hoechst1', 'none')
        >>> normalize_marker_name(
        ...     "Hoechst1", canonical_names=["dapi", "hoechst1"]
        ... )
        ('hoechst1', 'exact')
    """
    cleaned = clean_marker_name(name)
    if canonical_names and cleaned in canonical_names:
        return cleaned, "exact"
    if renaming_dict and cleaned in renaming_dict:
        return renaming_dict[cleaned], "alias"
    if canonical_names and len(cleaned) >= 3:
        close = difflib.get_close_matches(
            cleaned, canonical_names, n=1, cutoff=fuzzy_threshold
        )
        if close:
            return close[0], "fuzzy"
    return cleaned, "none"
