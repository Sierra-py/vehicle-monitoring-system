"""
Whitelist matching with tolerance for the specific kind of noise this OCR
pipeline actually produces — not arbitrary fuzzy matching.

Why not plain edit-distance <=2 against the whole whitelist: two real, different
plates can legitimately be edit-distance 1-2 apart (e.g. MH20BY3665 vs
MH20BY3660). Allowing ANY substitution within that distance risks a misread of
vehicle A's plate fuzzy-matching onto vehicle B's whitelist entry — a real
security problem, not just an accuracy nitpick.

So matching here is restricted to CHARACTER-CLASS-PLAUSIBLE substitutions only:
the same confusions your OCR evaluation actually observed (0/O, 1/I, 5/S, 8/B),
each substitution only within the same character class (digit-for-digit or
letter-for-letter, never digit-for-letter) — not "any character can become any
other character within N edits."

Three possible outcomes per plate reading, matching the review-queue design in
db_models.py:
  - exact match            -> match_type="exact"
  - unique plausible match  -> match_type="fuzzy", requires_review=False
  - two+ whitelist entries equally close -> match_type="ambiguous", requires_review=True
  - nothing close enough    -> match_type="no_match"
"""
import json
from dataclasses import dataclass
from pathlib import Path

# Observed OCR confusion pairs — see 07-23 eval sessions where these specific
# substitutions accounted for nearly all real (non-labeling-error) mismatches.
# Each pair is bidirectional and stays within one character class deliberately.
PLAUSIBLE_CONFUSIONS = {
    frozenset({"0", "O"}),
    frozenset({"1", "I"}),
    frozenset({"5", "S"}),
    frozenset({"8", "B"}),
}


def _is_plausible_substitution(a: str, b: str) -> bool:
    if a == b:
        return True
    return frozenset({a, b}) in PLAUSIBLE_CONFUSIONS


def _restricted_edit_distance(s1: str, s2: str, max_distance: int = 2):
    """
    Levenshtein distance, but a substitution only counts as "1 edit" if it's a
    plausible OCR confusion (see _is_plausible_substitution) — any other
    character mismatch makes the strings simply not comparable via this path
    (returns None), rather than a large-but-finite distance. This is
    deliberate: an implausible substitution shouldn't just score worse, it
    should be treated as "not the kind of noise this is meant to tolerate."

    Returns the edit distance (int) if within max_distance and every
    substitution used was plausible, else None.
    """
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > max_distance:
        return None

    # standard DP edit-distance table, but substitution cost is 1 only if
    # plausible, else effectively blocks that path (treated as insert+delete,
    # cost 2 — a length-preserving character swap doesn't get to sneak through
    # as a cheap "substitution" unless it's on the plausible list)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub_cost = 1 if _is_plausible_substitution(s1[i - 1], s2[j - 1]) else 2
                dp[i][j] = min(
                    dp[i - 1][j] + 1,          # deletion
                    dp[i][j - 1] + 1,          # insertion
                    dp[i - 1][j - 1] + sub_cost,  # substitution
                )

    distance = dp[len1][len2]
    return distance if distance <= max_distance else None


@dataclass
class WhitelistMatchResult:
    whitelist_match: str | None
    match_type: str  # "exact" | "fuzzy" | "ambiguous" | "no_match"
    match_edit_distance: int | None
    requires_review: bool
    review_reason: str | None


def load_whitelist(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    return list(data.keys())


def match_plate(plate_text: str, whitelist: list[str], max_distance: int = 2) -> WhitelistMatchResult:
    if not plate_text:
        return WhitelistMatchResult(None, "no_match", None, False, None)

    if plate_text in whitelist:
        return WhitelistMatchResult(plate_text, "exact", 0, False, None)

    candidates = []  # (whitelist_plate, distance)
    for entry in whitelist:
        distance = _restricted_edit_distance(plate_text, entry, max_distance)
        if distance is not None:
            candidates.append((entry, distance))

    if not candidates:
        return WhitelistMatchResult(None, "no_match", None, False, None)

    candidates.sort(key=lambda c: c[1])
    best_distance = candidates[0][1]
    best_matches = [c for c in candidates if c[1] == best_distance]

    if len(best_matches) > 1:
        tied = ", ".join(c[0] for c in best_matches)
        return WhitelistMatchResult(
            None, "ambiguous", best_distance, True,
            f"'{plate_text}' is equally close (distance {best_distance}) to multiple "
            f"whitelist entries: {tied}. Needs human review to disambiguate."
        )

    matched_plate = best_matches[0][0]
    return WhitelistMatchResult(
        matched_plate, "fuzzy", best_distance, False,
        f"OCR read '{plate_text}', fuzzy-matched to whitelisted '{matched_plate}' "
        f"(distance {best_distance}, plausible OCR confusion)."
    )