"""
app/owasp.py — OWASP LLM Top 10 mapping layer

Maps a scored prompt to the relevant OWASP LLM Top 10 categories based on
matched regex patterns and semantic signals. Returns a list of category codes
so the API response and dashboard can surface framework-level context.

OWASP LLM Top 10 (2025):
  LLM01 — Prompt Injection
  LLM02 — Sensitive Information Disclosure
  LLM03 — Supply Chain
  LLM04 — Data and Model Poisoning
  LLM05 — Improper Output Handling
  LLM06 — Excessive Agency
  LLM07 — System Prompt Leakage
  LLM08 — Vector and Embedding Weaknesses
  LLM09 — Misinformation
  LLM10 — Unbounded Consumption

Reference: https://owasp.org/www-project-top-10-for-large-language-model-applications/
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class OWASPCategory:
    code: str        # e.g. "LLM01"
    name: str        # short display name
    description: str # one-line summary


# ── Registry ──────────────────────────────────────────────────────────────────

CATEGORIES: dict[str, OWASPCategory] = {
    "LLM01": OWASPCategory(
        code="LLM01",
        name="Prompt Injection",
        description="Attacker manipulates the LLM via crafted input to override instructions or hijack behaviour.",
    ),
    "LLM02": OWASPCategory(
        code="LLM02",
        name="Sensitive Information Disclosure",
        description="LLM reveals confidential data, PII, credentials, or system internals in its output.",
    ),
    "LLM05": OWASPCategory(
        code="LLM05",
        name="Improper Output Handling",
        description="Downstream system blindly trusts LLM output, enabling injection into shells, SQL, or HTML.",
    ),
    "LLM06": OWASPCategory(
        code="LLM06",
        name="Excessive Agency",
        description="LLM is granted — or manipulated into exercising — more autonomy or permissions than intended.",
    ),
    "LLM07": OWASPCategory(
        code="LLM07",
        name="System Prompt Leakage",
        description="Attacker extracts the confidential system prompt through direct or indirect elicitation.",
    ),
    "LLM10": OWASPCategory(
        code="LLM10",
        name="Unbounded Consumption",
        description="Prompt causes excessive token generation or recursive calls, leading to DoS or cost abuse.",
    ),
}

# ── Pattern → category mapping ────────────────────────────────────────────────
# Keys are substrings that appear in matched_patterns from scorer.py.
# A single pattern can map to multiple categories.

PATTERN_TO_CATEGORIES: dict[str, list[str]] = {
    # Direct instruction override → LLM01
    "ignore_previous":          ["LLM01"],
    "ignore_instructions":      ["LLM01"],
    "disregard":                ["LLM01"],
    "override":                 ["LLM01"],
    "forget_instructions":      ["LLM01"],
    "new_instructions":         ["LLM01"],
    "your_instructions":        ["LLM01"],

    # Role/persona hijacking → LLM01 + LLM06
    "you_are_now":              ["LLM01", "LLM06"],
    "pretend":                  ["LLM01", "LLM06"],
    "act_as":                   ["LLM01", "LLM06"],
    "roleplay":                 ["LLM01", "LLM06"],
    "jailbreak":                ["LLM01", "LLM06"],
    "dan_mode":                 ["LLM01", "LLM06"],
    "developer_mode":           ["LLM01", "LLM06"],
    "unrestricted":             ["LLM01", "LLM06"],
    "no_restrictions":          ["LLM01", "LLM06"],
    "without_restrictions":     ["LLM01", "LLM06"],

    # System prompt extraction → LLM07
    "reveal_system":            ["LLM07"],
    "show_system":              ["LLM07"],
    "print_system":             ["LLM07"],
    "display_prompt":           ["LLM07"],
    "what_are_your_instructions": ["LLM07"],
    "initial_prompt":           ["LLM07"],
    "system_prompt":            ["LLM07"],
    "confidential_instructions": ["LLM07"],

    # Credential / secret extraction → LLM02
    "api_key":                  ["LLM02"],
    "password":                 ["LLM02"],
    "credentials":              ["LLM02"],
    "access_token":             ["LLM02"],
    "secret":                   ["LLM02"],
    "private_key":              ["LLM02"],

    # Code / command injection → LLM05
    "execute":                  ["LLM05"],
    "eval":                     ["LLM05"],
    "shell":                    ["LLM05"],
    "subprocess":               ["LLM05"],
    "os_command":               ["LLM05"],
    "sql_injection":            ["LLM05"],
    "script_injection":         ["LLM05"],

    # Token/resource abuse → LLM10
    "repeat_forever":           ["LLM10"],
    "infinite_loop":            ["LLM10"],
    "generate_unlimited":       ["LLM10"],
    "token_flood":              ["LLM10"],
    "resource_exhaustion":      ["LLM10"],
}


# ── Public API ────────────────────────────────────────────────────────────────

def map_categories(
    matched_patterns: list[str],
    injection_score: float,
    semantic_score: float,
) -> list[dict]:
    """
    Return a deduplicated, sorted list of OWASP category dicts relevant to
    this request.

    Args:
        matched_patterns: list of pattern names from scorer.py
        injection_score:  blended score (0.0–1.0)
        semantic_score:   semantic layer score (0.0–1.0)

    Returns:
        List of dicts: [{"code": "LLM01", "name": "...", "description": "..."}]
        Empty list if no categories matched.
    """
    codes: set[str] = set()

    # Pattern-based mapping
    for pattern in matched_patterns:
        pattern_lower = pattern.lower()
        for key, category_codes in PATTERN_TO_CATEGORIES.items():
            if key in pattern_lower:
                codes.update(category_codes)

    # Score-based fallback: if something scored high but no patterns named,
    # it's still almost certainly LLM01 (prompt injection is the catch-all)
    if injection_score >= 0.40 and not codes:
        codes.add("LLM01")

    # High semantic score with no regex match → indirect / paraphrased injection
    if semantic_score >= 0.60 and "LLM01" not in codes:
        codes.add("LLM01")

    # Build output — sorted by code for stable ordering
    return [
        {
            "code": cat.code,
            "name": cat.name,
            "description": cat.description,
        }
        for code in sorted(codes)
        if (cat := CATEGORIES.get(code))
    ]
