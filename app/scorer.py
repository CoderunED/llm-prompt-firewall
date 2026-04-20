import re
import logging
from app.config import settings
from app.semantic_scorer import semantic_scorer

logger = logging.getLogger(__name__)

# Each entry: (pattern, weight)
INJECTION_PATTERNS: list[tuple[str, float]] = [
    # Instruction override attempts
    (r"ignore (all |previous |prior |above |your )?(instructions?|prompts?|rules?|guidelines?)", 0.35),
    (r"disregard (all |previous |prior |above |your )?(instructions?|prompts?|rules?)", 0.35),
    (r"forget (all |previous |prior |above |your )?(instructions?|prompts?|rules?|context|guidelines?)", 0.30),
    (r"forget (everything|all of it|what you know)", 0.30),
    (r"do not follow (your )?(instructions?|rules?|guidelines?)", 0.35),
    (r"override (your )?(instructions?|rules?|settings?|system)", 0.35),

    # Persona hijacking
    (r"you are now", 0.25),
    (r"act as (a |an )?(?!helpful|an AI)", 0.20),
    (r"pretend (you are|to be|that you)", 0.20),
    (r"roleplay as", 0.20),
    (r"your (new |true |real |actual )?(role|persona|identity|purpose|goal|objective) is", 0.25),

    # Jailbreak keywords
    (r"\bDAN\b", 0.40),
    (r"jailbreak", 0.40),
    (r"developer mode", 0.35),
    (r"god mode", 0.35),
    (r"unrestricted mode", 0.35),
    (r"no restrictions?", 0.25),
    (r"without (any )?(restrictions?|limits?|filters?|constraints?)", 0.30),

    # System prompt extraction
    (r"reveal (your )?(system |original |initial )?(prompt|instructions?)", 0.40),
    (r"show (me )?(your )?(system |original |initial )?(prompt|instructions?)", 0.35),
    (r"what (are|were) (your )?(system |original |initial )?(prompt|instructions?)", 0.30),
    (r"repeat (your )?(system |original |initial )?(prompt|instructions?)", 0.35),
    (r"print (your )?(system |original |initial )?(prompt|instructions?)", 0.35),

    # Encoding / obfuscation tricks
    (r"base64", 0.20),
    (r"hex(adecimal)?( encoded)?", 0.15),
    (r"rot\s*13", 0.20),

    # Indirect injection markers
    (r"the (above|following) (is|are) (your )?(new |updated )?(instructions?|rules?)", 0.35),
    (r"from now on", 0.15),
    (r"new persona", 0.25),
]


def _compile_patterns() -> list[tuple[re.Pattern, float]]:
    return [
        (re.compile(p, re.IGNORECASE), w)
        for p, w in INJECTION_PATTERNS
    ]


_COMPILED = _compile_patterns()


def score_prompt(prompt: str) -> dict:
    """
    Score a prompt using blended regex + semantic pipeline.

    Returns:
        {
            "injection_score": float,       # blended 0.0 – 1.0
            "risk_level": str,              # "low" | "medium" | "high"
            "matched_patterns": list[str],
            "regex_score": float,           # raw regex-only score
            "semantic_score": float,        # raw semantic-only score
            "closest_phrase": str,          # closest attack phrase from semantic layer
        }
    """
    # ── Regex layer ───────────────────────────────────────────────────────────
    matched = []
    raw_regex = 0.0

    for pattern, weight in _COMPILED:
        if pattern.search(prompt):
            matched.append(pattern.pattern)
            raw_regex += weight

    regex_score = round(min(raw_regex, 1.0), 4)

    # ── Semantic layer ────────────────────────────────────────────────────────
    sem_result = semantic_scorer.score(prompt)
    semantic_score = sem_result["semantic_score"]
    closest_phrase = sem_result["closest_phrase"]

    # ── Blend ─────────────────────────────────────────────────────────────────
    blended = (
        regex_score * settings.regex_weight +
        semantic_score * settings.semantic_weight
    )
    injection_score = round(min(blended, 1.0), 4)

    if injection_score >= 0.5:
        risk_level = "high"
    elif injection_score >= 0.2:
        risk_level = "medium"
    else:
        risk_level = "low"

    logger.info(
        "Scored prompt | regex=%.4f semantic=%.4f blended=%.4f risk=%s matches=%d",
        regex_score, semantic_score, injection_score, risk_level, len(matched),
    )

    return {
        "injection_score": injection_score,
        "risk_level": risk_level,
        "matched_patterns": matched,
        "regex_score": regex_score,
        "semantic_score": semantic_score,
        "closest_phrase": closest_phrase,
    }
