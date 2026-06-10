"""
app/routes/analyze.py
"""

import json
import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.scorer import score_prompt
from app.semantic_scorer import SemanticScorer
from app.llm_client import call_llm, LLMError
from app.logger import log_request
from app.database import write_request
from app.owasp import map_categories

logger = logging.getLogger(__name__)
router = APIRouter()
_semantic = SemanticScorer()


# ── Schemas ───────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are a helpful assistant."


class LLMMeta(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class OWASPEntry(BaseModel):
    code: str
    name: str
    description: str


class AnalyzeResponse(BaseModel):
    status: str
    prompt: str
    injection_score: float
    regex_score: float
    semantic_score: float
    risk_level: str
    blocked: bool
    matched_patterns: list[str]
    closest_phrase: str | None = None
    owasp_categories: list[OWASPEntry] = []
    response: str | None = None
    llm_meta: LLMMeta | None = None
    error: str | None = None


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest):
    start = time.time()

    # ── Scoring ───────────────────────────────────────────────────────────────
    regex_result   = score_prompt(body.prompt)
    semantic_result = _semantic.score(body.prompt)

    regex_score    = regex_result["score"]
    semantic_score = semantic_result["score"]
    closest_phrase = semantic_result.get("closest_phrase")

    injection_score = min(
        regex_score * settings.regex_weight
        + semantic_score * settings.semantic_weight,
        1.0,
    )

    matched_patterns: list[str] = regex_result.get("matched_patterns", [])

    if injection_score < 0.20:
        risk_level = "low"
    elif injection_score < 0.40:
        risk_level = "medium"
    else:
        risk_level = "high"

    blocked = injection_score >= settings.block_threshold

    # ── OWASP mapping ─────────────────────────────────────────────────────────
    owasp = map_categories(matched_patterns, injection_score, semantic_score)

    # ── Logging ───────────────────────────────────────────────────────────────
    latency_ms = (time.time() - start) * 1000

    log_payload = dict(
        prompt_length   = len(body.prompt),
        injection_score = injection_score,
        regex_score     = regex_score,
        semantic_score  = semantic_score,
        closest_phrase  = closest_phrase,
        risk_level      = risk_level,
        matched_patterns= matched_patterns,
        blocked         = blocked,
        status          = "blocked" if blocked else "ok",
        latency_ms      = latency_ms,
        model           = settings.llm_model,
        input_tokens    = None,
        output_tokens   = None,
        error           = None,
    )

    if blocked:
        logger.warning(
            "BLOCKED prompt | score=%.3f risk=%s owasp=%s",
            injection_score,
            risk_level,
            [c["code"] for c in owasp],
        )
        log_request(log_payload)
        write_request(log_payload)

        return JSONResponse(
            status_code=403,
            content=AnalyzeResponse(
                status          = "blocked",
                prompt          = body.prompt,
                injection_score = injection_score,
                regex_score     = regex_score,
                semantic_score  = semantic_score,
                risk_level      = risk_level,
                blocked         = True,
                matched_patterns= matched_patterns,
                closest_phrase  = closest_phrase,
                owasp_categories= [OWASPEntry(**c) for c in owasp],
                error           = "Prompt blocked: injection risk detected.",
            ).model_dump(),
        )

    # ── LLM call ──────────────────────────────────────────────────────────────
    try:
        result = call_llm(body.prompt, system_prompt=body.system_prompt)
    except LLMError as e:
        log_payload.update(error=e.message)
        log_request(log_payload)
        write_request(log_payload)
        return JSONResponse(
            status_code=e.status_code,
            content=AnalyzeResponse(
                status          = "error",
                prompt          = body.prompt,
                injection_score = injection_score,
                regex_score     = regex_score,
                semantic_score  = semantic_score,
                risk_level      = risk_level,
                blocked         = False,
                matched_patterns= matched_patterns,
                closest_phrase  = closest_phrase,
                owasp_categories= [OWASPEntry(**c) for c in owasp],
                error           = e.message,
            ).model_dump(),
        )

    latency_ms = (time.time() - start) * 1000
    log_payload.update(
        status       = "ok",
        latency_ms   = latency_ms,
        input_tokens = result["input_tokens"],
        output_tokens= result["output_tokens"],
    )
    log_request(log_payload)
    write_request(log_payload)

    logger.info(
        "ALLOWED prompt | score=%.3f risk=%s latency=%.0fms",
        injection_score, risk_level, latency_ms,
    )

    return AnalyzeResponse(
        status          = "ok",
        prompt          = body.prompt,
        injection_score = injection_score,
        regex_score     = regex_score,
        semantic_score  = semantic_score,
        risk_level      = risk_level,
        blocked         = False,
        matched_patterns= matched_patterns,
        closest_phrase  = closest_phrase,
        owasp_categories= [OWASPEntry(**c) for c in owasp],
        response        = result["response"],
        llm_meta        = LLMMeta(
            model        = result["model"],
            input_tokens = result["input_tokens"],
            output_tokens= result["output_tokens"],
            latency_ms   = latency_ms,
        ),
    )


# ── Stats endpoint ────────────────────────────────────────────────────────────

@router.get("/stats")
async def stats():
    import sqlite3
    from pathlib import Path

    db_path = Path("logs/firewall.db")
    if not db_path.exists():
        return {"error": "No database found. Send some requests first."}

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM requests").fetchall()

    if not rows:
        return {"total_requests": 0, "blocked_requests": 0,
                "block_rate_pct": 0.0, "avg_injection_score": 0.0,
                "by_risk_level": {"low": 0, "medium": 0, "high": 0}}

    total    = len(rows)
    blocked  = sum(1 for r in rows if r["blocked"])
    avg      = sum(r["injection_score"] for r in rows) / total
    by_risk  = {"low": 0, "medium": 0, "high": 0}
    for r in rows:
        lvl = r["risk_level"]
        if lvl in by_risk:
            by_risk[lvl] += 1

    return {
        "total_requests":     total,
        "blocked_requests":   blocked,
        "block_rate_pct":     round(blocked / total * 100, 2),
        "avg_injection_score": round(avg, 4),
        "by_risk_level":      by_risk,
    }
