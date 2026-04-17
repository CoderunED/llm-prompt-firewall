import json
import logging
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "requests.jsonl"

logger = logging.getLogger(__name__)


class RequestLogger:
    def __init__(self):
        _LOG_DIR.mkdir(exist_ok=True)

    def log(
        self,
        *,
        prompt_length: int,
        injection_score: float,
        risk_level: str,
        matched_patterns: list[str],
        regex_score: float,
        semantic_score: float,
        closest_phrase: str,
        blocked: bool,
        status: str,
        latency_ms: float | None = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error: str | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_length": prompt_length,
            "injection_score": round(injection_score, 4),
            "regex_score": round(regex_score, 4),
            "semantic_score": round(semantic_score, 4),
            "closest_phrase": closest_phrase,
            "risk_level": risk_level,
            "matched_patterns": matched_patterns,
            "blocked": blocked,
            "status": status,
            "latency_ms": latency_ms,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": error,
        }
        try:
            with _LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.error("Failed to write request log: %s", e)


request_logger = RequestLogger()
