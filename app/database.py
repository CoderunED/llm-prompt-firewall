import sqlite3
import logging
from pathlib import Path

_DB_DIR = Path(__file__).resolve().parent.parent / "logs"
_DB_PATH = _DB_DIR / "firewall.db"

logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    _DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the requests table if it doesn't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                prompt_length   INTEGER NOT NULL,
                injection_score REAL    NOT NULL,
                regex_score     REAL    NOT NULL,
                semantic_score  REAL    NOT NULL,
                closest_phrase  TEXT,
                risk_level      TEXT    NOT NULL,
                matched_patterns TEXT,
                blocked         INTEGER NOT NULL,
                status          TEXT    NOT NULL,
                latency_ms      REAL,
                model           TEXT,
                input_tokens    INTEGER,
                output_tokens   INTEGER,
                error           TEXT
            )
        """)
        conn.commit()
    logger.info("SQLite DB ready at %s", _DB_PATH)


def insert_request(entry: dict) -> None:
    """Insert one log entry into the requests table."""
    import json
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO requests (
                timestamp, prompt_length, injection_score, regex_score,
                semantic_score, closest_phrase, risk_level, matched_patterns,
                blocked, status, latency_ms, model, input_tokens,
                output_tokens, error
            ) VALUES (
                :timestamp, :prompt_length, :injection_score, :regex_score,
                :semantic_score, :closest_phrase, :risk_level, :matched_patterns,
                :blocked, :status, :latency_ms, :model, :input_tokens,
                :output_tokens, :error
            )
        """, {**entry, "matched_patterns": json.dumps(entry["matched_patterns"])})
        conn.commit()
