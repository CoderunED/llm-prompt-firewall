# app/main.py
import logging
import logging.config
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routes.analyze import router as analyze_router

# ── Logging setup ────────────────────────────────────────────────────────────
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
})

logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LLM Prompt Firewall",
    description="Middleware that intercepts and scores prompts before they reach an LLM.",
    version="0.3.0",
)

app.include_router(analyze_router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": "Internal server error."},
    )


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "version": app.version}
