# app/llm_client.py
import time
import logging
from anthropic import Anthropic, APITimeoutError, APIStatusError, APIConnectionError
from app.config import settings

logger = logging.getLogger(__name__)

client = Anthropic(api_key=settings.anthropic_api_key)


class LLMError(Exception):
    """Base class for LLM client errors."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class LLMTimeoutError(LLMError):
    def __init__(self):
        super().__init__("LLM request timed out. Try again.", status_code=504)


class LLMAuthError(LLMError):
    def __init__(self):
        super().__init__("Invalid API key.", status_code=401)


class LLMRateLimitError(LLMError):
    def __init__(self):
        super().__init__("Rate limit reached. Slow down.", status_code=429)


class LLMUnavailableError(LLMError):
    def __init__(self, detail: str = ""):
        super().__init__(f"LLM service unavailable. {detail}".strip(), status_code=503)


def call_llm(prompt: str, system_prompt: str | None = None) -> dict:
    """
    Send a prompt to Claude and return a structured result dict.

    Returns:
        {
            "response": str,
            "model": str,
            "input_tokens": int,
            "output_tokens": int,
            "latency_ms": float,
        }

    Raises:
        LLMError subclass on any failure.
    """
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "model": settings.llm_model,
        "max_tokens": 1024,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    start = time.perf_counter()
    try:
        response = client.messages.create(**kwargs)
    except APITimeoutError:
        logger.error("Anthropic API timeout")
        raise LLMTimeoutError()
    except APIStatusError as e:
        logger.error("Anthropic API status error: %s %s", e.status_code, e.message)
        if e.status_code == 401:
            raise LLMAuthError()
        if e.status_code == 429:
            raise LLMRateLimitError()
        raise LLMUnavailableError(f"Status {e.status_code}")
    except APIConnectionError as e:
        logger.error("Anthropic connection error: %s", e)
        raise LLMUnavailableError("Connection failed.")
    except Exception as e:
        logger.exception("Unexpected LLM error")
        raise LLMUnavailableError(str(e))

    latency_ms = (time.perf_counter() - start) * 1000
    text = response.content[0].text if response.content else ""

    return {
        "response": text,
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_ms": round(latency_ms, 2),
    }
