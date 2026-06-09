# tests/conftest.py
import os
import pytest
from app.scorer import score_prompt

# Ensure consistent scoring config in all environments
os.environ.setdefault("BLOCK_THRESHOLD", "0.40")
os.environ.setdefault("REGEX_WEIGHT", "0.50")
os.environ.setdefault("SEMANTIC_WEIGHT", "0.50")

@pytest.fixture(scope="session")
def scorer():
    """
    Session-scoped fixture — the semantic model loads once
    for the entire test run, not once per test.
    """
    return score_prompt
