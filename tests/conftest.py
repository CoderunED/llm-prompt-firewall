# tests/conftest.py
import pytest
from app.scorer import score_prompt

@pytest.fixture(scope="session")
def scorer():
    """
    Session-scoped fixture — the semantic model loads once
    for the entire test run, not once per test.
    """
    return score_prompt
