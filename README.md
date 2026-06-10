# LLM Prompt Firewall

A FastAPI middleware that intercepts prompts before they reach an LLM, scores them for injection attempts using a blended regex + semantic pipeline, blocks high-risk requests, and logs everything to SQLite. Built as a portfolio project targeting Cloud Security, AI/ML Security, and DevSecOps engineering roles.

---

## Architecture

<img width="1170" height="1120" alt="Architecture" src="https://github.com/user-attachments/assets/30bbcbc5-7a04-4d84-ac80-d4cd70d8a1d6" />

---
## Dashboard and Results
<img width="1432" height="1000" alt="Dashboard1" src="https://github.com/user-attachments/assets/74553e38-8ab7-4745-9a68-fd6a0cd5ebef" />
<img width="1459" height="837" alt="Results" src="https://github.com/user-attachments/assets/2dd2588f-b5b0-468c-ab63-1185716b118c" />

---

## Detection Pipeline

Every prompt passes through two independent scoring layers. The scores are blended 50/50 into a final `injection_score`.

| Layer | Implementation | Signal |
|---|---|---|
| Regex | `scorer.py` — 25+ patterns | Explicit attack syntax |
| Semantic | `semantic_scorer.py` — `all-MiniLM-L6-v2`, 37 attack anchors, cosine similarity | Paraphrased and obfuscated attacks |
| Blended | `(regex × 0.50) + (semantic × 0.50)` | Final decision score |

Prompts scoring `≥ 0.40` are blocked with a `403` and never reach the LLM. The threshold is tunable via `.env`.

### OWASP LLM Top 10 Mapping

Every blocked request is mapped to the relevant [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) categories based on matched patterns and score signals:

| Category | Triggers on |
|---|---|
| LLM01 — Prompt Injection | Instruction override patterns, high blended score |
| LLM02 — Sensitive Information Disclosure | Credential/secret extraction patterns |
| LLM05 — Improper Output Handling | Code/command injection patterns |
| LLM06 — Excessive Agency | Role hijacking, jailbreak patterns |
| LLM07 — System Prompt Leakage | System prompt extraction patterns |
| LLM10 — Unbounded Consumption | Token flood / resource exhaustion patterns |

---

## API Reference

### `POST /api/v1/analyze`

Send a prompt through the firewall.

**Request**
```json
{
  "prompt": "What is the capital of France?",
  "system_prompt": "You are a helpful assistant."
}
```

**Response — allowed (200)**
```json
{
  "status": "ok",
  "injection_score": 0.021,
  "regex_score": 0.0,
  "semantic_score": 0.042,
  "risk_level": "low",
  "blocked": false,
  "matched_patterns": [],
  "owasp_categories": [],
  "response": "The capital of France is Paris.",
  "llm_meta": {
    "model": "claude-opus-4-6",
    "input_tokens": 24,
    "output_tokens": 9,
    "latency_ms": 843
  }
}
```

**Response — blocked (403)**
```json
{
  "status": "blocked",
  "injection_score": 0.847,
  "regex_score": 0.9,
  "semantic_score": 0.794,
  "risk_level": "high",
  "blocked": true,
  "matched_patterns": ["ignore_previous", "reveal_system"],
  "owasp_categories": [
    {"code": "LLM01", "name": "Prompt Injection", "description": "..."},
    {"code": "LLM07", "name": "System Prompt Leakage", "description": "..."}
  ],
  "error": "Prompt blocked: injection risk detected."
}
```

### `GET /api/v1/stats`

Returns aggregate metrics from the SQLite log.

```json
{
  "total_requests": 142,
  "blocked_requests": 38,
  "block_rate_pct": 26.76,
  "avg_injection_score": 0.2814,
  "by_risk_level": {"low": 89, "medium": 15, "high": 38}
}
```

### `GET /health`

```json
{"status": "ok", "version": "1.0.0"}
```

---

## Quick Start

### Local

```bash
git clone https://github.com/CoderunED/llm-prompt-firewall
cd llm-prompt-firewall

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

python -m uvicorn app.main:app --reload
# API: http://localhost:8000
# Docs: http://localhost:8000/docs

# Dashboard (separate terminal)
streamlit run dashboard.py
# http://localhost:8501
```

### Docker

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

docker compose up --build
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## Project Structure

```
llm-prompt-firewall/
├── app/
│   ├── main.py              # FastAPI app, startup, routing
│   ├── config.py            # pydantic-settings config
│   ├── scorer.py            # Regex scoring engine (25+ patterns)
│   ├── semantic_scorer.py   # Semantic scoring (all-MiniLM-L6-v2)
│   ├── owasp.py             # OWASP LLM Top 10 category mapper
│   ├── llm_client.py        # Anthropic SDK wrapper
│   ├── logger.py            # JSONL request logger
│   ├── database.py          # SQLite dual-write logger
│   └── routes/
│       └── analyze.py       # POST /analyze, GET /stats
├── tests/
│   ├── conftest.py
│   └── test_scoring.py      # 30 tests — clean, direct, paraphrased attacks
├── dashboard.py             # Streamlit monitoring dashboard
├── calibrate.py             # Calibration script
├── Dockerfile               # Multi-stage build
├── docker-compose.yml       # API + dashboard services, shared volume
├── .github/workflows/
│   └── ci.yml               # GitHub Actions — pytest on push/PR
└── .env.example
```

---

## Configuration

All tunable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required |
| `LLM_MODEL` | `claude-opus-4-6` | Anthropic model |
| `BLOCK_THRESHOLD` | `0.40` | Score at which prompts are blocked |
| `REGEX_WEIGHT` | `0.50` | Regex layer contribution to blended score |
| `SEMANTIC_WEIGHT` | `0.50` | Semantic layer contribution to blended score |
| `APP_ENV` | `development` | `development` or `production` |

---

## Test Suite

```bash
pytest tests/ -v
```

30 tests across three classes:

| Class | Count | Assertion |
|---|---|---|
| `TestCleanPrompts` | 10 | None blocked |
| `TestDirectAttacks` | 10 | All blocked |
| `TestParaphrasedAttacks` | 10 | All blocked |

CI runs on every push and PR to `main` via GitHub Actions (`ubuntu-latest`, Python 3.12).

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn, Pydantic |
| Scoring | sentence-transformers (`all-MiniLM-L6-v2`), scikit-learn, regex |
| LLM | Anthropic SDK (`claude-opus-4-6`) |
| Storage | SQLite (stdlib), JSONL flat files |
| Dashboard | Streamlit, Plotly |
| Container | Docker, Docker Compose |
| CI | GitHub Actions |
| Language | Python 3.12 |

---

## Security Notes

- Container runs as non-root user (`firewall`)
- API keys loaded via environment variables only — never committed
- All requests logged for audit; blocked prompts include full pattern and OWASP classification
- Threshold, weights, and model are runtime-configurable without code changes
