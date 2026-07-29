# Agentic Math Prof

An agentic math-tutoring system that solves problems with retrieval-augmented generation (RAG), a team of cooperating agents (Professor, Critic, Guardrail, HITL), and optional web search augmentation via an MCP tool server. A React + Vite frontend talks to a FastAPI backend orchestrating retrieval, reasoning, and validation.

## Features
- Retrieval-Augmented Generation (RAG) over curated math knowledge bases (GSM8K, Orca 200k sample)
- Multi-agent pipeline:
  - Guardrail: LLM-based filter (llama-3.1-8b-instant); returns structured `{"status": "allowed"|"blocked"|"error"}`; blocks non-math queries and prompt injection
  - Professor: drafts step-by-step solutions with RAG, optional MCP web search tools, and SymPy symbolic solver
  - Critic: strict LLM evaluator (llama-3.1-8b-instant); outputs JSON `{"Decision": "Accept"|"Refine", "Feedback": "..."}` with regex fallback; defaults to "Refine" on error
  - HITL: asyncio.Event-based pause/resume for human feedback; stored in JSONL + Qdrant vector store
- DSPy-based optimization: `MathFeedbackRefiner` module compiled via `BootstrapFewShot` teleprompter in a background async task
- Feedback persistence in JSONL (`Data/feedback/refiner_train.jsonl`) for durability and Qdrant for vector-similarity retrieval of similar past feedback during refinement
- SymPy math solver tool for symbolic equation solving
- Per-call `SummarizedMemory` isolation so conversation turns don't leak across tasks
- Tool usage tracking (`tool_usage` list returned in `/status` responses)
- Vector stores: local, file-backed Qdrant collections (no external DB server required) and support for Chroma paths
- Optional web search augmentation via an MCP server (tools: search/crawl/extract/scrape)
- React + Vite frontend, FastAPI backend, structured YAML logging

## Repository structure
```
backend/
  app/
    api.py                 # FastAPI app (endpoints /solve, /status/{id}, /feedback)
    orchestration.py       # LangGraph-based orchestration of agents & tools
    state.py               # Task state, events for HITL
    agents/                # BaseAgent + Professor, Critic, Guardrail, HITL modules
    config/                # logging_config.yaml + settings.py (env-driven)
    core/
      logger.py            # Logger bootstrapper
      feedback_qdrant.py   # Qdrant-backed vector store for feedback records
      registry.py          # Shared ProfessorAgent singleton & lifecycle
    Memory/custom_memory.py
    tools/
      RetrieverTool.py     # Qdrant-based retriever tool
      MathSolverTool.py    # SymPy expression solver
  requirements.txt         # Python dependency manifest
  tests/
    __init__.py
    criticAgent_tests.py
    guardrailAgent_tests.py
Data/
  knowledge_base/         # Local vector DBs (Qdrant/Chroma) and datasets
  feedback/               # Human-feedback examples (JSONL) and Qdrant store
docs/                      # Architecture, evaluation design docs
frontend/                  # React + Vite app (dev on :5173)
mcp_servers/
  websearch/               # MCP tool server (Python, stdio); used by ProfessorAgent
results/                   # Evaluation run outputs (CSV, JSONL, summaries)
Scripts/
  gsm8k_kb.py, orca200k.py # KB builders (ingest/embed/index)
  Eval.py, evaluate_responses.py  # Evaluation & scoring pipeline
  parquet_to_csv_umath.py  # Dataset format converter
```

## Tech stack
- Backend: Python, FastAPI, LangGraph, Pydantic Settings, SymPy
- Retrieval: Qdrant (local, file-backed via `qdrant-client`), LangChain (HuggingFace embeddings)
- Agents: Modular Python classes with shared memory and DSPy-based refinement hooks
- Web augmentation: MCP server (Python) with Firecrawl/Tavily/OpenAI clients
- Frontend: React + Vite + Axios
- Logging/Config: YAML logging, env-driven `settings.py`

## Prerequisites
- Python 3.11+
- Node.js LTS (for the frontend)
- Optional: API keys (place in root `.env`)
  - GROQ_API_KEY (required for LLM usage)
  - FIRECRAWL_API_KEY (web search MCP)
  - TAVILY_API_KEY (optional future use)

Example root `.env` (repo root):
```
GROQ_API_KEY=your_groq_key_here
FIRECRAWL_API_KEY=your_firecrawl_key_here
TAVILY_API_KEY=your_tavily_key_here
```

## Backend setup and run (FastAPI)
From the repository root:

```powershell
# 1) Create and activate a virtualenv (Windows PowerShell)
python -m venv venv
./venv/Scripts/Activate.ps1

# 2) Install backend dependencies
pip install fastapi uvicorn[standard] pydantic pydantic-settings python-dotenv \
            langgraph sympy langchain langchain-groq langchain-qdrant \
            langchain-huggingface qdrant-client sentence-transformers \
            langchain-mcp-adapters dspy-ai pyyaml

# 3) Launch the API (CORS allows http://localhost:5173)
uvicorn backend.app.api:app --reload --port 8000
```

API endpoints:
- `POST /solve` body: `{ "query": "<your math problem>" }` → returns `task_id` and status
- `GET /status/{task_id}` → returns status and final answer/tools/iterations when complete
- `POST /feedback` body: `{ "task_id": "...", "status": "needs_feedback", "feedback": "..." }` (for HITL refinement)

## Frontend setup and run (React + Vite)
```powershell
cd frontend
npm install
# Optional: start backend from here (runs uvicorn from repo root)
npm run start:backend
# Start the dev server
npm run dev
```
Open http://localhost:5173 and submit a math problem.

## MCP websearch server (optional)
The ProfessorAgent will attempt to initialize a local MCP server and load tools automatically when API keys are present. You typically do not need to run this manually. For local testing:

```powershell
cd mcp_servers/websearch
# Create .env from example and fill FIRECRAWL_API_KEY (and GROQ_API_KEY if required by your tools)
copy .env.example .env

# (Recommended) Use a dedicated virtualenv
python -m venv .venv
.\.venv/Scripts/Activate.ps1

# Install dependencies defined in pyproject
pip install -e .

# Run server manually (normally launched by the ProfessorAgent via stdio)
python main.py
```

> **Provider interchangeability:** The MCP server lists `OPENAI_API_KEY` in its `.env.example`, but the main pipeline uses Groq (OpenAI-compatible endpoint). Because both providers share the same chat-completion API format, swapping between them requires only changing the base URL and key — no code changes. The same principle applies to the search/extract backends (Firecrawl, Tavily) used by the server.

## Data and knowledge bases
- Prebuilt Qdrant collections live under `./Data/knowledge_base/qdrant_db*`. The retriever uses a local, file-backed Qdrant client (`QdrantClient(path=...)`), so no external DB server is required.
- If you need to (re)build a KB from raw datasets, see `./Scripts/gsm8k_kb.py` and `./Scripts/orca200k.py`.
- Large datasets and DB artifacts are ignored via `.gitignore` to keep the repo lean.

## Tests
Run agent tests from the repo root:
```powershell
# Critic agent tests
python -m backend.tests.criticAgent_tests

# Guardrail agent tests
python -m backend.tests.guardrailAgent_tests
```

## How it works (architecture)
1. Frontend posts a problem to `POST /solve`. A `task_id` is created; state is tracked in `state.py` (`tasks` dict + `asyncio.Event` per task).
2. Orchestrator (`orchestration.py`, LangGraph `StateGraph`) runs three nodes with conditional routing:
   - **Guardrail** — LLM classifies the query as pass/fail; blocked queries terminate immediately.
   - **Professor** — retrieves top-3 similar examples from two Qdrant KBs (GSM8K, Orca 200k); optionally runs MCP web search tools (search → extract → crawl → analyze_content); uses SymPy `math_solver`; generates a step-by-step solution. Each run uses an isolated `SummarizedMemory`.
   - **Critic/HITL** — LLM evaluates the solution against strict rules; outputs JSON `{"Decision": "Accept"|"Refine", "Feedback": "..."}`. On "Refine":
     1. Task status set to `needs_feedback`; frontend polls `/status` and shows the feedback form.
     2. Orchestrator pauses on `await hitl_events[task_id].wait()` (5-minute timeout).
     3. Human submits feedback via `POST /feedback`, which sets the event and resumes the workflow.
     4. Feedback saved to JSONL (`Data/feedback/refiner_train.jsonl`) and Qdrant (`feedback_qdrant.py`).
     5. A background task triggers DSPy `BootstrapFewShot` compilation of `MathFeedbackRefiner` when enough examples accumulate (min 5).
     6. Professor refines the solution using the `MathFeedbackRefiner` DSPy module, with context augmented by top-3 similar past feedback from Qdrant vector search.
   - The refine loop runs **up to 2 refinements** (3 professor runs max; controlled by `iterations <= 2` in `route_critic`).
3. All tool invocations (KB retrievers, web search tools, math solver, LLM) are tracked in `tool_usage` and returned with the final answer via `GET /status/{task_id}`.
4. Logging is centralized via `logging_config.yaml` and `core/logger.py`.

## Troubleshooting
- Missing API keys: set `GROQ_API_KEY` (required) and `FIRECRAWL_API_KEY` in root `.env`.
- Embeddings download: the first run may download HuggingFace models (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
- File locks on Qdrant (Windows): the app uses a singleton `QdrantClientManager` and closes clients on shutdown; avoid opening the same DB from multiple processes.
- CORS: frontend runs on `http://localhost:5173`; backend allows that origin by default.
- Ports: backend `:8000`, frontend `:5173`.

## License
This project inherits its license from the repository (see the LICENSE file on the remote).

---
