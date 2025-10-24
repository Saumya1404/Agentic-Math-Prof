# Agentic Math Prof

An agentic math-tutoring system that solves problems with retrieval-augmented generation (RAG), a team of cooperating agents (Professor, Critic, Guardrail, HITL), and optional web search augmentation via an MCP tool server. A React + Vite frontend talks to a FastAPI backend orchestrating retrieval, reasoning, and validation.

## Features
- Retrieval-Augmented Generation (RAG) over curated math knowledge bases (GSM8K, Orca 200k sample)
- Multi-agent pipeline:
  - Guardrail: filters non-math/injection attempts
  - Professor: drafts step-by-step solutions
  - Critic: strictly evaluates correctness and completeness
  - HITL: optional human feedback loop for refinement
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
    core/logger.py         # Logger bootstrapper
    Memory/custom_memory.py
    tools/RetrieverTool.py # Qdrant-based retriever tool
  tests/
    criticAgent_tests.py
    guardrailAgent_tests.py
Data/
  knowledge_base/         # Local vector DBs (Qdrant/Chroma) and datasets
frontend/                  # React + Vite app (dev on :5173)
mcp_servers/
  websearch/               # MCP tool server (Python, stdio); used by ProfessorAgent
Scripts/
  gsm8k_kb.py, orca200k.py # KB builders (ingest/embed/index)
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
            langgraph sympy langchain-qdrant langchain-huggingface qdrant-client \
            sentence-transformers langchain-mcp-adapters dspy-ai

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
./.venv/Scripts/Activate.ps1

# Install dependencies defined in pyproject
pip install -e .

# Run server manually (normally launched by the ProfessorAgent via stdio)
python main.py
```

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
1. Frontend posts a problem to the backend (`/solve`).
2. Orchestrator (`orchestration.py`, LangGraph) runs nodes:
   - Guardrail → filter non-math and injection attempts
   - Professor → retrieve context (Qdrant) and draft solution; optionally uses MCP web search tools
   - Critic → evaluate rigorously; on “Refine”, request HITL feedback and loop up to 3 iterations
3. State is stored in `state.py` (tasks map + asyncio events). Final answer is returned via `/status/{task_id}`.
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
If you want me to tailor the README with exact dependency pins or add badges/diagrams, I can update this file and wire up a simple CI workflow next.
