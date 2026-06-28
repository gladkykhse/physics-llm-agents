# Physics LLM Agents

Master's thesis: **Agentic Orchestration and Tool Augmentation for Physics Problem Solving with Small Language Models**

LangGraph-based agents that solve multiple-choice physics problems by combining LLM reasoning with symbolic math tools (SymPy) and retrieval over a physics knowledge base (pgvector RAG + Wikipedia). Agents and plain models are evaluated on the **MMLU** (physics subsets) and **SciEval** (physics) benchmarks.

## Project layout

```
config/                 YAML configs (benchmark, data, vector_rag, and one per agent)
src/
  agents/               LangGraph agents — each defines a class with a .solve(problem) method
    utils/              shared agent code: llm, tools (sympy + retrieval), plan helpers
  benchmarks/           MMLU + SciEval dataset loading, parsing, scoring, plotting
  models/               model backends: openai_api, ollama, vllm, langgraph (agent runner)
  knowledge_bases/      pgvector RAG: chunking, embedding, insert/search
  data/                 OpenStax textbook data preparation
  utils/                small helpers (yaml/json/file IO)
sql/postgres/           schema for the pgvector RAG table
docker-compose.yaml     local Postgres + pgvector for the RAG knowledge base
run_*.py                entry points (see below)
```

Benchmark datasets (`benchmarks/`), prepared data (`data/`), and run outputs (`artifacts/`) are git-ignored — they are generated locally.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. install dependencies (creates .venv from uv.lock)
uv sync

# 2. configure environment
cp .env.example .env        # then fill in OPENAI_API_KEY, RETRIEVER_API_TOKEN, etc.

# 3. (optional) start the pgvector knowledge base for retrieval agents
docker compose up -d
```

All commands below run inside the uv environment via `uv run`.

## Usage

**Run a benchmark + evaluate** (`run_all.py`). `-m` accepts either a model name
(listed in `config/benchmark.yaml`) or an agent module name from `src/agents/`:

```bash
# evaluate a plain model
uv run run_all.py -m gpt-4o-mini -b scieval -p cot

# evaluate an agent (any module name in src/agents/)
uv run run_all.py -m final_agent_c -b mmlu -s college_physics
```

Outputs (`.parquet`, `.json`, `.png`, `.log`) are written under `artifacts/`.

**Build the RAG knowledge base** (requires Postgres running and data prepared):

```bash
uv run run_dataprep.py                          # prepare OpenStax text (config/data.yaml)
uv run run_vector_rag_insert.py -m recursive    # chunk, embed, insert (config/vector_rag.yaml)
```

**Standalone GPU benchmarks** for HuggingFace models that need bespoke loading
(ChatGLM3 / SciDFM-MoE). These are self-contained and pin their own dependencies:

```bash
uv run --python 3.11 run_hf_benchmark.py -m OpenDFM/SciDFM-MoE-A5.6B-v1.0 -b scieval
uv run --python 3.11 run_scidfm_benchmark.py -b scieval
```

## Agents

Each file in `src/agents/` is an agent variant; the runner dynamically loads the
class exposing a `.solve(problem)` method. `final_agent_a`, `final_agent_b`, and
`final_agent_c` are the final pipelines; the other modules are the ablation
variants (planning / retrieval / react / thinking) developed along the way. Each
agent reads its hyperparameters and prompts from the matching `config/<name>.yaml`.
