# A2A Multi-Agent Chatbot

This repository contains a FastAPI-based multi-agent chatbot that follows the Agent-to-Agent (A2A) protocol. The top-level repo contains deployment files and documentation, while the application code lives inside `a2a_multiagent/`.

## Overview

The app provides:

- a FastAPI backend with task routing and A2A-style task lifecycle handling
- a static web client served by the backend
- a Streamlit client for alternate UI access
- knowledge upload and search helpers
- persisted task, memory, and grounding data under `.a2a_memory/`

## Repository Layout

```text
.
├── README.md
├── render.yaml
├── requirements.txt
├── runtime.txt
└── a2a_multiagent/
	├── A2A_COMPLIANCE.md
	├── experience_store.py
	├── knowledge_base.py
	├── server.py
	├── streamlit_app.py
	├── supervisor.py
	├── tools.py
	├── verification.py
	├── __init__.py
	├── a2a/
	│   └── protocol.py
	├── agents/
	│   ├── base.py
	│   ├── orchestrator.py
	│   ├── orchestrator_parallel.py
	│   ├── specialized.py
	│   └── supervised_orchestrator.py
	└── static/
		└── index.html
```

## Prerequisites

- Python 3.11+
- A Hugging Face token if you want the LLM-backed features to work
- Git for version control and deployment

## Local Setup

From the repository root:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Create a local environment file if needed:

```bash
copy .env.example .env
```

Set at least:

- `HF_TOKEN`
- optional `HF_MODEL`
- optional `LOG_LEVEL`

## Run Locally

FastAPI backend:

```bash
cd a2a_multiagent
python server.py
```

Development server with auto-reload:

```bash
cd a2a_multiagent
uvicorn server:app --reload
```

Streamlit UI:

```bash
streamlit run a2a_multiagent/streamlit_app.py
```

The backend serves the static web client at `/` and exposes the API endpoints below.

## Deploy to Render

The repository is configured for Render with `render.yaml`.

1. Push the repository to GitHub.
2. Create a new Render **Web Service** connected to that repo.
3. Render will use `render.yaml` automatically.
4. Set `HF_TOKEN` as a secret environment variable in Render.

Current Render config:

- build command: `pip install --upgrade pip && pip install -r requirements.txt`
- start command: `cd a2a_multiagent && gunicorn -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

Compatibility note:

- The root-level `app.py` file also exposes the same FastAPI app for platforms or dashboard settings that still use `app:app`.

## Environment Variables

Common variables used by the app:

- `HF_TOKEN` - Hugging Face token used for model access
- `HF_MODEL` - default model name, currently set to `Qwen/Qwen2.5-7B-Instruct` in Render config
- `LOG_LEVEL` - logging level, default `info`
- `PYTHON_VERSION` - pinned to `3.11.11` in Render config

## API Endpoints

The FastAPI server exposes:

- `GET /health` - health check
- `GET /agents` - agent discovery
- `POST /tasks/send` - streaming task submission
- `POST /tasks/send/sync` - synchronous task submission
- `GET /tasks/{task_id}` - task lookup
- `POST /knowledge/upload` - upload documents for grounding
- `GET /knowledge/search` - search uploaded knowledge
- `GET /memory/stats` - memory statistics

## Data Storage

Runtime memory, task history, and uploaded knowledge are stored in:

- `a2a_multiagent/.a2a_memory/`

This directory is ignored by Git and should stay out of source control.

## Notes

- The nested `a2a_multiagent/` directory is the real application package.
- `a2a_multiagent/README.md` contains additional project-level details for the app itself.
- `render.yaml` is the source of truth for deployment on Render.
