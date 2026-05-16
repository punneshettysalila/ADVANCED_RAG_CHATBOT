# A2A Multi-Agent Chatbot (Repository Root)

This repository contains the A2A Multi-Agent Chatbot. The application code lives in the `a2a_multiagent/` directory. This root README gives quick steps to deploy to Render and to run locally.

Quick Start (local):

```bash
# From repository root
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
# Run server from repo root (uses PYTHONPATH to point to inner folder)
PYTHONPATH=a2a_multiagent python -m server
# or with uvicorn for dev
PYTHONPATH=a2a_multiagent uvicorn server:app --reload
```

Deploy to Render (GitHub):

1. Initialize git and commit/push the repo to GitHub from the repository root.

```bash
git init
git add .
git commit -m "Initial commit — Render-ready"
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. In Render, create a new **Web Service**, connect your GitHub repo and Render will use `render.yaml` to build and run.
3. Set the `HF_TOKEN` secret under the Render service's environment variables.

Notes:
- `render.yaml` is at the repository root and sets `PYTHONPATH=a2a_multiagent` so the inner folder is importable.
- Uploaded documents and the SQLite DB are stored in `a2a_multiagent/.a2a_memory` — add this to `.gitignore`.

For full developer docs, see `a2a_multiagent/README.md`.
