# A2A Multi-Agent Chatbot

Auto-routing multi-agent chatbot following the **Agent-to-Agent (A2A) protocol**,
built with **Python · FastAPI · HuggingFace · HTML/CSS/JS**.

---

## Agents

| Agent | Role | Model |
|-------|------|-------|
| **Auto-Routing Orchestrator** | Routes each query to the best agent path | Mistral-7B-Instruct |
| **Research** | Gathers deep, well-structured information | Mistral-7B-Instruct |
| **Summarizer** | Distils research into TL;DR + Key Points | Mistral-7B-Instruct |
| **General QA** | Handles conversational / short questions | Mistral-7B-Instruct |

## Pipeline (Auto-Routing)

```
User Query
    │
    │
    ├── research_only
    ├── summarize_only
    ├── general_qa
    └── research_and_summarize
          │
   Streamed Agent Response
```

**Key Features:**
- ✅ **Auto-routing** to the best agent path
- ✅ **Streaming responses** from the selected route
- ✅ **Voice chat input** (browser speech-to-text) with auto-send
- ✅ **A2A Protocol compliant** - proper message format, task lifecycle, agent discovery
- ✅ Each agent uses HuggingFace token independently
- ✅ Clean static client + FastAPI server (client-server architecture)
- ✅ **3 selectable themes** with dynamic icons (Harrpotter, NFC Playback Game, Corporate)

**Routing behavior**
The orchestrator auto-selects the most suitable path based on the user query:
- **research_only**
- **summarize_only**
- **general_qa**
- **research_and_summarize**

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set your HuggingFace token
cp .env.example .env
# edit .env → HF_TOKEN=hf_...
# optional: choose model (default is Qwen/Qwen2.5-7B-Instruct)
# HF_MODEL=Qwen/Qwen2.5-7B-Instruct
# optional: add fallback models tried automatically when a model/provider task is unsupported
# HF_MODEL_FALLBACKS=HuggingFaceH4/zephyr-7b-beta,google/gemma-2-2b-it

# 3. Start the FastAPI backend
python server.py          # http://localhost:8000

# 4. Open the web client
# http://localhost:8000
```

Get a free HuggingFace token at: https://huggingface.co/settings/tokens

Note: provider auto-routing is disabled. The app always uses `provider="hf-inference"` with your `hf_` token.

---

## Themes

The web client includes a built-in theme selector in the sidebar:
- **Dark**
- **Pastel**
- **Light + Dark Mix**

When a theme changes, the background palette and theme-specific icons update across logo, assistant avatar, empty state, and agent badges.

---

## Project Structure

```
a2a_multiagent/
├── a2a/
│   └── protocol.py              # A2A types: Task, Message, Parts, Registry
├── agents/
│   ├── base.py                  # BaseAgent + HFInferenceWrapper
│   ├── specialized.py           # Research, Summarizer, GeneralQA
│   ├── orchestrator.py          # Deprecated LangGraph orchestrator
│   └── orchestrator_parallel.py # Auto-routing orchestrator ← ACTIVE
├── static/
│   └── index.html               # Main web client (pastel UI + cursor motion)
├── server.py                    # FastAPI gateway + static client hosting
├── streamlit_app.py             # Optional Streamlit client
├── requirements.txt
├── render.yaml                  # Render deployment config
└── .env.example
```

---

## Deploy on Render

1. Push this project to GitHub.
2. In Render, create a new **Web Service** and connect the repo.
3. Render will detect `render.yaml` automatically.
4. Set `HF_TOKEN` in Render environment variables.
5. Keep `PYTHON_VERSION=3.11.11` (already set in `render.yaml`).
6. Deploy and open the service URL.

Configured production commands:
- Build: `pip install --upgrade pip && pip install -r requirements.txt`
- Start: `gunicorn -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

Default environment values in `render.yaml`:
- `HF_MODEL=Qwen/Qwen2.5-7B-Instruct`
- `LOG_LEVEL=info`

Required secret:
- `HF_TOKEN` (set this manually in Render)

Optional customization:
- Set `HF_MODEL` in Render to override the default model.
- Set `HF_MODEL_FALLBACKS` for comma-separated fallback models.

The same FastAPI app serves both API and web client, so no separate frontend service is required.

### Voice input support

The static web client includes a 🎤 button next to the text input. Click it, speak, and your speech is converted to text and sent automatically.

Browser note: this relies on the Web Speech API (`SpeechRecognition` / `webkitSpeechRecognition`) and works best in Chromium-based browsers.

### Render runtime/dependencies

- `requirements.txt` includes:
  - `uvicorn[standard]` for ASGI serving
  - `gunicorn` for production process management on Render
- `render.yaml` currently starts with:
  - `gunicorn -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

If you want to lower memory usage on free tier, reduce workers to 1:
`gunicorn -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`

## API

`POST /tasks/send` — SSE stream  
`GET  /agents`     — A2A agent discovery  
`GET  /tasks/{id}` — Task state  
`GET  /health`     — Health check  
