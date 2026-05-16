"""
FastAPI A2A Server

This server implements the Agent-to-Agent (A2A) protocol with auto-routing.
Each query is routed to the most relevant agent path.

Exposes:
  POST /tasks/send           — submit a task, get streaming SSE response
  POST /tasks/send/sync      — submit a task, get synchronous response  
  GET  /tasks/{id}           — get task state
  GET  /agents               — list registered agents (A2A discovery)
  GET  /health               — health check

A2A Compliance:
  - Agent discovery via AgentRegistry and /agents endpoint
  - Task lifecycle management (submitted → working → completed)
  - Message format with parts (text, code, data, error)
  - Auto-routing to the best agent path with explicit task lifecycle
"""

from __future__ import annotations

import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from a2a.protocol import A2AMessage, A2ATask, AgentRegistry, TextPart
from agents.supervised_orchestrator import ParallelOrchestratorAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# App state
# ─────────────────────────────────────────────

registry = AgentRegistry()
orchestrator: Optional[ParallelOrchestratorAgent] = None
tasks: Dict[str, A2ATask] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        logger.warning("HF_TOKEN not set — LLM calls will fail without it")
    orchestrator = ParallelOrchestratorAgent(registry=registry, hf_token=hf_token)
    registry.register(orchestrator.card)
    logger.info("Auto-routing orchestrator and agents initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="A2A Multi-Agent Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request/Response models
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    task_id: Optional[str] = None


class ChatResponse(BaseModel):
    task_id: str
    session_id: str
    response: str
    agents_used: list
    reasoning: str


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    stats = orchestrator.store.stats() if orchestrator else {}
    return {"status": "ok", "agents": len(registry.list_all()), **stats}


@app.get("/agents")
def list_agents():
    """A2A agent discovery endpoint."""
    return {"agents": registry.list_all()}


@app.post("/knowledge/upload")
async def upload_knowledge(
    file: UploadFile = File(...),
    source_name: str = Form(""),
):
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")
    content = await file.read()
    metadata = {"source_name": source_name or file.filename}
    result = orchestrator.store.ingest_file(
        file_name=file.filename,
        content=content,
        mime_type=file.content_type or "",
        metadata=metadata,
    )
    return {"ok": True, **result}


@app.get("/knowledge/search")
def search_knowledge(query: str = Query(...), limit: int = Query(5, ge=1, le=10)):
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")
    return {"query": query, "results": orchestrator.store.search_knowledge(query, limit=limit)}


@app.get("/memory/stats")
def memory_stats():
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")
    return orchestrator.store.stats()


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_dict()


@app.post("/tasks/send")
def send_task(req: ChatRequest):
    """Submit a task — returns SSE stream."""

    # Resolve or create task
    task: A2ATask
    if req.task_id and req.task_id in tasks:
        task = tasks[req.task_id]
    else:
        task = A2ATask(session_id=req.session_id or "")
        tasks[task.task_id] = task

    # Add user message
    task.add_message(
        A2AMessage(
            role="user",
            parts=[TextPart(text=req.message)],
        )
    )

    def event_stream():
        try:
            for event in orchestrator.orchestrate_stream(task):
                data = json.dumps(event)
                yield f"data: {data}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            err = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {err}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Task-Id": task.task_id,
        },
    )


@app.post("/tasks/send/sync")
def send_task_sync(req: ChatRequest):
    """Non-streaming version for simpler clients."""
    task = A2ATask(session_id=req.session_id or "")
    tasks[task.task_id] = task
    task.add_message(
        A2AMessage(role="user", parts=[TextPart(text=req.message)])
    )
    result = orchestrator.orchestrate(task)
    return ChatResponse(
        task_id=task.task_id,
        session_id=task.session_id,
        response=result["response"],
        agents_used=result["agents_used"],
        reasoning=result["reasoning"],
    )


# ─────────────────────────────────────────────
# Serve static UI
# ─────────────────────────────────────────────

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
