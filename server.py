"""Compatibility entry point for tools that import server:app from the repo root."""

from __future__ import annotations

import importlib.util
from pathlib import Path


INNER_SERVER_PATH = Path(__file__).resolve().parent / "a2a_multiagent" / "server.py"
SPEC = importlib.util.spec_from_file_location("a2a_multiagent_server", INNER_SERVER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load FastAPI app from {INNER_SERVER_PATH}")

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

app = MODULE.app
