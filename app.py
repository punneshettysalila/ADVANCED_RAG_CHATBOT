"""Compatibility entry point for deployment platforms that expect app:app."""

from __future__ import annotations

import sys
from pathlib import Path


INNER_APP_DIR = Path(__file__).resolve().parent / "a2a_multiagent"
if str(INNER_APP_DIR) not in sys.path:
    sys.path.insert(0, str(INNER_APP_DIR))

from server import app as app  # noqa: E402
