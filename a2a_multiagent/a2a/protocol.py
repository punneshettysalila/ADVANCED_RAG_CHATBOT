"""
A2A (Agent-to-Agent) Protocol Implementation
Follows the Google A2A spec: task lifecycle, agent cards, message parts
"""

from __future__ import annotations

import uuid
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class TaskState(str, Enum):
    SUBMITTED   = "submitted"
    WORKING     = "working"
    COMPLETED   = "completed"
    FAILED      = "failed"
    CANCELLED   = "cancelled"


class PartType(str, Enum):
    TEXT        = "text"
    CODE        = "code"
    DATA        = "data"
    ERROR       = "error"


class AgentCapability(str, Enum):
    RESEARCH    = "research"
    CODE_GEN    = "code_generation"
    SUMMARIZE   = "summarization"
    GENERAL_QA  = "general_qa"
    ORCHESTRATE = "orchestration"


# ─────────────────────────────────────────────
# Message Parts
# ─────────────────────────────────────────────

@dataclass
class TextPart:
    text: str
    type: str = PartType.TEXT

    def to_dict(self) -> Dict:
        return {"type": self.type, "text": self.text}


@dataclass
class CodePart:
    code: str
    language: str = "python"
    type: str = PartType.CODE

    def to_dict(self) -> Dict:
        return {"type": self.type, "code": self.code, "language": self.language}


@dataclass
class DataPart:
    data: Any
    mime_type: str = "application/json"
    type: str = PartType.DATA

    def to_dict(self) -> Dict:
        return {"type": self.type, "data": self.data, "mime_type": self.mime_type}


@dataclass
class ErrorPart:
    message: str
    code: int = 500
    type: str = PartType.ERROR

    def to_dict(self) -> Dict:
        return {"type": self.type, "message": self.message, "code": self.code}


# ─────────────────────────────────────────────
# A2A Message
# ─────────────────────────────────────────────

@dataclass
class A2AMessage:
    role: str               # "user" | "agent"
    parts: List[Any]        # TextPart | CodePart | DataPart | ErrorPart
    agent_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "parts": [p.to_dict() if hasattr(p, "to_dict") else p for p in self.parts],
        }

    def get_text(self) -> str:
        texts = []
        for p in self.parts:
            if hasattr(p, "text"):
                texts.append(p.text)
            elif hasattr(p, "code"):
                texts.append(f"```{p.language}\n{p.code}\n```")
            elif hasattr(p, "message"):
                texts.append(p.message)
        return "\n".join(texts)


# ─────────────────────────────────────────────
# A2A Task
# ─────────────────────────────────────────────

@dataclass
class A2ATask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: TaskState = TaskState.SUBMITTED
    messages: List[A2AMessage] = field(default_factory=list)
    assigned_agent: Optional[str] = None  # Optional: used for tracking only, NOT for routing
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    
    # A2A Protocol Note: This implementation uses parallel execution.
    # The assigned_agent field is for tracking purposes only and does NOT
    # control which agent processes the task (all agents process all tasks).

    def add_message(self, message: A2AMessage):
        self.messages.append(message)
        self.updated_at = time.time()

    def set_state(self, state: TaskState):
        self.state = state
        self.updated_at = time.time()

    def get_latest_user_message(self) -> Optional[str]:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.get_text()
        return None

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "state": self.state,
            "assigned_agent": self.assigned_agent,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────
# Agent Card (A2A discovery spec)
# ─────────────────────────────────────────────

@dataclass
class AgentCard:
    agent_id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    version: str = "1.0.0"
    model: str = "unknown"

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": [c.value for c in self.capabilities],
            "version": self.version,
            "model": self.model,
        }


# ─────────────────────────────────────────────
# Agent Registry
# ─────────────────────────────────────────────

class AgentRegistry:
    """Central registry for agent discovery — A2A protocol requirement."""

    def __init__(self):
        self._agents: Dict[str, AgentCard] = {}

    def register(self, card: AgentCard):
        self._agents[card.agent_id] = card

    def discover(self, capability: Optional[AgentCapability] = None) -> List[AgentCard]:
        if capability is None:
            return list(self._agents.values())
        return [
            card for card in self._agents.values()
            if capability in card.capabilities
        ]

    def get(self, agent_id: str) -> Optional[AgentCard]:
        return self._agents.get(agent_id)

    def list_all(self) -> List[Dict]:
        return [card.to_dict() for card in self._agents.values()]


# ─────────────────────────────────────────────
# Status Event (SSE streaming)
# ─────────────────────────────────────────────

@dataclass
class StatusEvent:
    task_id: str
    state: TaskState
    agent_id: Optional[str]
    message: Optional[str] = None
    final: bool = False

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "state": self.state,
            "agent_id": self.agent_id,
            "message": self.message,
            "final": self.final,
        }
