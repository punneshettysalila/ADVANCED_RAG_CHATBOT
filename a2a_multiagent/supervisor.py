"""
Supervisor meta-agent.

Decides:
- sequential vs parallel execution
- short vs detailed response
- whether verification is required
- which tools to use
- whether knowledge grounding is needed

This is intentionally lightweight and deterministic by default, with
optional HF-assisted planning if available.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent, DEFAULT_HF_MODEL, HFInferenceWrapper
from a2a.protocol import AgentCapability

logger = logging.getLogger(__name__)


@dataclass
class SupervisorPlan:
    intent: str = "general_qa"
    execution_mode: str = "sequential"
    response_style: str = "short"
    verify: bool = True
    use_tools: List[str] = field(default_factory=list)
    knowledge_query: str = ""
    retrieve_memory: bool = True
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "execution_mode": self.execution_mode,
            "response_style": self.response_style,
            "verify": self.verify,
            "use_tools": list(self.use_tools),
            "knowledge_query": self.knowledge_query,
            "retrieve_memory": self.retrieve_memory,
            "reasoning": self.reasoning,
        }


class SupervisorAgent(BaseAgent):
    """Meta-agent that plans how the rest of the system should work."""

    def __init__(self, hf_token: Optional[str] = None):
        super().__init__(
            agent_id="supervisor",
            name="Astra",
            description="Supervisor AI that chooses execution strategy, tools, and verification.",
            capabilities=[AgentCapability.ORCHESTRATE],
            model=DEFAULT_HF_MODEL,
            hf_token=hf_token,
        )
        self.hf_token = hf_token

    @property
    def system_prompt(self) -> str:
        return (
            "You are a supervisor meta-agent. Classify the best execution plan for the user request. "
            "Return only JSON with keys: intent, execution_mode, response_style, verify, use_tools, "
            "knowledge_query, retrieve_memory, reasoning. Use short concise reasoning."
        )

    def _heuristic_plan(self, query: str) -> SupervisorPlan:
        lowered = query.lower().strip()
        plan = SupervisorPlan()
        plan.knowledge_query = query

        research_terms = ["explain", "compare", "how", "why", "architecture", "design", "analyze", "current", "latest", "research"]
        summary_terms = ["summarize", "summary", "tl;dr", "short version", "brief"]
        math_terms = ["calculate", "compute", "math", "equation", "solve", "+", "-", "*", "/"]
        code_terms = ["python", "code", "script", "debug", "run code", "execute", "snippet"]
        web_terms = ["latest", "current", "news", "web", "search", "internet", "today"]
        docs_terms = ["pdf", "dataset", "upload", "document", "file", "knowledge base", "internal knowledge"]

        if any(term in lowered for term in summary_terms):
            plan.intent = "summarize_only"
            plan.response_style = "short"
            plan.verify = True
        elif any(term in lowered for term in research_terms):
            plan.intent = "research_and_summarize"
            plan.execution_mode = "parallel"
            plan.response_style = "detailed"
            plan.verify = True
        else:
            plan.intent = "general_qa"
            plan.response_style = "short"
            plan.verify = False

        if any(term in lowered for term in math_terms):
            plan.use_tools.append("calculator")
            plan.verify = True
            plan.response_style = "short"

        if any(term in lowered for term in web_terms):
            plan.use_tools.append("web_search")
            plan.execution_mode = "parallel"
            plan.verify = True
            plan.response_style = "detailed"

        if any(term in lowered for term in code_terms):
            plan.use_tools.append("code_execution")
            plan.execution_mode = "sequential"
            plan.verify = True
            plan.response_style = "detailed"

        if any(term in lowered for term in docs_terms):
            plan.retrieve_memory = True
            plan.verify = True
            plan.response_style = "detailed"

        if len(query) > 220 or query.count("\n") > 2:
            plan.response_style = "detailed"
            plan.verify = True
            if plan.intent == "general_qa":
                plan.execution_mode = "parallel"

        if any(ch.isdigit() for ch in query) and any(sym in query for sym in ["+", "-", "*", "/", "="]):
            if "calculator" not in plan.use_tools:
                plan.use_tools.append("calculator")
            plan.verify = True

        plan.reasoning = self._build_reasoning(plan, query)
        return plan

    def _build_reasoning(self, plan: SupervisorPlan, query: str) -> str:
        if plan.use_tools:
            tools = ", ".join(plan.use_tools)
            return f"Use {tools} first, then answer with {plan.response_style} detail."
        if plan.intent == "research_and_summarize":
            return "Deep topic request benefits from parallel research and synthesis."
        if plan.intent == "summarize_only":
            return "Explicit summarization request should stay concise."
        return "Simple conversational query can be answered directly."

    def decide(self, query: str, memory_hits: Optional[List[Dict[str, Any]]] = None) -> SupervisorPlan:
        query = query or ""
        if not self.hf_token:
            return self._heuristic_plan(query)

        try:
            llm = HFInferenceWrapper(model=self.model, token=self.hf_token)
            memory_context = ""
            if memory_hits:
                lines = []
                for idx, hit in enumerate(memory_hits[:5], start=1):
                    lines.append(f"{idx}. {hit.get('kind', 'memory')}: {hit.get('title', '')} | {hit.get('content', '')[:220]}")
                memory_context = "\n".join(lines)

            prompt = (
                "Return a JSON plan for this user request.\n\n"
                f"User request: {query}\n\n"
                f"Relevant memory:\n{memory_context or 'none'}\n\n"
                "Valid values:\n"
                "- intent: general_qa | research_and_summarize | summarize_only\n"
                "- execution_mode: sequential | parallel\n"
                "- response_style: short | detailed\n"
                "- verify: true | false\n"
                "- use_tools: array of calculator | web_search | code_execution\n"
                "- knowledge_query: string\n"
                "- retrieve_memory: true | false\n"
                "- reasoning: one short sentence\n"
                "Return only JSON."
            )
            raw = llm.chat(messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=220)
            cleaned = raw.strip()
            if "```" in cleaned:
                cleaned = cleaned.split("```", 2)[1]
                cleaned = cleaned.replace("json", "", 1).strip()
            data = json.loads(cleaned)
            plan = SupervisorPlan(
                intent=data.get("intent", "general_qa"),
                execution_mode=data.get("execution_mode", "sequential"),
                response_style=data.get("response_style", "short"),
                verify=bool(data.get("verify", True)),
                use_tools=list(data.get("use_tools", [])),
                knowledge_query=data.get("knowledge_query") or query,
                retrieve_memory=bool(data.get("retrieve_memory", True)),
                reasoning=data.get("reasoning", ""),
            )
            if plan.intent not in {"general_qa", "research_and_summarize", "summarize_only"}:
                plan.intent = "general_qa"
            if plan.execution_mode not in {"sequential", "parallel"}:
                plan.execution_mode = "sequential"
            if plan.response_style not in {"short", "detailed"}:
                plan.response_style = "short"
            plan.reasoning = plan.reasoning or self._build_reasoning(plan, query)
            return plan
        except Exception as exc:
            logger.warning(f"Supervisor planning failed, using heuristics: {exc}")
            return self._heuristic_plan(query)
