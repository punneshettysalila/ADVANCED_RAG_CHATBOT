"""
DEPRECATED - LangGraph-based Orchestrator with Auto-Routing

⚠️ THIS FILE IS NOT IN USE - SUPERSEDED BY orchestrator_parallel.py ⚠️

This orchestrator used LangGraph to automatically route queries based on LLM classification:
  - classify → [research] → [summarize] → respond
  
It was replaced with parallel execution (all agents run simultaneously)
to avoid auto-routing and ensure full A2A protocol compliance.

DO NOT USE THIS FILE. It remains for reference only.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from a2a.protocol import (
    A2AMessage, A2ATask, AgentCapability, AgentRegistry,
    TextPart, TaskState,
)
from agents.base import BaseAgent, HFInferenceWrapper
from agents.specialized import ResearchAgent, SummarizerAgent, GeneralQAAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LangGraph State
# ─────────────────────────────────────────────

class PipelineState(TypedDict):
    query: str
    session_id: str
    intent: str                    # "research_and_summarize" | "general_qa" | "summarize_only"
    needs_research: bool
    needs_summary: bool
    research_output: str
    summary_output: str
    qa_output: str
    final_response: str
    agents_used: List[str]
    reasoning: str


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

class OrchestratorAgent(BaseAgent):
    """
    Orchestrates Research → Summarizer → General QA using LangGraph.
    Decides the right pipeline per query via an LLM classifier node.
    """

    def __init__(self, registry: AgentRegistry, hf_token: Optional[str] = None):
        super().__init__(
            agent_id="orchestrator",
            name="Hermes",
            description="Routes queries to Sophia, Eunoia, or Euphrosyne agents.",
            capabilities=[AgentCapability.ORCHESTRATE],
            model="mistralai/Mistral-7B-Instruct-v0.3",
            hf_token=hf_token,
        )
        self.registry   = registry
        self.hf_token   = hf_token

        # Specialized agents
        self.research   = ResearchAgent(hf_token=hf_token)
        self.summarizer = SummarizerAgent(hf_token=hf_token)
        self.general_qa = GeneralQAAgent(hf_token=hf_token)

        # Register all cards
        for agent in [self, self.research, self.summarizer, self.general_qa]:
            registry.register(agent.card)

        # Build LangGraph
        self._graph = self._build_graph()

    @property
    def system_prompt(self) -> str:
        return ""  # orchestrator uses per-step prompts

    # ── Graph ──────────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(PipelineState)

        g.add_node("classify",  self._classify_node)
        g.add_node("research",  self._research_node)
        g.add_node("summarize", self._summarize_node)
        g.add_node("general_qa",self._general_qa_node)
        g.add_node("finalize",  self._finalize_node)

        g.set_entry_point("classify")

        # After classify, route based on intent
        g.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "research":   "research",
                "general_qa": "general_qa",
                "summarize":  "summarize",
            },
        )

        # After research, always summarize
        g.add_edge("research",   "summarize")
        g.add_edge("summarize",  "finalize")
        g.add_edge("general_qa", "finalize")
        g.add_edge("finalize",   END)

        return g.compile()

    # ── Routing ────────────────────────────────────────────────────────

    def _route_after_classify(self, state: PipelineState) -> str:
        intent = state.get("intent", "research_and_summarize")
        if intent == "general_qa":
            return "general_qa"
        elif intent == "summarize_only":
            return "summarize"
        else:
            return "research"

    # ── Nodes ──────────────────────────────────────────────────────────

    def _classify_node(self, state: PipelineState) -> PipelineState:
        """Use an LLM to decide which pipeline path fits the query."""
        query = state["query"]

        prompt = (
            f'Classify this user query into exactly one of these categories:\n\n'
            f'1. "research_and_summarize" — needs factual research + a clean summary '
            f'(topics, explanations, how things work, current events, comparisons)\n'
            f'2. "general_qa" — short conversational answer is enough '
            f'(opinions, simple yes/no, follow-ups, chitchat, personal questions)\n'
            f'3. "summarize_only" — user has provided text and wants it summarised\n\n'
            f'Query: "{query}"\n\n'
            f'Respond ONLY with a JSON object:\n'
            f'{{"intent": "<category>", "reasoning": "<one sentence why>"}}\n'
            f'No other text.'
        )

        try:
            llm = HFInferenceWrapper(
                model="mistralai/Mistral-7B-Instruct-v0.3",
                token=self.hf_token,
            )
            raw = llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=120,
            ).strip()

            # Strip markdown fences if present
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()

            data      = json.loads(raw)
            intent    = data.get("intent", "research_and_summarize")
            reasoning = data.get("reasoning", "")

            valid_intents = {"research_and_summarize", "general_qa", "summarize_only"}
            if intent not in valid_intents:
                intent = "research_and_summarize"

        except Exception as e:
            logger.warning(f"Classify node failed ({e}), defaulting to research_and_summarize")
            intent    = "research_and_summarize"
            reasoning = "Classification failed — defaulting to full research pipeline."

        logger.info(f"Intent: {intent} | {reasoning}")
        return {**state, "intent": intent, "reasoning": reasoning}

    def _research_node(self, state: PipelineState) -> PipelineState:
        """Run Sophia."""
        task = self._make_task(state["query"], state["session_id"])
        try:
            msg    = self.research.run(task)
            output = msg.get_text()
        except Exception as e:
            logger.error(f"Research node error: {e}")
            output = f"Research failed: {e}"

        agents_used = state.get("agents_used", []) + ["research-agent"]
        return {**state, "research_output": output, "agents_used": agents_used}

    def _summarize_node(self, state: PipelineState) -> PipelineState:
        """Run Eunoia on research output (or directly on query)."""
        # The summarizer gets either the research output or the raw query
        source = state.get("research_output") or state["query"]
        summarize_query = (
            f"Please summarise the following information for me:\n\n{source}"
            if state.get("research_output")
            else state["query"]
        )
        task = self._make_task(summarize_query, state["session_id"])
        try:
            msg    = self.summarizer.run(task)
            output = msg.get_text()
        except Exception as e:
            logger.error(f"Summarize node error: {e}")
            output = source  # fallback: return research output unsummarised

        agents_used = state.get("agents_used", []) + ["summarizer-agent"]
        return {**state, "summary_output": output, "agents_used": agents_used}

    def _general_qa_node(self, state: PipelineState) -> PipelineState:
        """Run Euphrosyne for conversational queries."""
        task = self._make_task(state["query"], state["session_id"])
        try:
            msg    = self.general_qa.run(task)
            output = msg.get_text()
        except Exception as e:
            logger.error(f"General QA node error: {e}")
            output = f"Sorry, I encountered an error: {e}"

        agents_used = state.get("agents_used", []) + ["general-qa-agent"]
        return {**state, "qa_output": output, "agents_used": agents_used}

    def _finalize_node(self, state: PipelineState) -> PipelineState:
        """Pick the right output as the final response."""
        if state.get("summary_output"):
            final = state["summary_output"]
        elif state.get("qa_output"):
            final = state["qa_output"]
        else:
            final = state.get("research_output", "I couldn't generate a response.")
        return {**state, "final_response": final}

    # ── Helpers ────────────────────────────────────────────────────────

    def _make_task(self, query: str, session_id: str) -> A2ATask:
        task = A2ATask(session_id=session_id)
        task.add_message(A2AMessage(role="user", parts=[TextPart(text=query)]))
        return task

    # ── Public API ─────────────────────────────────────────────────────

    def orchestrate(self, task: A2ATask) -> Dict:
        """Blocking orchestration — returns full result dict."""
        query = task.get_latest_user_message() or ""

        init: PipelineState = {
            "query":            query,
            "session_id":       task.session_id,
            "intent":           "",
            "needs_research":   False,
            "needs_summary":    False,
            "research_output":  "",
            "summary_output":   "",
            "qa_output":        "",
            "final_response":   "",
            "agents_used":      [],
            "reasoning":        "",
        }

        result = self._graph.invoke(init)

        # Record final message on the task
        task.add_message(A2AMessage(
            role="agent",
            agent_id=self.agent_id,
            parts=[TextPart(text=result["final_response"])],
        ))
        task.set_state(TaskState.COMPLETED)

        return {
            "response":    result["final_response"],
            "agents_used": result["agents_used"],
            "reasoning":   result["reasoning"],
            "intent":      result["intent"],
            "task":        task.to_dict(),
        }

    def orchestrate_stream(self, task: A2ATask):
        """
        Streaming orchestration — yields event dicts for SSE / Streamlit consumption.
        Event types: "status" | "chunk" | "done" | "error"
        """
        query = task.get_latest_user_message() or ""

        # ── Step 1: Classify ──────────────────────────────────────────
        yield {"type": "status", "message": "🧠 Analysing your question...", "step": "classify"}

        init: PipelineState = {
            "query": query, "session_id": task.session_id,
            "intent": "", "needs_research": False, "needs_summary": False,
            "research_output": "", "summary_output": "", "qa_output": "",
            "final_response": "", "agents_used": [], "reasoning": "",
        }
        state = self._classify_node(init)
        intent    = state["intent"]
        reasoning = state["reasoning"]

        intent_labels = {
            "research_and_summarize": "Research + Summarize pipeline",
            "general_qa":             "Direct QA (conversational)",
            "summarize_only":         "Summarize pipeline",
        }
        yield {
            "type":      "status",
            "message":   f"📋 {intent_labels.get(intent, intent)} — {reasoning}",
            "step":      "routed",
            "intent":    intent,
            "reasoning": reasoning,
        }

        agents_used: List[str] = []
        full_text = ""

        try:
            # ── Step 2: Research (if needed) ──────────────────────────
            if intent in ("research_and_summarize",):
                yield {"type": "status", "message": "🦉 Sophia gathering information...", "step": "research"}
                agents_used.append("research-agent")

                research_chunks: List[str] = []
                sub = self._make_task(query, task.session_id)
                for chunk in self.research.run_stream(sub):
                    research_chunks.append(chunk)
                state["research_output"] = "".join(research_chunks)

            elif intent == "summarize_only":
                state["research_output"] = query

            # ── Step 3: Summarize ─────────────────────────────────────
            if intent in ("research_and_summarize", "summarize_only"):
                yield {"type": "status", "message": "🌸 Eunoia distilling results...", "step": "summarize"}
                agents_used.append("summarizer-agent")

                source = state.get("research_output") or query
                summarize_query = (
                    f"Please summarise the following information:\n\n{source}"
                    if state.get("research_output") else query
                )
                sub = self._make_task(summarize_query, task.session_id)
                for chunk in self.summarizer.run_stream(sub):
                    full_text += chunk
                    yield {"type": "chunk", "text": chunk, "agent_id": "summarizer-agent"}

            # ── Step 3 alt: General QA ────────────────────────────────
            else:
                yield {"type": "status", "message": "☀️ Euphrosyne responding...", "step": "qa"}
                agents_used.append("general-qa-agent")

                sub = self._make_task(query, task.session_id)
                for chunk in self.general_qa.run_stream(sub):
                    full_text += chunk
                    yield {"type": "chunk", "text": chunk, "agent_id": "general-qa-agent"}

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield {"type": "error", "message": str(e)}
            return

        # ── Finalise ──────────────────────────────────────────────────
        task.add_message(A2AMessage(
            role="agent",
            agent_id=self.agent_id,
            parts=[TextPart(text=full_text)],
        ))
        task.set_state(TaskState.COMPLETED)

        yield {
            "type":        "done",
            "agents_used": agents_used,
            "reasoning":   reasoning,
            "intent":      intent,
            "task_id":     task.task_id,
        }
