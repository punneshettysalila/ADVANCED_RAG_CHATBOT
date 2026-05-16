"""
Orchestrator Agent — Auto-Routing Multi-Agent Execution

Routes each query to the most relevant agent path:
  - research_and_summarize
  - research_only
  - summarize_only
  - general_qa
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from a2a.protocol import (
    A2AMessage, A2ATask, AgentCapability, AgentRegistry,
    TextPart, TaskState,
)
from agents.base import BaseAgent, DEFAULT_HF_MODEL
from agents.specialized import ResearchAgent, SummarizerAgent, GeneralQAAgent

logger = logging.getLogger(__name__)


class ParallelOrchestratorAgent(BaseAgent):
    """
    Auto-routes each query to the most suitable path while keeping a
    multi-agent architecture (Research, Summarizer, General QA).
    """

    def __init__(self, registry: AgentRegistry, hf_token: Optional[str] = None):
        super().__init__(
            agent_id="orchestrator",
            name="Hermes",
            description="Swiftly coordinates the Research, Summarizer, and General QA agents.",
            capabilities=[AgentCapability.ORCHESTRATE],
            model=DEFAULT_HF_MODEL,
            hf_token=hf_token,
        )
        self.registry = registry
        self.hf_token = hf_token

        # Initialize all specialized agents
        self.research = ResearchAgent(hf_token=hf_token)
        self.summarizer = SummarizerAgent(hf_token=hf_token)
        self.general_qa = GeneralQAAgent(hf_token=hf_token)

        # Register all agents
        for agent in [self, self.research, self.summarizer, self.general_qa]:
            registry.register(agent.card)

        logger.info("Auto-routing orchestrator initialized")

    @property
    def system_prompt(self) -> str:
        return ""  # orchestrator doesn't need its own prompt

    def _make_task(self, query: str, session_id: str) -> A2ATask:
        """Create a task from a query. No agent assignment per A2A protocol."""
        task = A2ATask(session_id=session_id)
        task.add_message(A2AMessage(role="user", parts=[TextPart(text=query)]))
        # Note: assigned_agent field is NOT set - parallel execution means
        # no single agent is "assigned" to the task
        return task

    def _run_agent(self, agent: BaseAgent, task: A2ATask, agent_name: str) -> Dict:
        """Execute a single agent and return its result."""
        try:
            logger.info(f"Starting {agent_name}...")
            msg = agent.run(task)
            output = msg.get_text()
            logger.info(f"{agent_name} completed")
            return {
                "agent_id": agent.agent_id,
                "agent_name": agent_name,
                "success": True,
                "output": output,
                "error": None
            }
        except Exception as e:
            logger.error(f"{agent_name} failed: {e}")
            return {
                "agent_id": agent.agent_id,
                "agent_name": agent_name,
                "success": False,
                "output": "",
                "error": str(e)
            }

    def _detect_route(self, query: str) -> Dict[str, str]:
        """
        Lightweight auto-router.
        Returns:
          {"intent": "...", "reasoning": "..."}
        """
        lowered = query.lower()
        query_len = len(query.strip())

        summarize_keywords = [
            "summarize", "summarise", "summary", "tl;dr", "in short",
            "condense", "briefly", "short version",
        ]
        research_keywords = [
            "explain", "how", "why", "compare", "difference",
            "architecture", "design", "tradeoff", "pros and cons",
            "history", "analyze", "analysis",
        ]
        casual_keywords = [
            "hi", "hello", "hey", "thanks", "thank you", "good morning",
            "how are you", "who are you",
        ]

        has_summary_intent = any(k in lowered for k in summarize_keywords)
        has_research_intent = any(k in lowered for k in research_keywords)
        has_casual_intent = any(k in lowered for k in casual_keywords)
        likely_long_text = query_len > 900 or query.count("\n") > 8

        if has_summary_intent and (likely_long_text or "following" in lowered or "text" in lowered):
            return {
                "intent": "summarize_only",
                "reasoning": "Detected explicit summarization request for provided content.",
            }

        if has_summary_intent and has_research_intent:
            return {
                "intent": "research_and_summarize",
                "reasoning": "Detected a deep topic request that also asks for concise synthesis.",
            }

        if has_research_intent and not has_casual_intent:
            return {
                "intent": "research_only",
                "reasoning": "Detected an explanatory or analytical query that benefits from research depth.",
            }

        if has_summary_intent:
            return {
                "intent": "summarize_only",
                "reasoning": "Detected summarization keywords in the query.",
            }

        return {
            "intent": "general_qa",
            "reasoning": "Detected conversational or general Q&A intent.",
        }

    def orchestrate(self, task: A2ATask) -> Dict:
        """
        Auto-route and execute the selected path.
        """
        query = task.get_latest_user_message() or ""
        route = self._detect_route(query)
        intent = route["intent"]
        reasoning = route["reasoning"]
        logger.info(f"Auto-routed query to intent={intent}")

        final_response = ""
        agents_used: List[str] = []
        results: Dict[str, Dict] = {}

        if intent == "research_and_summarize":
            research_task = self._make_task(query, task.session_id)
            research_result = self._run_agent(self.research, research_task, "Sophia")
            results["research"] = research_result
            if research_result["success"] and research_result["output"]:
                agents_used.append("research-agent")
                summarize_query = f"Please summarize clearly and concisely:\n\n{research_result['output']}"
            else:
                summarize_query = query

            summarizer_task = self._make_task(summarize_query, task.session_id)
            summarizer_result = self._run_agent(self.summarizer, summarizer_task, "Eunoia")
            results["summarizer"] = summarizer_result

            if summarizer_result["success"] and summarizer_result["output"]:
                agents_used.append("summarizer-agent")
                final_response = f"## 🌸 Eunoia\n\n{summarizer_result['output']}"
            elif research_result["success"] and research_result["output"]:
                final_response = f"## 🦉 Sophia\n\n{research_result['output']}"

        elif intent == "research_only":
            research_task = self._make_task(query, task.session_id)
            research_result = self._run_agent(self.research, research_task, "Sophia")
            results["research"] = research_result
            if research_result["success"] and research_result["output"]:
                agents_used.append("research-agent")
                final_response = f"## 🦉 Sophia\n\n{research_result['output']}"

        elif intent == "summarize_only":
            summarizer_task = self._make_task(query, task.session_id)
            summarizer_result = self._run_agent(self.summarizer, summarizer_task, "Eunoia")
            results["summarizer"] = summarizer_result
            if summarizer_result["success"] and summarizer_result["output"]:
                agents_used.append("summarizer-agent")
                final_response = f"## 🌸 Eunoia\n\n{summarizer_result['output']}"

        else:
            qa_task = self._make_task(query, task.session_id)
            qa_result = self._run_agent(self.general_qa, qa_task, "Euphrosyne")
            results["general_qa"] = qa_result
            if qa_result["success"] and qa_result["output"]:
                agents_used.append("general-qa-agent")
                final_response = f"## ☀️ Euphrosyne\n\n{qa_result['output']}"

        if not final_response:
            final_response = "No agent generated a response."

        # Add final message to task
        task.add_message(A2AMessage(
            role="agent",
            agent_id=self.agent_id,
            parts=[TextPart(text=final_response)],
        ))
        task.set_state(TaskState.COMPLETED)

        return {
            "response": final_response,
            "agents_used": agents_used,
            "reasoning": reasoning,
            "intent": intent,
            "task": task.to_dict(),
            "individual_results": results,
        }

    def orchestrate_stream(self, task: A2ATask):
        """
        Streaming version - auto-routes and streams from the selected path.
        """
        query = task.get_latest_user_message() or ""
        route = self._detect_route(query)
        intent = route["intent"]
        reasoning = route["reasoning"]

        yield {
            "type": "status",
            "message": "🌷 Auto-routing your request...",
            "step": "initialize"
        }
        yield {
            "type": "status",
            "message": f"🎯 Routed to {intent.replace('_', ' ')}",
            "step": "routed",
            "intent": intent,
            "reasoning": reasoning,
        }

        results: Dict[str, Dict] = {}
        agents_used = []
        full_text = ""
        try:
            if intent == "research_and_summarize":
                yield {"type": "status", "message": "🦉 Sophia researching...", "step": "research"}
                research_task = self._make_task(query, task.session_id)
                research_chunks: List[str] = []
                for chunk in self.research.run_stream(research_task):
                    research_chunks.append(chunk)
                research_text = "".join(research_chunks)
                results["research"] = {
                    "agent_id": "research-agent",
                    "agent_name": "Sophia",
                    "success": bool(research_text.strip()),
                    "output": research_text,
                    "error": None if research_text.strip() else "Empty research output",
                }
                if research_text.strip():
                    agents_used.append("research-agent")

                yield {"type": "status", "message": "🌸 Eunoia summarizing...", "step": "summarize"}
                summarize_query = f"Please summarize clearly and concisely:\n\n{research_text or query}"
                summarizer_task = self._make_task(summarize_query, task.session_id)
                yield {"type": "chunk", "text": "\n## 🌸 Eunoia\n\n", "agent_id": "summarizer-agent"}
                full_text += "\n## 🌸 Eunoia\n\n"
                for chunk in self.summarizer.run_stream(summarizer_task):
                    full_text += chunk
                    yield {"type": "chunk", "text": chunk, "agent_id": "summarizer-agent"}
                agents_used.append("summarizer-agent")

            elif intent == "research_only":
                yield {"type": "status", "message": "🦉 Sophia researching...", "step": "research"}
                research_task = self._make_task(query, task.session_id)
                yield {"type": "chunk", "text": "\n## 🦉 Sophia\n\n", "agent_id": "research-agent"}
                full_text += "\n## 🦉 Sophia\n\n"
                for chunk in self.research.run_stream(research_task):
                    full_text += chunk
                    yield {"type": "chunk", "text": chunk, "agent_id": "research-agent"}
                agents_used.append("research-agent")

            elif intent == "summarize_only":
                yield {"type": "status", "message": "🌸 Eunoia summarizing...", "step": "summarize"}
                summarizer_task = self._make_task(query, task.session_id)
                yield {"type": "chunk", "text": "\n## 🌸 Eunoia\n\n", "agent_id": "summarizer-agent"}
                full_text += "\n## 🌸 Eunoia\n\n"
                for chunk in self.summarizer.run_stream(summarizer_task):
                    full_text += chunk
                    yield {"type": "chunk", "text": chunk, "agent_id": "summarizer-agent"}
                agents_used.append("summarizer-agent")

            else:
                yield {"type": "status", "message": "☀️ Euphrosyne responding...", "step": "qa"}
                qa_task = self._make_task(query, task.session_id)
                yield {"type": "chunk", "text": "\n## ☀️ Euphrosyne\n\n", "agent_id": "general-qa-agent"}
                full_text += "\n## ☀️ Euphrosyne\n\n"
                for chunk in self.general_qa.run_stream(qa_task):
                    full_text += chunk
                    yield {"type": "chunk", "text": chunk, "agent_id": "general-qa-agent"}
                agents_used.append("general-qa-agent")
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield {"type": "error", "message": str(e)}
            return

        if not full_text.strip():
            full_text = "No agent generated a response."

        # Finalize
        task.add_message(A2AMessage(
            role="agent",
            agent_id=self.agent_id,
            parts=[TextPart(text=full_text)],
        ))
        task.set_state(TaskState.COMPLETED)

        yield {
            "type": "done",
            "agents_used": agents_used,
            "reasoning": reasoning,
            "intent": intent,
            "task_id": task.task_id,
        }
