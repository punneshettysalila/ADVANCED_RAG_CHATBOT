"""
Supervised orchestrator with learning loop, grounding, tools, and trust layer.

This file intentionally leaves the older orchestrator implementations in place
for reference, but the server can import this class to use the upgraded path.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from a2a.protocol import A2AMessage, A2ATask, AgentCapability, AgentRegistry, TextPart, TaskState
from agents.base import BaseAgent, DEFAULT_HF_MODEL
from agents.specialized import GeneralQAAgent, ResearchAgent, SummarizerAgent
from experience_store import ExperienceStore
from supervisor import SupervisorAgent, SupervisorPlan
from tools import (
    ToolResult,
    calculator,
    execute_python_code,
    extract_code_block,
    format_tool_context,
    web_search,
)
from verification import VerificationAgent, VerificationResult

logger = logging.getLogger(__name__)


class ParallelOrchestratorAgent(BaseAgent):
    """Agentic orchestrator that plans, grounds, verifies, and learns."""

    def __init__(self, registry: AgentRegistry, hf_token: Optional[str] = None):
        super().__init__(
            agent_id="orchestrator",
            name="Hermes",
            description="Supervisor-driven orchestrator with memory, tools, and verification.",
            capabilities=[AgentCapability.ORCHESTRATE],
            model=DEFAULT_HF_MODEL,
            hf_token=hf_token,
        )
        self.registry = registry
        self.hf_token = hf_token
        self.store = ExperienceStore(hf_token=hf_token)
        self.supervisor = SupervisorAgent(hf_token=hf_token)
        self.verifier = VerificationAgent(hf_token=hf_token)
        self.research = ResearchAgent(hf_token=hf_token)
        self.summarizer = SummarizerAgent(hf_token=hf_token)
        self.general_qa = GeneralQAAgent(hf_token=hf_token)

        for agent in [self, self.supervisor, self.verifier, self.research, self.summarizer, self.general_qa]:
            registry.register(agent.card)

        logger.info("Supervised orchestrator initialized")

    @property
    def system_prompt(self) -> str:
        return ""

    def _make_task(self, query: str, session_id: str) -> A2ATask:
        task = A2ATask(session_id=session_id)
        task.add_message(A2AMessage(role="user", parts=[TextPart(text=query)]))
        return task

    def _run_agent(self, agent: BaseAgent, query: str, session_id: str) -> str:
        task = self._make_task(query, session_id)
        message = agent.run(task)
        return message.get_text().strip()

    def _run_agent_stream(self, agent: BaseAgent, query: str, session_id: str) -> str:
        task = self._make_task(query, session_id)
        chunks: List[str] = []
        for chunk in agent.run_stream(task):
            chunks.append(chunk)
        return "".join(chunks).strip()

    @staticmethod
    def _format_hits(hits: List[Dict[str, Any]], heading: str) -> str:
        if not hits:
            return ""
        lines = [heading]
        for idx, hit in enumerate(hits, start=1):
            preview = (hit.get("content") or "")[:260].replace("\n", " ")
            lines.append(f"{idx}. {hit.get('title', 'item')}: {preview}")
        return "\n".join(lines)

    def _execute_tools(self, query: str, plan: SupervisorPlan) -> List[ToolResult]:
        results: List[ToolResult] = []
        lowered = query.lower()

        if "calculator" in plan.use_tools or any(sym in lowered for sym in ["calculate", "compute", "+", "-", "*", "/", "="]):
            expression = self._extract_math_expression(query)
            results.append(calculator(expression or query))

        if "web_search" in plan.use_tools:
            results.append(web_search(query))

        if "code_execution" in plan.use_tools:
            code = extract_code_block(query)
            if not code:
                code = self._extract_code_request(query)
            if code:
                results.append(execute_python_code(code))
            else:
                results.append(ToolResult("code_execution", False, "", error="No executable code found"))

        return results

    @staticmethod
    def _extract_math_expression(query: str) -> str:
        import re

        match = re.search(r"([0-9\s\+\-\*\/\(\)\.^]+)", query)
        return match.group(1).strip() if match else query.strip()

    @staticmethod
    def _extract_code_request(query: str) -> str:
        lowered = query.lower()
        if "python" in lowered and "print(" in lowered:
            return query
        return ""

    def _build_context(self, query: str, plan: SupervisorPlan) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], List[ToolResult]]:
        memory_hits = [hit.to_dict() for hit in self.store.search_experiences(query, limit=4)]
        knowledge_hits = [hit.to_dict() for hit in self.store.search_knowledge(query, limit=5)]
        memory_block = self.store.summarize_memory_context(query, limit=3) if plan.retrieve_memory else ""
        knowledge_block = self._format_hits(knowledge_hits, "Internal knowledge:")
        tool_results = self._execute_tools(query, plan)
        tool_block = format_tool_context(tool_results)

        blocks = [block for block in [memory_block, knowledge_block, tool_block] if block]
        return "\n\n".join(blocks), memory_hits, knowledge_hits, tool_results

    def _run_parallel_agents(self, query: str, session_id: str, context_block: str) -> Dict[str, str]:
        augmented_query = query if not context_block else f"{query}\n\n{context_block}"
        with ThreadPoolExecutor(max_workers=2) as pool:
            research_future = pool.submit(self._run_agent, self.research, augmented_query, session_id)
            qa_future = pool.submit(self._run_agent, self.general_qa, augmented_query, session_id)
            research_text = research_future.result()
            qa_text = qa_future.result()
        return {"research": research_text, "qa": qa_text}

    def _verify(self, query: str, response: str, context_block: str, plan: SupervisorPlan) -> VerificationResult:
        if not plan.verify and len(response) > 0:
            return VerificationResult(approved=True, confidence=0.55, summary="Verification skipped by plan.")
        return self.verifier.verify(query=query, answer=response, evidence=context_block, memory=context_block)

    def _record(self, task: A2ATask, query: str, response: str, agents_used: List[str], plan: SupervisorPlan, verification: VerificationResult, tool_results: List[ToolResult], knowledge_hits: List[Dict[str, Any]]) -> None:
        metadata = {
            "tool_results": [tool.to_dict() for tool in tool_results],
            "knowledge_hits": knowledge_hits,
        }
        self.store.record_experience(
            query=query,
            agents_used=agents_used,
            response=response,
            task_id=task.task_id,
            session_id=task.session_id,
            plan=plan.to_dict(),
            verification=verification.to_dict(),
            metadata=metadata,
        )

    def _finalize_response(self, task: A2ATask, text: str) -> None:
        task.add_message(
            A2AMessage(
                role="agent",
                agent_id=self.agent_id,
                parts=[TextPart(text=text)],
            )
        )
        task.set_state(TaskState.COMPLETED)

    def orchestrate(self, task: A2ATask) -> Dict[str, Any]:
        query = task.get_latest_user_message() or ""
        memory_hits = [hit.to_dict() for hit in self.store.search_experiences(query, limit=4)]
        plan = self.supervisor.decide(query, memory_hits)
        context_block, _, knowledge_hits, tool_results = self._build_context(query, plan)

        agents_used: List[str] = []
        response = ""

        if plan.intent == "summarize_only":
            summarize_query = query if not context_block else f"{query}\n\n{context_block}"
            response = self._run_agent(self.summarizer, summarize_query, task.session_id)
            if response:
                agents_used.append("summarizer-agent")

        elif plan.intent == "research_and_summarize":
            if plan.execution_mode == "parallel":
                parallel_outputs = self._run_parallel_agents(query, task.session_id, context_block)
                response_source = parallel_outputs.get("research") or query
                if parallel_outputs.get("qa"):
                    response_source = f"{response_source}\n\nAdditional perspective:\n{parallel_outputs['qa']}"
                summarize_query = response_source if not context_block else f"{response_source}\n\n{context_block}"
                response = self._run_agent(self.summarizer, summarize_query, task.session_id)
                if parallel_outputs.get("research"):
                    agents_used.append("research-agent")
                if parallel_outputs.get("qa"):
                    agents_used.append("general-qa-agent")
                if response:
                    agents_used.append("summarizer-agent")
            else:
                research_query = query if not context_block else f"{query}\n\n{context_block}"
                research_text = self._run_agent(self.research, research_query, task.session_id)
                if research_text:
                    agents_used.append("research-agent")
                summarize_query = research_text or research_query
                response = self._run_agent(self.summarizer, summarize_query, task.session_id)
                if response:
                    agents_used.append("summarizer-agent")

        else:
            qa_query = query if not context_block else f"{query}\n\n{context_block}"
            if len(plan.use_tools) == 1 and plan.use_tools[0] == "calculator":
                calc_result = next((tool for tool in tool_results if tool.tool_name == "calculator" and tool.success), None)
                if calc_result:
                    response = f"The result is {calc_result.output}."
                    agents_used.append("calculator-tool")
                else:
                    response = self._run_agent(self.general_qa, qa_query, task.session_id)
                    if response:
                        agents_used.append("general-qa-agent")
            else:
                response = self._run_agent(self.general_qa, qa_query, task.session_id)
                if response:
                    agents_used.append("general-qa-agent")

        if not response:
            response = "No agent generated a response."

        verification = self._verify(query=query, response=response, context_block=context_block, plan=plan)
        if verification.corrected_answer.strip():
            response = verification.corrected_answer.strip()

        self._record(task, query, response, agents_used, plan, verification, tool_results, knowledge_hits)
        self._finalize_response(task, response)

        return {
            "response": response,
            "agents_used": agents_used,
            "reasoning": plan.reasoning,
            "intent": plan.intent,
            "execution_mode": plan.execution_mode,
            "response_style": plan.response_style,
            "verification": verification.to_dict(),
            "tool_results": [tool.to_dict() for tool in tool_results],
            "memory_hits": memory_hits,
            "knowledge_hits": knowledge_hits,
            "plan": plan.to_dict(),
            "task": task.to_dict(),
        }

    def orchestrate_stream(self, task: A2ATask):
        query = task.get_latest_user_message() or ""
        memory_hits = [hit.to_dict() for hit in self.store.search_experiences(query, limit=4)]
        yield {"type": "status", "message": "Supervisor analyzing your request...", "step": "plan"}
        plan = self.supervisor.decide(query, memory_hits)
        yield {
            "type": "status",
            "message": f"Plan: {plan.intent} via {plan.execution_mode} execution",
            "step": "planned",
            "intent": plan.intent,
            "execution_mode": plan.execution_mode,
            "response_style": plan.response_style,
            "reasoning": plan.reasoning,
        }

        context_block, _, knowledge_hits, tool_results = self._build_context(query, plan)
        if context_block:
            yield {"type": "status", "message": "Grounding with memory and knowledge...", "step": "grounding", "context": context_block[:1200]}
        if tool_results:
            yield {"type": "status", "message": "Tools completed.", "step": "tools", "tools": [tool.to_dict() for tool in tool_results]}

        agents_used: List[str] = []
        response = ""

        try:
            if plan.intent == "summarize_only":
                summarize_query = query if not context_block else f"{query}\n\n{context_block}"
                yield {"type": "status", "message": "Summarizer distilling the answer...", "step": "summarize"}
                response = self._run_agent_stream(self.summarizer, summarize_query, task.session_id)
                if response:
                    agents_used.append("summarizer-agent")

            elif plan.intent == "research_and_summarize":
                if plan.execution_mode == "parallel":
                    yield {"type": "status", "message": "Running research and QA in parallel...", "step": "parallel"}
                    outputs = self._run_parallel_agents(query, task.session_id, context_block)
                    if outputs.get("research"):
                        agents_used.append("research-agent")
                    if outputs.get("qa"):
                        agents_used.append("general-qa-agent")
                    response_source = outputs.get("research") or query
                    if outputs.get("qa"):
                        response_source = f"{response_source}\n\nAdditional perspective:\n{outputs['qa']}"
                    summarize_query = response_source if not context_block else f"{response_source}\n\n{context_block}"
                    yield {"type": "status", "message": "Summarizer synthesizing parallel outputs...", "step": "synthesize"}
                    response = self._run_agent_stream(self.summarizer, summarize_query, task.session_id)
                    if response:
                        agents_used.append("summarizer-agent")
                else:
                    research_query = query if not context_block else f"{query}\n\n{context_block}"
                    yield {"type": "status", "message": "Research agent gathering facts...", "step": "research"}
                    research_text = self._run_agent_stream(self.research, research_query, task.session_id)
                    if research_text:
                        agents_used.append("research-agent")
                    summarize_query = research_text or research_query
                    yield {"type": "status", "message": "Summarizer condensing findings...", "step": "summarize"}
                    response = self._run_agent_stream(self.summarizer, summarize_query, task.session_id)
                    if response:
                        agents_used.append("summarizer-agent")

            else:
                qa_query = query if not context_block else f"{query}\n\n{context_block}"
                if len(plan.use_tools) == 1 and plan.use_tools[0] == "calculator":
                    calc_result = next((tool for tool in tool_results if tool.tool_name == "calculator" and tool.success), None)
                    if calc_result:
                        response = f"The result is {calc_result.output}."
                        agents_used.append("calculator-tool")
                    else:
                        yield {"type": "status", "message": "General QA agent responding...", "step": "qa"}
                        response = self._run_agent_stream(self.general_qa, qa_query, task.session_id)
                        if response:
                            agents_used.append("general-qa-agent")
                else:
                    yield {"type": "status", "message": "General QA agent responding...", "step": "qa"}
                    response = self._run_agent_stream(self.general_qa, qa_query, task.session_id)
                    if response:
                        agents_used.append("general-qa-agent")
        except Exception as exc:
            logger.error(f"Stream error: {exc}")
            yield {"type": "error", "message": str(exc)}
            return

        if not response:
            response = "No agent generated a response."

        verification = self._verify(query=query, response=response, context_block=context_block, plan=plan)
        if verification.corrected_answer.strip():
            response = verification.corrected_answer.strip()

        self._record(task, query, response, agents_used, plan, verification, tool_results, knowledge_hits)
        self._finalize_response(task, response)

        yield {
            "type": "chunk",
            "text": response,
            "agent_id": self.agent_id,
        }
        yield {
            "type": "done",
            "agents_used": agents_used,
            "reasoning": plan.reasoning,
            "intent": plan.intent,
            "execution_mode": plan.execution_mode,
            "response_style": plan.response_style,
            "verification": verification.to_dict(),
            "tool_results": [tool.to_dict() for tool in tool_results],
            "task_id": task.task_id,
        }
