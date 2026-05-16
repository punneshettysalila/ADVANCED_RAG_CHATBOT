"""
Specialized Agents — Research · Summarizer · General QA (fallback)
All use HuggingFace InferenceClient via BaseAgent.
"""

from __future__ import annotations

from typing import Optional

from a2a.protocol import AgentCapability
from agents.base import BaseAgent, DEFAULT_HF_MODEL


# ─────────────────────────────────────────────
# Research Agent
# ─────────────────────────────────────────────

class ResearchAgent(BaseAgent):
    """
    Deep-dives into any topic: facts, explanations, comparisons, context.
    Optimised for breadth + accuracy over brevity.
    """

    def __init__(self, hf_token: Optional[str] = None):
        super().__init__(
            agent_id="research-agent",
            name="Sophia",
            description="Wisdom-driven research that finds, explains, and synthesises information.",
            capabilities=[AgentCapability.RESEARCH],
            model=DEFAULT_HF_MODEL,
            hf_token=hf_token,
        )

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert research assistant with deep knowledge across science, "
            "technology, history, culture, and current events.\n\n"
            "Your job:\n"
            "- Provide thorough, accurate, well-reasoned answers\n"
            "- Structure your response with clear markdown (headers, bullets, bold key terms)\n"
            "- Back claims with reasoning; flag uncertainty explicitly\n"
            "- Give useful context, comparisons, or examples where helpful\n"
            "- Never fabricate citations or URLs\n\n"
            "Aim for depth and clarity. The reader wants to genuinely understand the topic."
        )


# ─────────────────────────────────────────────
# Summarizer Agent
# ─────────────────────────────────────────────

class SummarizerAgent(BaseAgent):
    """
    Distils the research output into a clean, reader-friendly summary.
    Handles both incoming raw text and final synthesis.
    """

    def __init__(self, hf_token: Optional[str] = None):
        super().__init__(
            agent_id="summarizer-agent",
            name="Eunoia",
            description="Gentle synthesis that distils complex information into clear summaries.",
            capabilities=[AgentCapability.SUMMARIZE],
            model=DEFAULT_HF_MODEL,
            hf_token=hf_token,
        )

    @property
    def system_prompt(self) -> str:
        return (
            "You are a master communicator. Your job is to take information and make it "
            "immediately accessible to a general audience.\n\n"
            "Always format your response as:\n\n"
            "## TL;DR\n"
            "One or two sentences — the absolute core takeaway.\n\n"
            "## Key Points\n"
            "3-6 bullet points, each one a complete, self-contained insight.\n\n"
            "## Why It Matters\n"
            "A short paragraph on relevance or real-world impact.\n\n"
            "Rules:\n"
            "- No jargon without explanation\n"
            "- Prefer plain language\n"
            "- Every bullet must add new information — no filler\n"
            "- Keep the whole response under 300 words unless the topic demands more"
        )


# ─────────────────────────────────────────────
# General QA Agent  (lightweight fallback)
# ─────────────────────────────────────────────

class GeneralQAAgent(BaseAgent):
    """
    Handles conversational, opinion, or simple factual questions
    that don't need deep research or a formal summary.
    """

    def __init__(self, hf_token: Optional[str] = None):
        super().__init__(
            agent_id="general-qa-agent",
            name="Euphrosyne",
            description="Cheerful conversational help for questions, opinions, and quick facts.",
            capabilities=[AgentCapability.GENERAL_QA],
            model=DEFAULT_HF_MODEL,
            hf_token=hf_token,
        )

    @property
    def system_prompt(self) -> str:
        return (
            "You are a friendly, knowledgeable AI assistant. "
            "Answer questions directly, clearly, and conversationally. "
            "Use examples to make complex ideas concrete. "
            "Be honest about uncertainty. Keep responses focused and concise."
        )
