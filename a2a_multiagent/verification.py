"""
Verification agent / trust layer.

Evaluates an answer for consistency, likely factual quality, and
confidence. It can optionally use an HF model, but also provides a
heuristic fallback so the system keeps working without extra services.
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
class VerificationResult:
    approved: bool
    confidence: float
    issues: List[str] = field(default_factory=list)
    corrected_answer: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "confidence": round(self.confidence, 3),
            "issues": list(self.issues),
            "corrected_answer": self.corrected_answer,
            "summary": self.summary,
        }


class VerificationAgent(BaseAgent):
    """Trust layer that checks consistency and confidence."""

    def __init__(self, hf_token: Optional[str] = None):
        super().__init__(
            agent_id="verification-agent",
            name="Sentinel",
            description="Trust layer that verifies, scores, and optionally corrects answers.",
            capabilities=[AgentCapability.GENERAL_QA],
            model=DEFAULT_HF_MODEL,
            hf_token=hf_token,
        )
        self.hf_token = hf_token

    @property
    def system_prompt(self) -> str:
        return (
            "You are a verification agent. Review the answer against the user query and any evidence. "
            "Return only JSON with keys: approved, confidence, issues, corrected_answer, summary. "
            "issues must be an array of short strings."
        )

    def _heuristic_verify(self, query: str, answer: str, evidence: str = "") -> VerificationResult:
        lower = answer.lower()
        issues: List[str] = []
        confidence = 0.72

        if any(phrase in lower for phrase in ["i think", "maybe", "possibly", "not sure"]):
            confidence -= 0.12
            issues.append("Answer contains uncertainty language.")

        if any(phrase in lower for phrase in ["always", "never"]) and "evidence" not in evidence.lower():
            confidence -= 0.08
            issues.append("Absolute claim without support.")

        if len(answer.strip()) < 40:
            confidence -= 0.08
            issues.append("Answer is very short and may be incomplete.")

        if evidence and len(evidence.strip()) > 30:
            confidence += 0.08

        confidence = min(max(confidence, 0.05), 0.98)
        approved = confidence >= 0.55
        summary = "Heuristic trust-layer check completed."
        corrected = answer.strip()
        return VerificationResult(approved=approved, confidence=confidence, issues=issues, corrected_answer=corrected, summary=summary)

    def verify(self, query: str, answer: str, evidence: str = "", memory: str = "") -> VerificationResult:
        if not self.hf_token:
            return self._heuristic_verify(query, answer, evidence=evidence)

        try:
            llm = HFInferenceWrapper(model=self.model, token=self.hf_token)
            prompt = (
                "Verify the answer below. Check consistency, factual correctness, and confidence.\n\n"
                f"User query:\n{query}\n\n"
                f"Evidence / grounding:\n{evidence or 'none'}\n\n"
                f"Relevant memory:\n{memory or 'none'}\n\n"
                f"Draft answer:\n{answer}\n\n"
                "Return only JSON with keys: approved, confidence, issues, corrected_answer, summary."
            )
            raw = llm.chat(messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=260)
            cleaned = raw.strip()
            if "```" in cleaned:
                cleaned = cleaned.split("```", 2)[1]
                cleaned = cleaned.replace("json", "", 1).strip()
            data = json.loads(cleaned)
            issues = data.get("issues", [])
            if not isinstance(issues, list):
                issues = [str(issues)]
            confidence = float(data.get("confidence", 0.5))
            return VerificationResult(
                approved=bool(data.get("approved", True)),
                confidence=min(max(confidence, 0.0), 1.0),
                issues=[str(item) for item in issues],
                corrected_answer=str(data.get("corrected_answer", answer)).strip(),
                summary=str(data.get("summary", "Verification completed.")).strip(),
            )
        except Exception as exc:
            logger.warning(f"Verification agent failed; using heuristic fallback: {exc}")
            return self._heuristic_verify(query, answer, evidence=evidence)
