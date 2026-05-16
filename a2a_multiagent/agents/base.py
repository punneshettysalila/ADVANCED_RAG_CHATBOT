"""
Base Agent — all specialized agents extend this.
Wraps HuggingFace InferenceClient for LLM calls.
"""

from __future__ import annotations

import os
import re
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from a2a.protocol import (
    A2AMessage, A2ATask, AgentCard, AgentCapability,
    TextPart, CodePart, ErrorPart, TaskState
)

logger = logging.getLogger(__name__)
load_dotenv()
FALLBACK_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_HF_MODEL = os.getenv("HF_MODEL", FALLBACK_HF_MODEL)
FALLBACK_CHAT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "google/gemma-2-2b-it",
]


class HFInferenceWrapper:
    """Thin async wrapper around HuggingFace InferenceClient."""

    def __init__(self, model: str, token: Optional[str] = None):
        self.model = model
        self.token = token or os.getenv("HF_TOKEN", "")

        # HF tokens should always be standard Hugging Face user tokens.
        if self.token and not self.token.startswith("hf_"):
            raise ValueError(
                "HF_TOKEN must be a Hugging Face token (starts with 'hf_')."
            )

        # Start with direct hf-inference for low latency, then fall back to auto-routing if needed.
        self.client = self._make_client(model=self.model, provider="hf-inference")

    @staticmethod
    def _is_unsupported_model_error(error: Exception) -> bool:
        return "Model not supported by provider hf-inference" in str(error)

    @staticmethod
    def _is_unsupported_conversational_task_error(error: Exception) -> bool:
        msg = str(error)
        return "doesn't support task 'conversational'" in msg or "got: 'conversational'" in msg

    def _make_client(self, model: str, provider: Optional[str] = "hf-inference") -> InferenceClient:
        if provider:
            return InferenceClient(
                model=model,
                provider=provider,
                token=self.token,
            )
        return InferenceClient(
            model=model,
            token=self.token,
        )

    @staticmethod
    def _is_empty_content(content: Optional[str]) -> bool:
        return not content or content.strip() == ""

    def _ordered_fallback_models(self) -> List[str]:
        env_models = [
            m.strip() for m in os.getenv("HF_MODEL_FALLBACKS", "").split(",") if m.strip()
        ]
        candidates = [self.model, FALLBACK_HF_MODEL, *env_models, *FALLBACK_CHAT_MODELS]
        deduped: List[str] = []
        seen = set()
        for model_name in candidates:
            if model_name not in seen:
                seen.add(model_name)
                deduped.append(model_name)
        return deduped

    def _try_alternative_chat_models(
        self,
        full_messages: List[Dict],
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        """Try alternate models through auto provider routing before giving up."""
        for model_name in self._ordered_fallback_models():
            try:
                alt_client = self._make_client(model=model_name, provider=None)
                response = alt_client.chat_completion(
                    messages=full_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                if self._is_empty_content(content):
                    continue

                logger.warning(
                    f"Switched to fallback chat model {model_name} via auto provider routing."
                )
                self.model = model_name
                self.client = alt_client
                return content
            except Exception as alt_error:
                logger.debug(f"Fallback chat model {model_name} failed: {alt_error}")
                continue
        return None

    @staticmethod
    def _to_instruct_prompt(messages: List[Dict]) -> str:
        """Flatten chat messages into an instruction-style prompt for text_generation fallback."""
        lines: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                lines.append(f"System: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
            else:
                lines.append(f"User: {content}")
        lines.append("Assistant:")
        return "\n\n".join(lines)

    def _text_generation_fallback(
        self,
        full_messages: List[Dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Fallback for models/endpoints that reject conversational task routing."""
        prompt = self._to_instruct_prompt(full_messages)
        output = self.client.text_generation(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            return_full_text=False,
        )
        return (output or "").strip()

    @staticmethod
    def _extract_last_user_message(messages: List[Dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return str(msg.get("content")).strip()
        return ""

    def _local_fallback_response(self, messages: List[Dict], error: Exception) -> str:
        """Final safety net when remote inference is unavailable or unsupported."""
        user_text = self._extract_last_user_message(messages)
        lowered = user_text.lower()

        if lowered in {"hi", "hello", "hey", "hii", "hola"}:
            return "Hello. I am here to help. Ask me anything and I will do my best to answer clearly."

        what_is = re.match(r"^\s*what\s+is\s+(.+?)\??\s*$", user_text, flags=re.IGNORECASE)
        if what_is:
            topic = what_is.group(1).strip()
            return (
                f"{topic} is a field or concept studied through structured methods, real-world context, "
                f"and practical application. If you want, I can break {topic} down into basics, examples, "
                f"and why it matters."
            )

        return (
            "I am having trouble reaching the model service right now. "
            "Please try again in a moment."
        )

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Synchronous chat completion."""
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        try:
            response = self.client.chat_completion(
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            
            # Check if response is empty
            if not content or content.strip() == "":
                logger.warning(f"HF model ({self.model}) returned empty response")
                return "I apologize, but I couldn't generate a response. The model returned empty content. Please try rephrasing your question or check if the HuggingFace API token is valid."
            
            return content
        except Exception as e:
            if self._is_unsupported_conversational_task_error(e):
                logger.warning(
                    f"Model {self.model} rejected conversational task. Falling back to text_generation."
                )
                try:
                    content = self._text_generation_fallback(
                        full_messages=full_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                except Exception as fallback_error:
                    logger.warning(
                        f"Model {self.model} rejected text_generation fallback: {fallback_error}. Using local fallback response."
                    )
                    alt_content = self._try_alternative_chat_models(
                        full_messages=full_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if alt_content:
                        return alt_content
                    return self._local_fallback_response(full_messages, fallback_error)
                if not content:
                    logger.warning(f"HF model ({self.model}) returned empty fallback response")
                    alt_content = self._try_alternative_chat_models(
                        full_messages=full_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if alt_content:
                        return alt_content
                    return self._local_fallback_response(full_messages, e)
                return content
            if self._is_unsupported_model_error(e) and self.model != FALLBACK_HF_MODEL:
                logger.warning(
                    f"Model {self.model} unsupported on hf-inference. Falling back to {FALLBACK_HF_MODEL}."
                )
                self.model = FALLBACK_HF_MODEL
                self.client = self._make_client(self.model, provider="hf-inference")
                try:
                    response = self.client.chat_completion(
                        messages=full_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    content = response.choices[0].message.content
                    if self._is_empty_content(content):
                        logger.warning(f"HF model ({self.model}) returned empty response")
                        alt_content = self._try_alternative_chat_models(
                            full_messages=full_messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        if alt_content:
                            return alt_content
                        return "I apologize, but I couldn't generate a response. The model returned empty content. Please try rephrasing your question or check if the HuggingFace API token is valid."
                    return content
                except Exception:
                    alt_content = self._try_alternative_chat_models(
                        full_messages=full_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if alt_content:
                        return alt_content
                    raise
            logger.error(f"HF inference error ({self.model}): {e}")
            raise

    def chat_stream(
        self,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ):
        """Streaming chat — yields text chunks."""
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        try:
            has_content = False
            for chunk in self.client.chat_completion(
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ):
                delta = chunk.choices[0].delta.content
                if delta:
                    has_content = True
                    yield delta
            
            # If no content was streamed, warn and provide fallback
            if not has_content:
                logger.warning(f"HF model ({self.model}) streamed no content")
                yield "I apologize, but I couldn't generate a response. The model returned empty content."
        except Exception as e:
            if self._is_unsupported_conversational_task_error(e):
                logger.warning(
                    f"Model {self.model} rejected conversational stream. Falling back to non-streaming text_generation."
                )
                try:
                    content = self._text_generation_fallback(
                        full_messages=full_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                except Exception as fallback_error:
                    logger.warning(
                        f"Model {self.model} rejected text_generation fallback in stream: {fallback_error}. Using local fallback response."
                    )
                    alt_content = self._try_alternative_chat_models(
                        full_messages=full_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if alt_content:
                        yield alt_content
                        return
                    yield self._local_fallback_response(full_messages, fallback_error)
                    return
                if content:
                    yield content
                else:
                    alt_content = self._try_alternative_chat_models(
                        full_messages=full_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if alt_content:
                        yield alt_content
                        return
                    yield self._local_fallback_response(full_messages, e)
                return
            if self._is_unsupported_model_error(e) and self.model != FALLBACK_HF_MODEL:
                logger.warning(
                    f"Model {self.model} unsupported on hf-inference stream. Falling back to {FALLBACK_HF_MODEL}."
                )
                self.model = FALLBACK_HF_MODEL
                self.client = self._make_client(self.model, provider="hf-inference")
                has_content = False
                try:
                    for chunk in self.client.chat_completion(
                        messages=full_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    ):
                        delta = chunk.choices[0].delta.content
                        if delta:
                            has_content = True
                            yield delta
                    if not has_content:
                        logger.warning(f"HF model ({self.model}) streamed no content")
                        alt_content = self._try_alternative_chat_models(
                            full_messages=full_messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        if alt_content:
                            yield alt_content
                            return
                        yield "I apologize, but I couldn't generate a response. The model returned empty content."
                except Exception:
                    alt_content = self._try_alternative_chat_models(
                        full_messages=full_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if alt_content:
                        yield alt_content
                        return
                    raise
                return
            logger.error(f"HF streaming error ({self.model}): {e}")
            raise


# ─────────────────────────────────────────────
# Base Agent
# ─────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Every specialized agent inherits from this.
    Follows the A2A protocol: receives a task, processes it, emits messages.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        capabilities: List[AgentCapability],
        model: str,
        hf_token: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.model = model
        self.llm = HFInferenceWrapper(model=model, token=hf_token)

        # Register agent card
        self.card = AgentCard(
            agent_id=agent_id,
            name=name,
            description=description,
            capabilities=capabilities,
            model=model,
        )

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Each agent has a specialized system prompt."""
        ...

    def build_hf_messages(self, task: A2ATask) -> List[Dict]:
        """Convert A2A task messages → HuggingFace message format."""
        messages = []
        for msg in task.messages:
            role = "user" if msg.role == "user" else "assistant"
            content = msg.get_text()
            if content:
                messages.append({"role": role, "content": content})
        return messages

    def run(self, task: A2ATask) -> A2AMessage:
        """
        Execute this agent on a task. Returns a single A2AMessage.
        Override for specialized behavior.
        
        A2A Protocol Note: In parallel execution mode, multiple agents
        may process the same task simultaneously. assigned_agent is for
        tracking only and does NOT imply exclusive task ownership.
        """
        task.set_state(TaskState.WORKING)
        task.assigned_agent = self.agent_id  # Tracking only, not routing

        try:
            messages = self.build_hf_messages(task)
            logger.info(f"Agent {self.agent_id} processing {len(messages)} message(s)")
            
            response_text = self.llm.chat(
                messages=messages,
                system_prompt=self.system_prompt,
            )
            
            logger.info(f"Agent {self.agent_id} generated {len(response_text)} characters")
            task.set_state(TaskState.COMPLETED)
            return A2AMessage(
                role="agent",
                agent_id=self.agent_id,
                parts=[TextPart(text=response_text)],
            )
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed: {e}")
            task.set_state(TaskState.FAILED)
            return A2AMessage(
                role="agent",
                agent_id=self.agent_id,
                parts=[ErrorPart(message=str(e))],
            )

    def run_stream(self, task: A2ATask):
        """Streaming version — yields text chunks."""
        task.set_state(TaskState.WORKING)
        task.assigned_agent = self.agent_id  # Tracking only, not routing
        messages = self.build_hf_messages(task)
        try:
            for chunk in self.llm.chat_stream(
                messages=messages,
                system_prompt=self.system_prompt,
            ):
                yield chunk
            task.set_state(TaskState.COMPLETED)
        except Exception as e:
            task.set_state(TaskState.FAILED)
            yield f"\n[Error: {e}]"
