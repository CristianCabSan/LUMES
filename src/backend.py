"""LLM backend abstractions.

A Backend exposes a single method:
    chat(system, user) -> ChatResponse

The Backend is responsible for all transport details (HTTP, retries, parsing
token counts). The pruner / evaluator only cares about ChatResponse fields.

The ``Backend`` / ``OllamaBackend`` / ``ChatResponse`` / ``build_backend_from_env``
classes are adopted verbatim from the project-provided ``backends.py`` (Ollama
native ``/api/chat`` behind the Ollamus proxy).  ``MockBackend`` is added on top for
offline tests, ``--debug`` runs and the sanity check (no network).
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import requests


@dataclass
class ChatResponse:
    """Normalized response from any LLM backend."""
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    raw: dict | None = None  # backend-native response, kept for debugging


class Backend(ABC):
    """Stateless wrapper around an LLM endpoint."""
    name: str
    model: str

    @abstractmethod
    def chat(self, system: str, user: str) -> ChatResponse: ...


class OllamaBackend(Backend):
    """Direct calls to Ollama's native /api/chat (no SDK)."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.3,
        timeout: int = 180,
        max_retries: int = 2,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, system: str, user: str) -> ChatResponse:
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": self.temperature},
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                dt = time.perf_counter() - t0
                return ChatResponse(
                    text=data.get("message", {}).get("content", ""),
                    input_tokens=data.get("prompt_eval_count", 0),
                    output_tokens=data.get("eval_count", 0),
                    latency_s=dt,
                    raw=data,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                continue
        raise RuntimeError(f"OllamaBackend failed after {self.max_retries + 1} attempts: {last_exc}")


class MockBackend(Backend):
    """Deterministic, network-free backend for tests / ``--debug`` / sanity checks.

    Supply behaviour in one of three ways (checked in this order):

    * ``responder``: ``Callable[[system, user], str]`` — full control.
    * ``rule="curve"``: a built-in heuristic that reads the partial validation
      curve embedded in the user prompt and replies ``DECISION: PRUNE`` when the
      curve is clearly diverging/plateauing above the best-so-far, else CONTINUE.
    * ``scripted``: a list of canned reply strings, returned round-robin.

    Always reports a small, fixed ``latency_s`` and synthetic token counts so the
    trackers and Variant-B savings have something to aggregate.
    """

    name = "mock"

    def __init__(
        self,
        model: str = "mock",
        responder: Optional[Callable[[str, str], str]] = None,
        scripted: Optional[List[str]] = None,
        rule: str = "curve",
        latency_s: float = 0.01,
    ):
        self.model = model
        self.responder = responder
        self.scripted = list(scripted) if scripted else None
        self.rule = rule
        self._latency = latency_s
        self._i = 0
        self.n_calls = 0

    # -- built-in heuristic over the prompt text ---------------------------- #
    _FLOAT = r"-?\d+\.\d+(?:[eE][-+]?\d+)?"

    @classmethod
    def _bracketed_vals(cls, text: str) -> List[float]:
        """Extract the first bracketed list (the partial validation curve)."""
        m = re.search(r"\[([^\]]*)\]", text)
        if not m:
            return []
        return [float(x) for x in re.findall(cls._FLOAT, m.group(1))]

    @classmethod
    def _best_so_far(cls, text: str) -> float | None:
        m = re.search(r"best_final_val_so_far\s*=\s*(" + cls._FLOAT + ")", text)
        return float(m.group(1)) if m else None

    def _curve_rule(self, user: str) -> str:
        # Reads ONLY the validation curve (and best-so-far if present) so extra
        # numbers from L1/L2/L3 context never contaminate the signal.
        vals = self._bracketed_vals(user)
        decision = "CONTINUE"
        if len(vals) >= 2:
            recent = vals[-3:] if len(vals) >= 3 else vals
            slope = recent[-1] - recent[0]
            cur = vals[-1]
            best = self._best_so_far(user)
            eps = 0.005
            if slope > eps:                               # diverging -> prune
                decision = "PRUNE"
            elif slope < -eps:                            # improving -> continue
                decision = "CONTINUE"
            else:                                         # plateau
                if best is not None and cur > best * 1.05:
                    decision = "PRUNE"
        return f"DECISION: {decision}"

    def chat(self, system: str, user: str) -> ChatResponse:
        self.n_calls += 1
        if self.responder is not None:
            text = self.responder(system, user)
        elif self.scripted is not None:
            text = self.scripted[self._i % len(self.scripted)]
            self._i += 1
        else:
            text = self._curve_rule(user)
        return ChatResponse(
            text=text,
            input_tokens=len(system.split()) + len(user.split()),
            output_tokens=len(text.split()),
            latency_s=self._latency,
            raw={"mock": True},
        )


def build_backend_from_env(backend: str, model: str | None, temperature: float, timeout: int) -> Backend:
    """Construct a Backend instance using LLM_* env vars for credentials."""
    if backend == "ollama":
        api_key = os.environ.get("LLM_API_KEY")
        base_url = os.environ.get("LLM_BASE_URL")
        if not base_url:
            raise RuntimeError("LLM_BASE_URL not set in environment.")
        if model is None:
            model = os.environ.get("LLM_MODEL", "llama3.1:8b")
        return OllamaBackend(
            model=model, base_url=base_url, api_key=api_key,
            temperature=temperature, timeout=timeout,
        )
    if backend == "mock":
        return MockBackend(model=model or "mock")
    raise NotImplementedError(f"Backend '{backend}' not implemented in this version.")
