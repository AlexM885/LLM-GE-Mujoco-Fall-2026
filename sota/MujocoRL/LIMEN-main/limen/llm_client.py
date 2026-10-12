"""OpenAI-compatible LLM client for LIMEN.

Works with any provider that exposes the OpenAI Chat Completions API:
OpenAI, OpenRouter, Together, Groq, Fireworks, Azure OpenAI, vLLM, Ollama,
LM Studio, etc. Set `api_base` to the provider endpoint and `api_key` to
the credential.

Wraps `client.chat.completions.create(...)` to generate MDP interface code.
Takes prompt dicts from PromptBuilder, calls the model, extracts code.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai

from limen.config import LLMConfig
from limen.prompts import extract_code

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reasoning model detection (uses max_completion_tokens + reasoning_effort)
# ---------------------------------------------------------------------------

_REASONING_MODEL_PREFIXES = (
    "o1", "o3", "o4",
    "gpt-5", "gpt-oss",
)

_OPENAI_API_PREFIXES = (
    "https://api.openai.com",
    "https://eu.api.openai.com",
    "https://apac.api.openai.com",
)


def _is_reasoning_model(model_name: str, api_base: str) -> bool:
    """Detect OpenAI reasoning models that need special parameters."""
    api_base_lower = (api_base or "").lower()
    is_openai_native = (
        any(api_base_lower.startswith(p) for p in _OPENAI_API_PREFIXES)
        or ".openai.azure.com" in api_base_lower
    )
    if not is_openai_native:
        return False
    name = model_name.lower()
    return any(name.startswith(p + "-") or name == p for p in _REASONING_MODEL_PREFIXES)


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Result from a single LLM call."""
    text: str
    code: Optional[str]
    input_tokens: int
    output_tokens: int
    model: str


# ---------------------------------------------------------------------------
# Single-model client
# ---------------------------------------------------------------------------


class LLMClient:
    """OpenAI-compatible LLM client for generating MDP interface code.

    Usage:
        client = LLMClient(config)
        prompt = prompt_builder.build_prompt(...)
        response = client.generate(prompt)
        if response.code:
            ...
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._local = threading.local()

        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY in your environment "
                "or pass api_key in your LLM config."
            )
        self._api_key = api_key
        self._api_base = config.api_base
        self._model = config.model_name
        self._is_reasoning = _is_reasoning_model(self._model, self._api_base)

        logger.info(
            "LLMClient initialized: model=%s api_base=%s%s",
            self._model,
            self._api_base,
            " (reasoning)" if self._is_reasoning else "",
        )

    @property
    def client(self):
        """Thread-local OpenAI client (created lazily)."""
        c = getattr(self._local, "client", None)
        if c is None:
            c = openai.OpenAI(
                api_key=self._api_key,
                base_url=self._api_base,
                timeout=self.config.timeout,
                max_retries=0,  # we handle retries ourselves
            )
            self._local.client = c
        return c

    def _build_params(self, system_msg: str, user_msg: str) -> Dict[str, Any]:
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user_msg})

        params: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }

        if self._is_reasoning:
            params["max_completion_tokens"] = self.config.max_tokens
            if self.config.reasoning_effort is not None:
                params["reasoning_effort"] = self.config.reasoning_effort
        else:
            params["max_tokens"] = self.config.max_tokens
            if self.config.temperature is not None:
                params["temperature"] = self.config.temperature
            if self.config.top_p is not None:
                params["top_p"] = self.config.top_p

        return params

    def generate(self, prompt: Dict[str, str]) -> LLMResponse:
        """Call the LLM with a prompt dict from PromptBuilder.

        Args:
            prompt: {"system": str, "user": str} as returned by
                    PromptBuilder.build_prompt().

        Returns:
            LLMResponse with the raw text, extracted code (if any),
            and token usage.
        """
        params = self._build_params(prompt.get("system", ""), prompt["user"])

        last_error: Optional[Exception] = None
        delay = self.config.retry_delay

        for attempt in range(1, self.config.retries + 1):
            try:
                response = self.client.chat.completions.create(**params)
                text = response.choices[0].message.content or ""
                usage = response.usage
                code = extract_code(text)

                return LLMResponse(
                    text=text,
                    code=code,
                    input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                    model=self._model,
                )

            except (openai.RateLimitError, openai.APITimeoutError) as e:
                last_error = e
                if attempt < self.config.retries:
                    logger.warning(
                        "Rate-limited/timeout (attempt %d/%d), retrying in %.1fs...",
                        attempt, self.config.retries, delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                logger.error("LLM call failed after retries: %s", e)
                raise

            except openai.APIStatusError as e:
                last_error = e
                # 5xx -> retry, 4xx -> raise
                if 500 <= getattr(e, "status_code", 0) < 600 and attempt < self.config.retries:
                    logger.warning(
                        "Server error %s (attempt %d/%d), retrying in %.1fs...",
                        e.status_code, attempt, self.config.retries, delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                logger.error("OpenAI API error: %s", e)
                raise

            except Exception as e:
                last_error = e
                if attempt < self.config.retries:
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt, self.config.retries, e, delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

        raise RuntimeError(
            f"LLM call failed after {self.config.retries} attempts: {last_error}"
        )

    def generate_code(self, prompt: Dict[str, str]) -> Optional[str]:
        """Convenience method: generate and return just the extracted code."""
        return self.generate(prompt).code


# ---------------------------------------------------------------------------
# LLM-GE server client
# ---------------------------------------------------------------------------


# Llama 3.x instruct chat layout. The tokenizer adds <|begin_of_text|> itself.
_LLAMA3_TURN = "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
_LLAMA3_ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def format_chat_prompt(system_msg: str, user_msg: str, template: str) -> str:
    """Turn a system and a user message into the single text prompt the server takes."""
    if template == "llama3":
        text = _LLAMA3_TURN.format(role="system", content=system_msg) if system_msg else ""
        return text + _LLAMA3_TURN.format(role="user", content=user_msg) + _LLAMA3_ASSISTANT
    if template == "none":
        return f"{system_msg}\n\n{user_msg}" if system_msg else user_msg
    raise ValueError(f"Unknown chat_template: {template!r}")


class LLMGEServerClient:
    """Client for the LLM-GE model server (server.py at the repository root).

    The server takes a raw prompt on POST /generate and returns
    {"generated_text": ...}. Its host is written to hostname.log by server.sh
    and its port is LLMGE_PORT, exactly as LLM-GE itself finds it.
    """

    # The server job is restarted every few hours; wait this long for it to come back
    _SERVER_WAIT_SECONDS = 3600
    _POLL_SECONDS = 30

    def __init__(self, config: LLMConfig):
        self.config = config
        self._model = config.model_name
        # The server answers a batch at a time; more requests than that only queue up
        self._slots = threading.BoundedSemaphore(max(config.max_concurrent_requests, 1))
        if config.hostname_file:
            self._hostname_file = Path(config.hostname_file)
        else:
            # limen/ -> LIMEN-main/ -> MujocoRL/ -> sota/ -> repository root
            self._hostname_file = Path(__file__).resolve().parents[4] / "hostname.log"
        format_chat_prompt("", "", config.chat_template)  # fail early on a bad template name
        logger.info(
            "LLMGEServerClient initialized: model=%s url=%s chat_template=%s",
            self._model, self._url(), config.chat_template,
        )

    def _url(self) -> str:
        base = self.config.api_base
        if not base.startswith("http"):
            host = os.environ.get("SERVER_HOSTNAME", "localhost")
            if self._hostname_file.exists():
                host = self._hostname_file.read_text().splitlines()[0].strip() or host
            base = f"http://{host}:{os.environ.get('LLMGE_PORT', '8169')}"
        return base.rstrip("/") + "/generate"

    def generate(self, prompt: Dict[str, str]) -> LLMResponse:
        """Call the server with a prompt dict from PromptBuilder."""
        text_prompt = format_chat_prompt(
            prompt.get("system", ""), prompt["user"], self.config.chat_template
        )
        payload: Dict[str, Any] = {
            "prompt": text_prompt,
            "max_new_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        body = json.dumps(payload).encode()

        deadline = time.time() + self._SERVER_WAIT_SECONDS
        while True:
            # Re-read the host every attempt: it changes when the server job restarts
            url = self._url()
            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            try:
                with self._slots:
                    with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                        text = json.loads(response.read()).get("generated_text", "")
                return LLMResponse(
                    text=text,
                    code=extract_code(text),
                    input_tokens=0,
                    output_tokens=0,
                    model=self._model,
                )
            except urllib.error.HTTPError as e:
                # The server answered with an error; asking again would not help
                raise RuntimeError(f"LLM-GE server at {url} returned HTTP {e.code}: {e.reason}") from e
            except urllib.error.URLError as e:
                # Nothing is listening (yet): the request was never accepted, so it is safe to retry
                if time.time() >= deadline:
                    raise RuntimeError(f"LLM-GE server at {url} not reachable: {e}") from e
                logger.warning(
                    "LLM-GE server at %s not reachable (%s), retrying in %ds...",
                    url, e, self._POLL_SECONDS,
                )
                time.sleep(self._POLL_SECONDS)
            except TimeoutError as e:
                # The server accepted the request and is still generating. It cannot be
                # cancelled, so sending it again would only pile more work onto its queue.
                raise RuntimeError(
                    f"LLM-GE server at {url} accepted the request but did not answer within "
                    f"{self.config.timeout}s (llm.timeout)"
                ) from e


# ---------------------------------------------------------------------------
# Weighted ensemble
# ---------------------------------------------------------------------------


class LLMEnsemble:
    """Weighted ensemble of LLM clients with random model selection.

    When config.models is empty, behaves as a single-model wrapper around
    LLMClient. When populated, selects a model per generate() call using
    weighted random sampling.

    Usage:
        ensemble = LLMEnsemble(config)
        response = ensemble.generate(prompt)  # same interface as LLMClient
    """

    def __init__(self, config: LLMConfig):
        self.config = config

        if config.provider == "llmge":
            self._clients = [LLMGEServerClient(config)]
            self._weights = [1.0]
            self._names = [config.model_name]
        elif not config.models:
            self._clients = [LLMClient(config)]
            self._weights = [1.0]
            self._names = [config.model_name]
        else:
            self._clients = []
            self._weights = []
            self._names = []

            for model_cfg in config.models:
                if model_cfg.weight < 0:
                    raise ValueError(
                        f"Model weight must be non-negative (got {model_cfg.weight} "
                        f"for {model_cfg.name})"
                    )
                merged = LLMConfig(
                    api_base=model_cfg.api_base or config.api_base,
                    api_key=model_cfg.api_key or config.api_key,
                    model_name=model_cfg.name,
                    temperature=(
                        model_cfg.temperature
                        if model_cfg.temperature is not None
                        else config.temperature
                    ),
                    top_p=(
                        model_cfg.top_p
                        if model_cfg.top_p is not None
                        else config.top_p
                    ),
                    max_tokens=(
                        model_cfg.max_tokens
                        if model_cfg.max_tokens is not None
                        else config.max_tokens
                    ),
                    timeout=(
                        model_cfg.timeout
                        if model_cfg.timeout is not None
                        else config.timeout
                    ),
                    retries=config.retries,
                    retry_delay=config.retry_delay,
                    reasoning_effort=(
                        model_cfg.reasoning_effort
                        if model_cfg.reasoning_effort is not None
                        else config.reasoning_effort
                    ),
                )
                self._clients.append(LLMClient(merged))
                self._weights.append(model_cfg.weight)
                self._names.append(model_cfg.name)

            total = sum(self._weights)
            if total <= 0:
                raise ValueError("LLM ensemble weights must sum to a positive value")
            self._weights = [w / total for w in self._weights]

        logger.info(
            "LLMEnsemble initialized: %s",
            ", ".join(
                f"{n} (w={w:.2f})" for n, w in zip(self._names, self._weights)
            ),
        )

    def generate(self, prompt: Dict[str, str]) -> LLMResponse:
        """Generate using a weighted-random selected model."""
        selected = random.choices(self._clients, weights=self._weights, k=1)[0]
        return selected.generate(prompt)
