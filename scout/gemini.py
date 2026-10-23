"""Gemini extraction adapter and a router that leaves other stages unchanged."""

import json
import math
import time
from typing import Any, cast

from openai import APIConnectionError, APIStatusError, OpenAI

from scout.contracts import ModelCall
from scout.errors import ConfigurationError, ModelError
from scout.llm import ChatMessages, LLMClient, LLMResponse, Role

GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiExtractionClient:
    """Call only Gemini 3.5 Flash-Lite through Google's documented OpenAI API."""

    def __init__(self, api_key: str):
        if not api_key.strip():
            raise ConfigurationError(
                "Set GEMINI_KEY in the project .env to test Gemini extraction."
            )
        self.calls: list[ModelCall] = []
        self.warnings = [
            "Gemini call cost is recorded as zero for free-tier testing; confirm your "
            "Google project billing tier."
        ]
        self._client = OpenAI(
            api_key=api_key,
            base_url=GEMINI_BASE_URL,
            timeout=60,
            max_retries=0,
        )

    def close(self) -> None:
        self._client.close()

    def chat(
        self,
        stage: str,
        messages: ChatMessages,
        *,
        role: Role = "fast",
        max_tokens: int = 8192,
        require_vision: bool = False,
    ) -> LLMResponse:
        del role, require_vision
        started = time.perf_counter()
        create = cast(Any, self._client.chat.completions.create)
        try:
            response = create(
                model=GEMINI_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except (APIStatusError, APIConnectionError) as error:
            self.calls.append(
                ModelCall(
                    stage=stage,
                    model=GEMINI_MODEL,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    cost_usd=0,
                    ok=False,
                )
            )
            status = getattr(error, "status_code", None)
            detail = f"HTTP {status}" if status else "network error"
            raise ModelError(
                f"Gemini request failed ({detail}); check Google API availability and quota."
            ) from None

        elapsed = int((time.perf_counter() - started) * 1000)
        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice else None
        if not isinstance(text, str):
            text = ""
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        warnings = []
        if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
            prompt_tokens = math.ceil(len(json.dumps(messages, ensure_ascii=False)) / 4)
            warnings.append("Gemini prompt token usage estimated from serialized messages.")
        if not isinstance(completion_tokens, int) or completion_tokens < 0:
            completion_tokens = math.ceil(len(text or "") / 4)
            warnings.append("Gemini completion tokens estimated; hidden reasoning may be omitted.")
        finish = str(choice.finish_reason or "").lower() if choice else ""
        ok = bool(text and text.strip()) and finish not in {
            "length",
            "max_tokens",
            "max_tokens_reached",
        }
        self.calls.append(
            ModelCall(
                stage=stage,
                model=GEMINI_MODEL,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=elapsed,
                cost_usd=0,
                ok=ok,
            )
        )
        if not ok:
            raise ModelError(
                "Gemini returned empty or truncated output; retry with a smaller item limit."
            )
        return LLMResponse(
            text=text,
            model=GEMINI_MODEL,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed,
            cost_usd=0,
            warnings=warnings,
        )


class ExtractionRouter:
    """Use Gemini only for extraction; keep planning and repair on the other client."""

    def __init__(self, other: LLMClient, gemini: LLMClient):
        self.other = other
        self.gemini = gemini
        self.calls: list[ModelCall] = []
        self.warnings = [
            *getattr(other, "warnings", []),
            *getattr(gemini, "warnings", []),
        ]

    def chat(self, stage: str, messages: ChatMessages, **kwargs: Any) -> LLMResponse:
        client = self.gemini if stage == "extract" else self.other
        call_start = len(client.calls)
        warning_start = len(client.warnings)
        try:
            response = client.chat(stage, messages, **kwargs)
        except Exception:
            self.calls.extend(client.calls[call_start:])
            self.warnings.extend(client.warnings[warning_start:])
            raise
        recorded = client.calls[call_start:]
        if recorded:
            self.calls.extend(recorded)
        else:
            self.calls.append(
                ModelCall(
                    stage=stage,
                    model=response.model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    latency_ms=response.latency_ms,
                    cost_usd=response.cost_usd,
                    ok=True,
                )
            )
        self.warnings.extend(client.warnings[warning_start:])
        self.warnings.extend(response.warnings)
        return response

    def close(self) -> None:
        self.gemini.close()
        self.other.close()
