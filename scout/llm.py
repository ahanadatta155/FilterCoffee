"""OpenAI-compatible model adapter; tests intercept HTTP, not SDK behavior."""

import json
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from openai import APIConnectionError, APIStatusError, OpenAI
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from scout.config import Settings
from scout.contracts import ModelCall
from scout.errors import ConfigurationError, ModelError
from scout.models import ModelInfo, price, resolve_models

Role = Literal["fast", "strong", "escalation"]
ChatMessages = list[dict[str, Any]]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cost_usd: float
    warnings: list[str] = field(default_factory=list)


class LLMClient(Protocol):
    calls: list[ModelCall]
    warnings: list[str]

    def chat(
        self,
        stage: str,
        messages: ChatMessages,
        *,
        role: Role = "fast",
        max_tokens: int = 4096,
        require_vision: bool = False,
    ) -> LLMResponse: ...

    def close(self) -> None: ...


def parse_json(text: str) -> dict[str, Any]:
    """Remove model wrappers, then decode the first object without fixing JSON."""
    clean = text.strip()
    while re.match(r"^<(think|thinking)>", clean, re.I):
        clean, count = re.subn(
            r"^<(think|thinking)>.*?</\1>\s*", "", clean, count=1, flags=re.I | re.S
        )
        if not count:
            raise ModelError("Model returned unfinished reasoning instead of JSON.")
    start = re.search(r"[\[{]", clean)
    if start is None or start.group() != "{":
        raise ModelError("Model response did not contain a JSON object.")
    try:
        result, _ = json.JSONDecoder().raw_decode(clean[start.start() :])
    except (ValueError, RecursionError):
        raise ModelError("Model returned malformed JSON; try extraction again.") from None
    return result


def _transient(error: BaseException) -> bool:
    return isinstance(error, APIConnectionError) or (
        isinstance(error, APIStatusError) and (error.status_code == 429 or error.status_code >= 500)
    )


def _friendly(error: APIStatusError | APIConnectionError) -> ModelError:
    status = getattr(error, "status_code", None)
    messages = {
        401: "API key was rejected; check the selected provider and key.",
        402: "budget exhausted; check your provider credits or free-tier allowance.",
        403: "Provider denied access; check model permissions and privacy settings.",
        429: "Provider rate limit reached; wait and try again.",
        404: "Model unavailable; refresh model discovery or choose another free model.",
    }
    message = (
        messages[status]
        if isinstance(status, int) and status in messages
        else "Model request failed; check provider availability."
    )
    if status == 429 and isinstance(error, APIStatusError):
        headers = error.response.headers
        limit = headers.get("x-ratelimit-limit")
        remaining = headers.get("x-ratelimit-remaining")
        retry_after = headers.get("retry-after")
        details = []
        if limit and remaining:
            details.append(f"OpenRouter rate limit remaining {remaining}/{limit}")
        if retry_after and retry_after.isdigit():
            details.append(f"Retry-After: {retry_after}s")
        if details:
            message += " " + "; ".join(details) + "."
    return ModelError(message)


def _format_error(error: APIStatusError) -> bool:
    body = error.body if isinstance(error.body, dict) else {}
    detail = body.get("error", body)
    return (
        error.status_code == 400
        and isinstance(detail, dict)
        and (
            detail.get("param") == "response_format"
            or (
                "response_format" in str(detail.get("message", "")).lower()
                and any(
                    word in str(detail.get("message", "")).lower()
                    for word in ("unsupported", "not supported", "unrecognized")
                )
            )
        )
    )


class ModelClient:
    """Discover once; allow free OpenRouter IDs only, with bounded retries."""

    def __init__(self, settings: Settings, *, sleep: Callable[[float], None] = time.sleep):
        self.settings = settings
        self.calls: list[ModelCall] = []
        self.warnings: list[str] = []
        self._sleep = sleep
        self._blocked = False
        self._sdk = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key.get_secret_value(),
            max_retries=0,
            timeout=30,
        )
        try:
            listing = self._retry(self._sdk.models.list)
            rows = getattr(listing, "data", None)
            if not isinstance(rows, list) or any(not hasattr(row, "model_dump") for row in rows):
                raise ModelError("Provider returned an invalid model catalog; try discovery again.")
            self.models = resolve_models(
                [model.model_dump() for model in rows], settings, self.warnings
            )
        except (APIStatusError, APIConnectionError) as error:
            self.close()
            raise _friendly(error) from None
        except Exception:
            self.close()
            raise
        logger.info("Resolved models: %s", self.models)

    def close(self) -> None:
        self._sdk.close()

    def __enter__(self) -> "ModelClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _retry(self, action: Callable[..., Any]) -> Any:
        return Retrying(
            retry=retry_if_exception(_transient),
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            sleep=self._sleep,
            reraise=True,
        )(action)

    def chat(
        self,
        stage: str,
        messages: ChatMessages,
        *,
        role: Role = "fast",
        max_tokens: int = 4096,
        require_vision: bool = False,
    ) -> LLMResponse:
        if role not in {"fast", "strong", "escalation"}:
            raise ConfigurationError("Unknown model role.")
        model = getattr(self.models, role)
        if role == "escalation" and not self.settings.enable_ultra:
            raise ConfigurationError("Ultra is disabled; set SCOUT_ENABLE_ULTRA=1 to enable it.")
        if model is None:
            raise ConfigurationError("Ultra model is unavailable.")
        if require_vision and not model.supports_vision:
            raise ConfigurationError("Selected model does not support images.")
        if self._blocked:
            raise ModelError("Provider requests stopped after an unexpected charge.")
        use_json = model.supports_json is not False
        fallback_used = False
        # Product intent/presentation have a strict one-physical-attempt contract.
        if stage in {"product_present", "product_intent"}:
            try:
                return self._attempt(stage, messages, model, max_tokens, use_json)
            except (APIStatusError, APIConnectionError) as error:
                raise _friendly(error) from None
        while True:
            try:
                return self._retry(
                    lambda: self._attempt(stage, messages, model, max_tokens, use_json)
                )
            except APIStatusError as error:
                if use_json and _format_error(error):
                    use_json = False
                    self.warnings.append("Provider rejected JSON mode; retried without it.")
                    continue
                missing = (
                    error.status_code == 404 or getattr(error, "code", None) == "model_not_found"
                )
                if missing and not fallback_used:
                    other = (
                        self.models.strong if model.id == self.models.fast.id else self.models.fast
                    )
                    if other.id != model.id and (not require_vision or other.supports_vision):
                        model, fallback_used = other, True
                        use_json = model.supports_json is not False
                        self.warnings.append(f"Unavailable model; fallback to {model.id}.")
                        continue
                raise _friendly(error) from None
            except APIConnectionError as error:
                raise _friendly(error) from None

    def _attempt(
        self, stage: str, messages: ChatMessages, model: ModelInfo, max_tokens: int, use_json: bool
    ) -> LLMResponse:
        started = time.perf_counter()
        create = cast(Any, self._sdk.chat.completions.create)
        kwargs = {"response_format": {"type": "json_object"}} if use_json else {}
        try:
            response = create(
                model=model.id,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                **kwargs,
            )
        except (APIStatusError, APIConnectionError):
            self.calls.append(
                ModelCall(
                    stage=stage,
                    model=model.id,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    cost_usd=0,
                    ok=False,
                )
            )
            self.warnings.append("Failed API attempt has no reported token usage or cost.")
            raise
        elapsed = int((time.perf_counter() - started) * 1000)
        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice else None
        if not isinstance(text, str):
            text = ""
        warnings: list[str] = []
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
            prompt_tokens = math.ceil(len(json.dumps(messages, ensure_ascii=False)) / 4)
            warnings.append(
                "Prompt tokens estimated from serialized input; images are approximate."
            )
        if not isinstance(completion_tokens, int) or completion_tokens < 0:
            completion_tokens = math.ceil(len(text or "") / 4)
            warnings.append("Completion tokens estimated; hidden reasoning is not counted.")
        reported_cost = price(getattr(usage, "cost", None))
        if reported_cost is not None:
            cost = reported_cost
        elif model.input_per_million is not None and model.output_per_million is not None:
            cost = (
                prompt_tokens * model.input_per_million
                + completion_tokens * model.output_per_million
            ) / 1_000_000
            if self.settings.provider == "nebius":
                warnings.append(
                    "Cost estimated using unverified spec rates; verify before the demo."
                )
        else:
            cost = 0.0
            warnings.append("Cost unknown: no provider usage cost or verified pricing available.")
        unexpected_charge = self.settings.provider == "openrouter" and cost > 0
        finish_reason = choice.finish_reason if choice else ""
        valid = bool(text and text.strip()) and finish_reason not in {
            "length",
            "content_filter",
        }
        self.calls.append(
            ModelCall(
                stage=stage,
                model=model.id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=elapsed,
                cost_usd=cost,
                ok=valid and not unexpected_charge,
            )
        )
        self.warnings.extend(warnings)
        if unexpected_charge:
            self._blocked = True
            raise ModelError(
                "Unexpected charge reported for a free model; further requests stopped."
            )
        if not valid:
            raise ModelError(
                "Model returned empty or truncated output; increase max_tokens and retry."
            )
        return LLMResponse(
            text, model.id, prompt_tokens, completion_tokens, elapsed, cost, warnings
        )
