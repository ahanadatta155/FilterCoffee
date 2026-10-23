"""Resolve provider model IDs and capabilities without paid OpenRouter fallbacks."""

import math
from dataclasses import dataclass
from typing import Any

from scout.config import Settings
from scout.errors import ConfigurationError

# Spec estimates only; not a claim of current Nebius billing rates.
NEBIUS_ESTIMATES = {
    "nvidia/nemotron-3.5-lightning": (0.06, 0.24),
    "nvidia/nemotron-3.5-super-vl-120b-a12b": (0.30, 0.90),
    "nvidia/nemotron-3-ultra-550b-a55b": (1.00, 3.00),
}


def price(value: Any) -> float | None:
    """Parse a nonnegative finite price; missing/invalid is not free."""
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ModelInfo:
    id: str
    input_per_million: float | None
    output_per_million: float | None
    supports_json: bool | None
    supports_vision: bool


@dataclass(frozen=True)
class ModelMapping:
    fast: ModelInfo
    strong: ModelInfo
    escalation: ModelInfo | None


def resolve_models(
    rows: list[dict[str, Any]], settings: Settings, warnings: list[str]
) -> ModelMapping:
    """Exact/case-insensitive match first; deterministic family fallback second."""
    requested = (settings.model_fast, settings.model_strong, settings.model_escalation)
    free_only = settings.provider == "openrouter"
    if free_only and any(not model.lower().endswith(":free") for model in requested):
        raise ConfigurationError("OpenRouter requires explicit :free model IDs in this project.")
    available = []
    for row in rows:
        model_id = row.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        pricing = row.get("pricing") or {}
        rates = (price(pricing.get("prompt")), price(pricing.get("completion")))
        if free_only and (
            not model_id.lower().endswith(":free")
            or rates != (0, 0)
            or any(price(value) != 0 for value in pricing.values())
        ):
            continue
        if settings.provider == "nebius":
            rates = NEBIUS_ESTIMATES.get(model_id.lower(), (None, None))
        elif settings.provider == "nvidia":
            rates = (None, None)
        else:
            rates = tuple(None if rate is None else rate * 1_000_000 for rate in rates)
        parameters = row.get("supported_parameters")
        modalities = (row.get("architecture") or {}).get("input_modalities")
        available.append(
            ModelInfo(
                id=model_id,
                input_per_million=rates[0],
                output_per_million=rates[1],
                supports_json=None if parameters is None else "response_format" in parameters,
                supports_vision=(
                    "image" in modalities
                    if modalities is not None
                    else "super-vl" in model_id.lower()
                ),
            )
        )
    available.sort(key=lambda model: model.id.lower())

    def resolve(wanted: str, families: tuple[str, ...], optional: bool = False) -> ModelInfo | None:
        match = next((model for model in available if model.id.lower() == wanted.lower()), None)
        for family in families:
            if match is not None:
                break
            match = next((model for model in available if family in model.id.lower()), None)
        if match is None:
            if optional:
                warnings.append("Optional Ultra model unavailable; escalation is disabled.")
                return None
            qualifier = "free " if free_only else ""
            raise ConfigurationError(f"No suitable {qualifier}model available for {wanted}.")
        if match.id != wanted:
            warnings.append(f"Model {wanted} resolved to {match.id}.")
        return match

    fast = resolve(requested[0], ("lightning", "nano"))
    strong = resolve(requested[1], ("super-vl", "super"))
    escalation = resolve(requested[2], ("ultra",), optional=not settings.enable_ultra)
    assert fast is not None and strong is not None
    return ModelMapping(fast, strong, escalation)
