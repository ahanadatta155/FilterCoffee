"""Load local configuration without networking or revealing credentials."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import SecretStr

from scout.errors import ConfigurationError

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
Provider = Literal["nebius", "openrouter", "nvidia"]
_PROVIDERS: tuple[Provider, ...] = ("nebius", "openrouter", "nvidia")
_KEY_NAMES = {
    "nebius": ("NEBIUS_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_KEY"),
    "nvidia": ("NVIDIA_API_KEY",),
}
_BASE_URL_NAMES = {
    "nebius": "NEBIUS_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
    "nvidia": "NVIDIA_BASE_URL",
}
_DEFAULT_BASE_URLS = {
    "nebius": "https://api.tokenfactory.nebius.com/v1/",
    "openrouter": "https://openrouter.ai/api/v1/",
    "nvidia": "https://integrate.api.nvidia.com/v1",
}
_DEFAULT_MODELS = {
    "nebius": (
        "nvidia/nemotron-3.5-lightning",
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "nvidia/nemotron-3-ultra-550b-a55b",
    ),
    "openrouter": (
        "nvidia/nemotron-3.5-lightning:free",
        "nvidia/nemotron-3.5-super-vl:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
    ),
    "nvidia": (
        "nvidia/nemotron-3.5-lightning",
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "nvidia/nemotron-3-ultra-550b-a55b",
    ),
}


@dataclass(frozen=True)
class Settings:
    """Configured model IDs are requests, not discovered provider capabilities."""

    provider: Provider
    api_key: SecretStr
    base_url: str
    model_fast: str
    model_strong: str
    model_escalation: str
    user_agent: str
    allow_localhost: bool = False
    enable_ultra: bool = False


def load_environment(env_file: str | Path | None = None) -> None:
    """Read only the project's .env (or an explicit file); shell values win."""
    load_dotenv(dotenv_path=ENV_FILE if env_file is None else env_file, override=False)


def _text(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ConfigurationError(f"{name} must not be blank.")
    return value


def _flag(name: str) -> bool:
    value = os.getenv(name, "0").strip()
    if value not in {"0", "1"}:
        raise ConfigurationError(f"{name} must be 0 or 1.")
    return value == "1"


def _base_url(name: str, default: str) -> str:
    value = _text(name, default)
    try:
        parsed = urlsplit(value)
        port = parsed.port
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and "@" not in parsed.netloc
            and not parsed.query
            and not parsed.fragment
            and (port is None or port > 0)
            and not any(char.isspace() for char in value)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ConfigurationError(f"{name} must be an HTTP(S) URL without credentials or query.")
    return value.rstrip("/") + "/"


def _first_key(provider: Provider) -> str:
    for name in _KEY_NAMES[provider]:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _select_provider() -> Provider:
    override = os.getenv("SCOUT_PROVIDER", "auto").strip().lower() or "auto"
    if override != "auto" and override not in _PROVIDERS:
        raise ConfigurationError("SCOUT_PROVIDER must be auto, nebius, openrouter, or nvidia.")
    if override != "auto":
        provider = cast(Provider, override)
        if not _first_key(provider):
            names = " or ".join(_KEY_NAMES[provider])
            raise ConfigurationError(f"SCOUT_PROVIDER={provider} requires {names}.")
        return provider
    for provider in _PROVIDERS:
        if _first_key(provider):
            return provider
    raise ConfigurationError(
        "Set NEBIUS_API_KEY (https://tokenfactory.nebius.com/), "
        "OPENROUTER_KEY (https://openrouter.ai/keys), or "
        "NVIDIA_API_KEY (https://build.nvidia.com/)."
    )


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Select a provider without contacting it; shell values override .env."""
    load_environment(env_file)
    provider = _select_provider()
    api_key = _first_key(provider)
    defaults = _DEFAULT_MODELS[provider]
    return Settings(
        provider=provider,
        api_key=SecretStr(api_key),
        base_url=_base_url(_BASE_URL_NAMES[provider], _DEFAULT_BASE_URLS[provider]),
        model_fast=_text("SCOUT_MODEL_FAST", defaults[0]),
        model_strong=_text("SCOUT_MODEL_STRONG", defaults[1]),
        model_escalation=_text("SCOUT_MODEL_ESCALATION", defaults[2]),
        user_agent=_text("SCOUT_USER_AGENT", "NemotronScout/0.1"),
        allow_localhost=_flag("SCOUT_ALLOW_LOCALHOST"),
        enable_ultra=_flag("SCOUT_ENABLE_ULTRA"),
    )
