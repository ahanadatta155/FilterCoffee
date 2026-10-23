"""End-to-end page extraction with evidence validation and one repair pass."""

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import SecretStr, ValidationError

from scout.cache import Cache, digest
from scout.clean import CleanPage, clean_page
from scout.config import Settings, load_settings
from scout.contracts import ExtractionPlan, ExtractResult, FieldSpec, ModelCall, RunMeta
from scout.errors import ScoutError
from scout.fetch import Fetcher
from scout.gemini import ExtractionRouter, GeminiExtractionClient
from scout.llm import ChatMessages, LLMClient, ModelClient, Role, parse_json
from scout.represent import build_representation, find_candidate_groups
from scout.security import resolve_host, validate_url
from scout.validate import validate_items

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
LIST_WORDS = re.compile(r"\b(each|all|list|products|rows|every)\b", re.I)


@dataclass(frozen=True)
class RunOptions:
    ignore_robots: bool = False
    allow_localhost: bool = False
    force_refresh: bool = False
    max_items: int = 100
    force_browser: bool = False
    routing: str = "gemini"


def _schema(value: dict[str, Any] | None) -> ExtractionPlan | None:
    if value is None:
        return None
    if "fields" in value:
        fields = value["fields"]
    else:
        properties = value.get("properties")
        if not isinstance(properties, dict):
            raise ValueError("Schema must contain fields or JSON Schema properties.")
        required = set(value.get("required", []))
        types = {
            "string": "string",
            "number": "number",
            "integer": "integer",
            "boolean": "boolean",
            "array": "list[string]",
        }
        fields = []
        for name, definition in properties.items():
            kind = definition.get("type")
            if kind == "array" and (definition.get("items") or {}).get("type") != "string":
                raise ValueError(f"Only arrays of strings are supported: {name}.")
            if kind not in types:
                raise ValueError(f"Unsupported schema type for {name}.")
            fields.append(
                {
                    "name": name,
                    "type": types[kind],
                    "description": definition.get("description", name),
                    "required": name in required,
                }
            )
    return ExtractionPlan.model_validate({"mode": "single", "fields": fields})


def _fallback_plan(instruction: str, schema: dict[str, Any] | None) -> ExtractionPlan:
    try:
        provided = _schema(schema)
    except (ValueError, ValidationError):
        provided = None
    if provided:
        return provided
    name = re.sub(r"[^a-z0-9]+", "_", instruction.lower()).strip("_")[:35] or "result"
    if name[0].isdigit():
        name = f"field_{name}"
    return ExtractionPlan(
        mode="list" if LIST_WORDS.search(instruction) else "single",
        fields=[FieldSpec(name=name, type="string", description=instruction)],
    )


def _plan_prompt(instruction: str, page: CleanPage, representation: str) -> str:
    signals = {
        "title": page.title,
        "meta": page.meta_description,
        "jsonld_keys": sorted(
            {key for value in page.jsonld if isinstance(value, dict) for key in value}
        ),
        "table_headers": [table.headers for table in page.tables],
        "candidate_groups": [
            {"count": len(group.items), "sample": [item.text[:300] for item in group.items[:2]]}
            for group in find_candidate_groups(page)
        ],
        "representation_preview": representation[:2400],
    }
    system = (PROMPTS / "plan.md").read_text(encoding="utf-8")
    return json.dumps(
        {"system": system, "instruction": instruction, "page_signals": signals}, ensure_ascii=False
    )


def _messages(system: str, payload: str) -> ChatMessages:
    return cast(
        ChatMessages,
        [{"role": "system", "content": system}, {"role": "user", "content": payload}],
    )


def _mode(plan: ExtractionPlan, instruction: str, groups: int) -> ExtractionPlan:
    mode = "list" if LIST_WORDS.search(instruction) or groups >= 3 else plan.mode
    return plan.model_copy(update={"mode": mode})


def _merge_repair(original: Any, repaired: Any) -> Any:
    """Merge correction fields by item index while retaining omitted valid values."""
    if not isinstance(original, dict) or not isinstance(repaired, dict):
        return repaired
    old_items, new_items = original.get("items"), repaired.get("items")
    if not isinstance(old_items, list) or not isinstance(new_items, list):
        return repaired
    merged = []
    for index, old in enumerate(old_items):
        new = new_items[index] if index < len(new_items) else {}
        if not isinstance(old, dict) or not isinstance(new, dict):
            merged.append(new if new else old)
            continue
        old_data, new_data = old.get("data", {}), new.get("data", {})
        data = {
            **(old_data if isinstance(old_data, dict) else {}),
            **(new_data if isinstance(new_data, dict) else {}),
        }
        evidence = {}
        for item in (old, new):
            for entry in item.get("evidence", []) if isinstance(item.get("evidence"), list) else []:
                if isinstance(entry, dict) and isinstance(entry.get("field"), str):
                    evidence[entry["field"]] = entry
        merged.append({**old, **new, "data": data, "evidence": list(evidence.values())})
    return {**original, **repaired, "items": merged}


def _totals(calls: list[ModelCall]) -> tuple[int, int, float, int]:
    prompt = sum(call.prompt_tokens for call in calls)
    completion = sum(call.completion_tokens for call in calls)
    cost = sum(call.cost_usd for call in calls)
    latency = sum(call.latency_ms for call in calls)
    return prompt, completion, cost, latency


def run(
    url: str,
    instruction: str,
    schema: dict[str, Any] | None = None,
    options: RunOptions | None = None,
    *,
    fetcher: Any = None,
    llm: LLMClient | None = None,
    cache: Cache | None = None,
    cache_path: str | Path = "data/cache.db",
    settings: Settings | None = None,
) -> ExtractResult:
    """Fetch, plan, extract, verify, optionally repair once, and cache a result.

    Inject `fetcher` and `llm` for offline runs. A cache path under data/ is used
    by default; force_refresh skips result and page cache reads but writes results.
    """
    started = time.perf_counter()
    options = options or RunOptions()
    settings_error = None
    if settings is None:
        try:
            settings = load_settings()
        except ScoutError as error:
            settings_error = error
            settings = Settings(
                provider="openrouter",
                api_key=SecretStr(""),
                base_url="https://openrouter.ai/api/v1/",
                model_fast="",
                model_strong="",
                model_escalation="",
                user_agent="NemotronScout/0.1",
            )
    cache = cache or Cache(cache_path)
    owned_fetcher = fetcher is None
    fetcher = fetcher or Fetcher(
        user_agent=settings.user_agent,
        allow_localhost=options.allow_localhost or settings.allow_localhost,
    )
    owned_llm = llm is None
    if llm is None and settings_error is None:
        try:
            nemotron = ModelClient(settings)
            if options.routing == "gemini":
                try:
                    gemini = GeminiExtractionClient(os.getenv("GEMINI_KEY", ""))
                except ScoutError:
                    nemotron.close()
                    raise
                llm = ExtractionRouter(nemotron, gemini)
            else:
                llm = nemotron
        except ScoutError as error:
            settings_error = error
    if llm is None:
        llm = _UnavailableLLM()
    run_id = uuid.uuid4().hex
    call_start = len(getattr(llm, "calls", []))
    warnings: list[str] = []
    plan = _fallback_plan(instruction, schema)
    page: CleanPage | None = None
    status_code = None
    tier = "http"
    items = []
    result_key = digest(
        json.dumps(
            {
                "url": url,
                "instruction": instruction,
                "schema": schema,
                "provider": settings.provider,
                "models": [settings.model_fast, settings.model_strong, settings.model_escalation],
                "options": {
                    "allow_localhost": options.allow_localhost,
                    "ignore_robots": options.ignore_robots,
                    "max_items": options.max_items,
                    "routing": options.routing,
                    "gemini_key_configured": bool(os.getenv("GEMINI_KEY", "").strip())
                    if options.routing == "gemini"
                    else False,
                    "version": 2,
                },
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    if settings_error:
        warnings.append(str(settings_error))
    try:
        if settings_error:
            raise _PipelineStop
        validate_url(
            url,
            allow_localhost=options.allow_localhost or settings.allow_localhost,
            resolver=getattr(fetcher, "resolver", resolve_host),
        )
        if not options.force_refresh:
            cached = cache.get_result(result_key)
            if cached is not None:
                restored = ExtractResult.model_validate(cached)
                return restored.model_copy(
                    update={"meta": restored.meta.model_copy(update={"cache_hit": True})}
                )

        cached_page = None if options.force_refresh else cache.get_page(url)
        if cached_page:
            html, tier = cached_page
            final_url = url
        else:
            fetched = fetcher.fetch(url, ignore_robots=options.ignore_robots)
            html, final_url = fetched.html, fetched.url
            status_code, tier = fetched.status_code, fetched.tier
            warnings.extend(fetched.warnings)
            cache.set_page(url, html, tier)
        page = clean_page(html, final_url)
        warnings.extend(page.warnings)
        if page.blocked:
            warnings.append(
                "Page appears blocked; CAPTCHA or bot-bypass attempts are not supported."
            )
        elif options.force_browser or page.needs_render:
            warnings.append(
                "Page needs browser rendering; Playwright fallback is not implemented yet."
            )
        else:
            representation = build_representation(page, max_items=options.max_items)
            groups = find_candidate_groups(page)
            if schema is None:
                plan_messages = _messages(
                    (PROMPTS / "plan.md").read_text(encoding="utf-8"),
                    _plan_prompt(instruction, page, representation),
                )
                planned = None
                for attempt in range(2):
                    try:
                        plan_response = llm.chat(
                            "plan",
                            plan_messages,
                            role="strong" if len(instruction) > 160 or len(groups) > 2 else "fast",
                            max_tokens=2048,
                        )
                        planned = ExtractionPlan.model_validate(parse_json(plan_response.text))
                        break
                    except (ScoutError, ValidationError, ValueError) as error:
                        warnings.append(f"Schema planning attempt {attempt + 1} failed: {error}")
                if planned is None:
                    plan = _mode(plan, instruction, len(groups))
                    raise _PipelineStop
                plan = _mode(planned, instruction, len(groups))
            else:
                supplied_plan = _schema(schema)
                if supplied_plan is None:
                    raise ValueError(
                        "The supplied schema could not be converted to an extraction plan."
                    )
                plan = _mode(supplied_plan, instruction, len(groups))
            system = (PROMPTS / "extract.md").read_text(encoding="utf-8")
            payload = json.dumps(
                {"instruction": instruction, "schema": plan.model_dump(), "page": representation},
                ensure_ascii=False,
            )
            raw = None
            for attempt in range(2):
                try:
                    role = "strong" if options.routing == "strong" else "fast"
                    if options.routing == "fast":
                        role = "fast"
                    if attempt == 1 and options.routing == "auto":
                        role = "strong"
                    if options.routing == "auto" and (
                        len(groups) > 2 or len(plan.fields) > 6 or tier != "http"
                    ):
                        role = "strong"
                    response = llm.chat(
                        "extract",
                        _messages(system, payload),
                        role=role,
                        max_tokens=8192 if plan.mode == "list" else 4096,
                    )
                    raw = parse_json(response.text)
                    break
                except (ScoutError, ValueError) as error:
                    warnings.append(f"Extraction attempt {attempt + 1} failed: {error}")
                    if attempt == 1:
                        raise _PipelineStop
            validation = validate_items(raw, plan, page.source_text)
            items = validation.items
            warnings.extend(validation.errors)
            average = sum(item.confidence for item in items) / len(items) if items else 0
            needs_repair = any(
                "required field" in error
                or "invalid type" in error
                or "does not support" in error
                or "evidence" in error
                for error in validation.errors
            ) or (items and average < 0.6)
            if needs_repair:
                try:
                    repair_system = (PROMPTS / "repair.md").read_text(encoding="utf-8")
                    repair_payload = json.dumps(
                        {
                            "schema": plan.model_dump(),
                            "items": raw,
                            "errors": validation.errors,
                            "source": page.source_text[:4000],
                        },
                        ensure_ascii=False,
                    )
                    repaired_response = llm.chat(
                        "repair",
                        _messages(repair_system, repair_payload),
                        role="strong",
                        max_tokens=8192,
                    )
                    repaired = _merge_repair(raw, parse_json(repaired_response.text))
                    repaired_validation = validate_items(repaired, plan, page.source_text)
                    items = repaired_validation.items
                    warnings.extend(repaired_validation.errors)
                except (ScoutError, ValueError) as error:
                    warnings.append(f"Repair failed; keeping validated values: {error}")
    except _PipelineStop:
        pass
    except ScoutError as error:
        warnings.append(str(error))
    except Exception as error:
        warnings.append(
            f"Run could not complete ({type(error).__name__}); check the URL and configuration."
        )
    finally:
        if owned_fetcher:
            fetcher.close()
        if owned_llm:
            llm.close()
    final_url = page.url if page else url
    try:
        call_slice = list(getattr(llm, "calls", []))[call_start:]
        model_calls = [
            call if isinstance(call, ModelCall) else ModelCall.model_validate(call)
            for call in call_slice
        ]
    except (AttributeError, IndexError, TypeError):
        model_calls = []
    warnings.extend(getattr(llm, "warnings", []))
    prompt_tokens, completion_tokens, cost, model_latency = _totals(model_calls)
    meta = RunMeta(
        run_id=run_id,
        url=final_url,
        fetch_tier=tier,
        status_code=status_code,
        page_title=page.title if page else None,
        model_calls=model_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost_usd=cost,
        latency_ms=max(model_latency, int((time.perf_counter() - started) * 1000)),
        warnings=list(dict.fromkeys(warnings)),
    )
    result = ExtractResult(
        url=final_url,
        instruction=instruction,
        plan=plan,
        items=items,
        meta=meta,
        raw_text_preview=page.main_text[:1000] if page else "",
    )
    cache.set_result(
        result_key if "result_key" in locals() else digest(url + instruction),
        result.model_dump(mode="json"),
    )
    return result


class _PipelineStop(Exception):
    """Exit remaining extraction stages while preserving a partial result."""


class _UnavailableLLM:
    """Allow configuration failures to return the promised partial result."""

    def __init__(self) -> None:
        self.calls: list[ModelCall] = []
        self.warnings: list[str] = []

    def close(self) -> None:
        return None

    def chat(
        self,
        stage: str,
        messages: ChatMessages,
        *,
        role: Role = "fast",
        max_tokens: int = 4096,
        require_vision: bool = False,
    ):
        raise ScoutError("Model provider is not configured.")
