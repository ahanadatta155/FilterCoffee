"""Run a live, evidence-backed extraction from a public web page."""

import argparse
import sys
import time
import unicodedata

from scout import extract
from scout.config import load_settings
from scout.contracts import ExtractedItem, FieldSpec
from scout.discovery import DiscoveryResult, resolve_request
from scout.errors import ScoutError
from scout.limits import get_openrouter_key_status
from scout.pipeline import RunOptions

DEFAULT_URL = "https://books.toscrape.com/"
DEFAULT_INSTRUCTION = "Extract each book's title, price, rating, and availability."


def _display(value: object) -> str:
    return "".join(char for char in str(value) if unicodedata.category(char) not in {"Cc", "Cf"})


def _table(items: list[ExtractedItem], fields: list[FieldSpec]) -> None:
    columns = [field.name for field in fields]
    widths = [
        min(42, max(12, len(name), *(len(_display(item.data.get(name) or "-")) for item in items)))
        for name in columns
    ]
    print(
        " | ".join(name.upper().ljust(width) for name, width in zip(columns, widths, strict=True))
    )
    print("-+-".join("-" * width for width in widths))
    for item in items:
        cells = []
        for name, width in zip(columns, widths, strict=True):
            value = item.data.get(name)
            text = "-" if value is None else _display(value)
            cells.append(text[:width].ljust(width))
        print(" | ".join(cells))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Page to scrape; prompts if omitted")
    parser.add_argument(
        "--request",
        "--instruction",
        dest="request",
        help="What to find and extract; interpreted when --url is omitted",
    )
    parser.add_argument(
        "--max-items", type=int, default=20, help="Maximum rows to ask the model for"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Fetch and extract again instead of using cache"
    )
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument("--browser", action="store_true", help="Force Chromium rendering")
    browser.add_argument(
        "--no-browser", dest="use_browser", action="store_false", help="Disable browser fallback"
    )
    parser.set_defaults(use_browser=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        settings = load_settings()
        if settings.provider != "openrouter":
            print(
                "This runner is intentionally free-only. Set SCOUT_PROVIDER=openrouter in .env.",
                file=sys.stderr,
            )
            return 1
        print("Checking OpenRouter quota and preparing the request...", flush=True)
        try:
            status = get_openrouter_key_status(
                settings.api_key.get_secret_value(), settings.base_url
            )
            for line in status.summary():
                print(f"[debug] OpenRouter: {line}", flush=True)
        except (RuntimeError, ValueError) as error:
            print(f"[debug] OpenRouter quota check unavailable: {error}", flush=True)
        discovery: DiscoveryResult | None = None
        if args.url:
            url = args.url
            instruction = (
                args.request
                or input(f"What data should I extract? [{DEFAULT_INSTRUCTION}]: ").strip()
                or DEFAULT_INSTRUCTION
            )
        else:
            request = args.request or input("What would you like me to find? ").strip()
            if request:
                discovery = resolve_request(request, settings=settings)
                url, instruction = discovery.url, discovery.instruction
                print(f"[request] Mode: {_display(discovery.route)}")
                if discovery.query:
                    print(f"[search] Query: {_display(discovery.query)}")
                    print(f"[search] Results considered: {discovery.search_result_count}")
                print(
                    f"[search] Selected page: {_display(discovery.title or discovery.url)} "
                    f"| {_display(discovery.url)}"
                )
                for stage, response in (
                    ("discover", discovery.intent_response),
                    ("discover_select", discovery.selection_response),
                ):
                    if response is not None:
                        print(
                            f"[debug] {stage} | {_display(response.model)} | "
                            f"{response.prompt_tokens} input / {response.completion_tokens} output "
                            f"tokens | {response.latency_ms} ms"
                        )
                if discovery.query:
                    print(f"[debug] Web search: {discovery.search_latency_ms} ms")
            else:
                url, instruction = DEFAULT_URL, DEFAULT_INSTRUCTION
                print(f"Using sample page: {url}")
        print("Fetching and extracting one page...", flush=True)
        result = extract(
            url,
            instruction,
            schema=None,
            options=RunOptions(
                max_items=args.max_items,
                force_refresh=args.refresh,
                force_browser=args.browser,
                use_browser=args.use_browser,
                routing="gemini",
            ),
            settings=settings,
        )
    except (ScoutError, ValueError) as error:
        print(f"Could not start Nemotron Scout: {error}", file=sys.stderr)
        return 1

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="backslashreplace")
    elapsed = time.perf_counter() - started
    extractor = (
        "Nemotron Super VL (vision)"
        if result.meta.fetch_tier == "browser+vision"
        else "Gemini 3.5 Flash-Lite"
    )
    print(f"Nemotron Scout | {_display(result.meta.url)}")
    print(
        f"Fetch: {result.meta.fetch_tier} | Items: {len(result.items)} "
        f"| Cache hit: {result.meta.cache_hit} | Extractor: {extractor}"
    )
    print(
        f"Extraction model calls: {len(result.meta.model_calls)} "
        f"| Recorded extraction cost: ${result.meta.estimated_cost_usd:.4f}"
    )
    print(
        f"[debug] Elapsed: {elapsed:.2f} seconds | Extraction tokens: "
        f"{result.meta.prompt_tokens} input / {result.meta.completion_tokens} output"
    )
    for call in result.meta.model_calls:
        print(
            f"[debug] {call.stage} | {call.model} | {call.prompt_tokens} input / "
            f"{call.completion_tokens} output tokens | {call.latency_ms} ms | "
            f"USD {call.cost_usd:.4f} | {'ok' if call.ok else 'failed/truncated'}"
        )
    if result.items:
        print()
        _table(result.items, result.plan.fields)
        print("\nEvidence quotes:")
        for index, item in enumerate(result.items, start=1):
            for evidence in item.evidence:
                if evidence.quote:
                    print(f'  {index}. {evidence.field}: "{_display(evidence.quote)}"')
    for warning in result.meta.warnings:
        print(f"Warning: {_display(warning)}")
    if not result.items:
        print(
            "No verified rows returned. Check the URL, provider availability, and warnings above."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
