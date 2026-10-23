"""Run a live, free-model product extraction from the project folder."""

import argparse
import sys
import unicodedata

from scout import extract
from scout.config import load_settings
from scout.errors import ScoutError
from scout.pipeline import RunOptions

DEFAULT_URL = "https://books.toscrape.com/"
SCHEMA = {
    "fields": [
        {"name": "title", "type": "string", "description": "Book title", "required": True},
        {"name": "price", "type": "number", "description": "Listed price in GBP"},
        {"name": "rating", "type": "string", "description": "Star rating"},
        {"name": "availability", "type": "string", "description": "Availability text"},
    ]
}


def _display(value: object) -> str:
    return "".join(char for char in str(value) if unicodedata.category(char) not in {"Cc", "Cf"})


def _table(items: list) -> None:
    columns = ("title", "price", "rating", "availability")
    widths = (48, 10, 12, 24)
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
    parser.add_argument(
        "--url", default=DEFAULT_URL, help=f"Page to scrape (default: {DEFAULT_URL})"
    )
    parser.add_argument(
        "--max-items", type=int, default=20, help="Maximum rows to ask the model for"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Fetch and extract again instead of using cache"
    )
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        if settings.provider != "openrouter":
            print(
                "This runner is intentionally free-only. Set SCOUT_PROVIDER=openrouter in .env.",
                file=sys.stderr,
            )
            return 1
        print("Fetching the page and running the OpenRouter free model...", flush=True)
        result = extract(
            args.url,
            "Extract each book's title, price, rating, and availability.",
            schema=SCHEMA,
            options=RunOptions(max_items=args.max_items, force_refresh=args.refresh),
        )
    except (ScoutError, ValueError) as error:
        print(f"Could not start Nemotron Scout: {error}", file=sys.stderr)
        return 1

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    print(f"Nemotron Scout | {_display(result.meta.url)}")
    print(
        f"Fetch: {result.meta.fetch_tier} | Items: {len(result.items)} "
        f"| Cache hit: {result.meta.cache_hit}"
    )
    print(
        f"Model calls: {len(result.meta.model_calls)} | Cost: ${result.meta.estimated_cost_usd:.4f}"
    )
    if result.items:
        print()
        _table(result.items)
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
