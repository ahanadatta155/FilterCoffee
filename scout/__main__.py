"""Command-line diagnostics. Extraction commands arrive with the pipeline."""

import argparse
import sys

from scout.config import load_settings
from scout.errors import ScoutError
from scout.llm import ModelClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scout", description="Nemotron Scout diagnostics")
    commands = parser.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models", help="Discover model IDs and show pricing")
    models.add_argument("--env-file", help="Optional explicit .env file")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.env_file)
        with ModelClient(settings) as client:
            print(f"Provider: {settings.provider}")
            print("Role | Model | Input / Output USD per 1M tokens | Images")
            for role in ("fast", "strong", "escalation"):
                model = getattr(client.models, role)
                if model is None:
                    print(f"{role} | unavailable")
                    continue
                rates = [
                    "unknown" if rate is None else f"${rate:.2f}"
                    for rate in (model.input_per_million, model.output_per_million)
                ]
                enabled = (
                    " (disabled)" if role == "escalation" and not settings.enable_ultra else ""
                )
                print(
                    f"{role}{enabled} | {model.id} | {' / '.join(rates)} | {model.supports_vision}"
                )
            if settings.provider == "nebius":
                print("Prices are unverified spec estimates, not confirmed billing rates.")
            for warning in client.warnings:
                print(f"Warning: {warning}")
    except ScoutError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
