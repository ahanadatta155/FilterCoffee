"""Nemotron Scout: evidence-backed extraction from public web pages."""

from typing import Any


def extract(url: str, instruction: str, schema: dict | None = None, **options: Any):
    """Extract verified fields from a public page; see `scout.pipeline.run`."""
    from scout.pipeline import run

    return run(url, instruction, schema=schema, **options)


__all__ = ["extract"]
