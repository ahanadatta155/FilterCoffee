"""Normalize model output and verify every non-null value against page text."""

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlsplit

from scout.clean import normalize_text
from scout.contracts import Evidence, ExtractedItem, ExtractionPlan


@dataclass(frozen=True)
class ValidationResult:
    items: list[ExtractedItem]
    errors: list[str]


def _normalized(value: str) -> str:
    return normalize_text(unicodedata.normalize("NFKC", value)).casefold()


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"\w+", _normalized(value), re.UNICODE)}


def _number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = str(value).strip().replace("−", "-")
        text = re.sub(r"[^\d,.+\-eE]", "", text)
        if not text:
            raise ValueError
        if "," in text and "." in text:
            text = (
                text.replace(",", "")
                if text.rfind(".") > text.rfind(",")
                else text.replace(".", "").replace(",", ".")
            )
        elif text.count(",") == 1:
            left, right = text.split(",")
            text = left + right if len(right) == 3 else left + "." + right
        result = float(text)
    if not math.isfinite(result):
        raise ValueError
    return result


def _coerce(value: Any, kind: str) -> tuple[Any, bool]:
    if kind == "string":
        if not isinstance(value, str):
            raise ValueError
        normalized = normalize_text(value)
        return normalized, normalized != value
    if kind == "number":
        result = _number(value)
        return result, not isinstance(value, (float, int)) or result != value
    if kind == "integer":
        result = _number(value)
        if not result.is_integer():
            raise ValueError
        integer = int(result)
        return integer, not isinstance(value, int) or integer != value
    if kind == "boolean":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, str) and value.strip().casefold() in {"yes", "true", "in stock"}:
            return True, True
        if isinstance(value, str) and value.strip().casefold() in {"no", "false", "out of stock"}:
            return False, True
        raise ValueError
    if kind == "date":
        text = value.isoformat() if isinstance(value, date) else str(value).strip()
        result = date.fromisoformat(text)
        return result.isoformat(), result.isoformat() != text
    if kind == "url":
        if not isinstance(value, str):
            raise ValueError
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError
        return value, False
    if kind == "list[string]":
        if isinstance(value, str):
            return [normalize_text(value)], True
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return [normalize_text(item) for item in value], False
        raise ValueError
    raise ValueError


def _supported(value: Any, kind: str, quote: str) -> bool:
    source, evidence = _normalized(quote), _normalized(str(value))
    if not source:
        return False
    if kind in {"number", "integer"}:
        try:
            numeric = _number(value)
            found = {
                _number(match.group())
                for match in re.finditer(
                    r"(?<!\w)[\$€£¥]?\s*[-+]?\d[\d.,]*(?:[eE][-+]?\d+)?", quote
                )
            }
            return numeric in found
        except (ValueError, OverflowError):
            return False
    if kind == "boolean":
        phrases = (
            {"in stock", "available", "present"}
            if value is True
            else {"out of stock", "unavailable", "not available"}
        )
        expected = "true" if value is True else "false"
        return bool(re.search(rf"\b{expected}\b", source)) or any(
            _normalized(phrase) in source for phrase in phrases
        )
    if kind == "url":
        return evidence in source
    values = value if isinstance(value, list) else [value]
    for entry in values:
        tokens = _tokens(str(entry))
        if not tokens:
            return False
        overlap = len(tokens & _tokens(quote)) / len(tokens)
        if _normalized(str(entry)) not in source and overlap < 0.6:
            return False
    return True


def _context(source: str, value: str, size: int = 180) -> str | None:
    index = _normalized(source).find(_normalized(value))
    if index < 0:
        return None
    # Keep context lookup unicode-normalized without returning text that was fabricated.
    tokens = re.finditer(re.escape(value), source, re.I)
    match = next(tokens, None)
    if match:
        start, end = match.span()
        return source[max(0, start - size // 2) : min(len(source), end + size // 2)]
    return None


def validate_items(
    raw: Any, plan: ExtractionPlan, source: str, *, vision: bool = False
) -> ValidationResult:
    errors: list[str] = []
    raw_items = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(raw_items, list):
        return ValidationResult([], ["items must be a JSON array"])
    fields = {field.name: field for field in plan.fields}
    output, seen = [], set()
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict) or not isinstance(raw_item.get("data"), dict):
            errors.append(f"item {index}: missing object data")
            continue
        data_in = raw_item["data"]
        evidence_in = raw_item.get("evidence", [])
        if not isinstance(evidence_in, list):
            evidence_in = []
            errors.append(f"item {index}: evidence must be an array")
        evidence_by_field = {}
        for evidence in evidence_in:
            if isinstance(evidence, dict) and isinstance(evidence.get("field"), str):
                evidence_by_field[evidence["field"]] = evidence
        data, evidence_out, deterministic, non_null, identity = {}, [], 0.9, 0, []
        coercions = 0
        for name, field in fields.items():
            raw_value = data_in.get(name)
            if raw_value is None:
                data[name] = None
                if field.required:
                    deterministic -= 0.3
                    errors.append(f"item {index}: required field {name} is missing")
                continue
            try:
                value, coerced = _coerce(raw_value, field.type)
            except (ValueError, TypeError, OverflowError):
                data[name] = None
                errors.append(f"item {index}: field {name} has invalid type or value")
                continue
            if coerced:
                coercions += 1
            evidence = evidence_by_field.get(name, {})
            quote = evidence.get("quote") if isinstance(evidence, dict) else None
            if not isinstance(quote, str) or not quote.strip():
                quote = None if vision else _context(source, str(value))
                if quote:
                    deterministic -= 0.35
                    errors.append(
                        f"item {index}: evidence missing for {name}; source context added"
                    )
                else:
                    deterministic -= 0.35
                    errors.append(f"item {index}: evidence missing for {name}")
            elif not vision and _normalized(quote) not in _normalized(source):
                deterministic -= 0.35
                errors.append(f"item {index}: quote for {name} is not in source")
                quote = None
            elif not _supported(value, field.type, quote):
                deterministic -= 0.25
                errors.append(f"item {index}: quote does not support {name}")
                quote = None
            else:
                deterministic += 0.1
            if quote:
                evidence_out.append(
                    Evidence(
                        field=name,
                        quote=quote,
                        selector="screenshot" if vision else evidence.get("selector"),
                    )
                )
            else:
                data[name] = None
                continue
            data[name] = value
            non_null += 1
            if len(identity) < 3:
                identity.append(_normalized(str(value)))
        deterministic -= min(0.1 * coercions, 0.3)
        deterministic = min(0.99, max(0.05, deterministic))
        model_conf = raw_item.get("confidence")
        try:
            if model_conf is None:
                raise ValueError
            model_conf = min(1.0, max(0.0, float(model_conf)))
            if not math.isfinite(model_conf):
                model_conf = None
        except (ValueError, TypeError):
            model_conf = None
        confidence = (
            0.7 * deterministic + 0.3 * model_conf if model_conf is not None else deterministic
        )
        if vision:
            confidence = min(confidence, 0.75)
        if non_null:
            key = hashlib.sha256("|".join(identity).encode("utf-8")).hexdigest()
            if plan.mode == "list" and key in seen:
                continue
            seen.add(key)
            output.append(
                ExtractedItem(
                    data=data,
                    evidence=evidence_out,
                    confidence=confidence,
                    model_confidence=model_conf,
                )
            )
    return ValidationResult(output, errors)
