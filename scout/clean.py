"""Extract page signals and text without executing scripts or fetching assets."""

import json
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import trafilatura
from selectolax.lexbor import LexborHTMLParser


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def safe_link(base: str, href: str) -> str | None:
    """Links are evidence, not fetch targets; strip credentials and non-web schemes."""
    try:
        parsed = urlsplit(urljoin(base, href))
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
        ):
            return None
        if "\\" in href or any(ord(char) < 32 for char in href):
            return None
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
    except ValueError:
        return None


@dataclass(frozen=True)
class Table:
    headers: list[str]
    rows: list[list[str]]


@dataclass(frozen=True)
class CleanPage:
    url: str
    title: str | None
    meta_description: str | None
    jsonld: list
    tables: list[Table]
    main_text: str
    source_text: str
    tree: LexborHTMLParser
    needs_render: bool
    canvas_only: bool
    blocked: bool
    warnings: list[str]


def clean_page(html: str, url: str) -> CleanPage:
    tree = LexborHTMLParser(html)
    title_node = tree.css_first("title")
    title = normalize_text(title_node.text()) if title_node else None
    meta = tree.css_first('meta[name="description"]')
    description = normalize_text(meta.attributes.get("content", "")) if meta else None
    scripts = bool(tree.css("script"))
    app_shell = bool(tree.css("#app, #root"))
    noscript = any("enable javascript" in node.text().lower() for node in tree.css("noscript"))
    canvas = bool(tree.css("canvas"))
    warnings, jsonld = [], []
    for node in tree.css('script[type="application/ld+json"]'):
        try:
            value = json.loads(node.text())
            if isinstance(value, (dict, list)):
                jsonld.append(value)
            else:
                warnings.append("Ignored non-object JSON-LD.")
        except (ValueError, RecursionError):
            warnings.append("Ignored malformed JSON-LD.")
    for node in reversed(
        tree.css("script, style, svg, noscript, template, [hidden], [aria-hidden='true']")
    ):
        node.decompose()
    for node in reversed(tree.css("[style]")):
        if re.search(
            r"(?:display\s*:\s*none|visibility\s*:\s*hidden)",
            node.attributes.get("style", ""),
            re.I,
        ):
            node.decompose()
    visible = normalize_text(tree.body.text(separator=" ") if tree.body else "")
    blocked = len(visible) < 1500 and bool(
        re.search(
            r"(?:complete (?:the |a )?captcha|verify (?:that )?you are human|"
            r"access denied|cf-chl-)",
            visible,
            re.I,
        )
    )
    needs_render = (
        noscript
        or ((scripts or app_shell) and len(visible) < 200)
        or (canvas and len(visible) < 200)
    )
    source_parts = [title or "", description or "", visible]
    source_parts.extend(json.dumps(value, ensure_ascii=False) for value in jsonld)
    for node in tree.css("a[href], [title], [aria-label], img[alt], time[datetime]"):
        for name in ("title", "aria-label", "alt", "datetime"):
            if node.attributes.get(name):
                source_parts.append(node.attributes[name])
        if node.attributes.get("href"):
            link = safe_link(url, node.attributes["href"])
            if link:
                source_parts.append(link)
    for node in tree.css("[class]"):
        classes = normalize_text(node.attributes.get("class", ""))
        if "star-rating" in classes.casefold():
            source_parts.append(classes)
    for node in reversed(tree.css("nav, footer, [role='navigation']")):
        node.decompose()
    tables = []
    for table in tree.css("table"):
        rows = [
            [normalize_text(cell.text(separator=" ")) for cell in row.css("th, td")]
            for row in table.css("tr")
        ]
        rows = [row for row in rows if any(row)]
        first_row = table.css_first("tr")
        has_headers = bool(first_row and first_row.css("th"))
        tables.append(
            Table(rows[0] if rows and has_headers else [], rows[1:] if has_headers else rows)
        )
    main_text = (
        trafilatura.extract(
            tree.html, url=url, include_comments=False, include_tables=False, output_format="txt"
        )
        if visible
        else None
    )
    fallback = tree.css_first("main, article") or tree.body
    if not main_text:
        main_text = fallback.text(separator=" ") if fallback else ""
    return CleanPage(
        url,
        title or None,
        description or None,
        jsonld,
        tables,
        normalize_text(main_text),
        normalize_text(" ".join(source_parts)),
        tree,
        needs_render,
        canvas and len(visible) < 200,
        blocked,
        warnings,
    )
