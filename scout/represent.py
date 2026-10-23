"""Find sibling item groups and build bounded text, never full HTML prompts."""

import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass

from scout.clean import CleanPage, normalize_text, safe_link


@dataclass(frozen=True)
class CandidateItem:
    text: str
    links: list[str]
    attributes: dict[str, str]


@dataclass(frozen=True)
class CandidateGroup:
    signature: str
    items: list[CandidateItem]
    score: float


def find_candidate_groups(page: CleanPage) -> list[CandidateGroup]:
    groups = []
    for parent in page.tree.css("body, body *"):
        siblings = defaultdict(list)
        child = parent.first_child
        while child is not None:
            if child.tag in {"article", "div", "li", "tr", "section", "a"}:
                classes = ".".join(sorted(set(child.attributes.get("class", "").split())))
                siblings[(child.tag, classes)].append(child)
            child = child.next
        for (tag, classes), nodes in siblings.items():
            if len(nodes) < 3:
                continue
            texts = [normalize_text(node.text(separator=" ")) for node in nodes]
            lengths = [len(text) for text in texts]
            median = statistics.median(lengths)
            if median < 20 or any(length < median / 3 or length > median * 3 for length in lengths):
                continue
            items = []
            link_count = image_count = price_count = 0
            for node, text in zip(nodes, texts, strict=True):
                links = []
                anchors = ([node] if node.tag == "a" else []) + node.css("a[href]")
                for anchor in anchors:
                    link = safe_link(page.url, anchor.attributes.get("href", ""))
                    if link and link not in links:
                        links.append(link)
                attrs = {}
                for detail in [node, *node.css("[title], [aria-label], img[alt], [class]")]:
                    class_value = detail.attributes.get("class", "")
                    if "star-rating" in class_value.casefold():
                        attrs[f"{len(attrs)}:class"] = class_value[:200]
                    for name in ("title", "aria-label", "alt"):
                        value = detail.attributes.get(name)
                        if value:
                            attrs[f"{len(attrs)}:{name}"] = value[:200]
                        if len(attrs) >= 12:
                            break
                    if len(attrs) >= 12:
                        break
                link_count += bool(links)
                image_count += bool(node.css("img"))
                price_count += bool(re.search(r"[$€£¥]\s*\d|\d[.,]\d{2}\s*(?:USD|EUR|GBP)", text))
                items.append(CandidateItem(text[:1600], links[:8], attrs))
            score = len(nodes) * 2 + math.log1p(median) + link_count + 3 * price_count + image_count
            groups.append(CandidateGroup(f"{tag}.{classes}".rstrip("."), items, score))
    return sorted(groups, key=lambda group: group.score, reverse=True)[:3]


def build_representation(page: CleanPage, *, max_chars: int = 60_000, max_items: int = 100) -> str:
    """Priority: page identity, JSON-LD, tables, repeated groups, then main text."""
    if max_chars <= 0:
        return ""
    if max_items < 1:
        raise ValueError("max_items must be positive.")
    sections = [f"URL: {page.url}\nTITLE: {page.title or ''}\nMETA: {page.meta_description or ''}"]
    if page.jsonld:
        sections.append("JSON-LD\n" + json.dumps(page.jsonld, ensure_ascii=False))
    if page.tables:
        sections.append(
            "TABLES\n"
            + json.dumps(
                [
                    {"headers": table.headers, "rows": table.rows[:max_items]}
                    for table in page.tables
                ],
                ensure_ascii=False,
            )
        )
    groups = find_candidate_groups(page)
    if groups:
        rows = []
        for group in groups:
            rows.append(f"Group {group.signature}; {len(group.items)} items in page order")
            for index, item in enumerate(group.items[:max_items]):
                rows.append(
                    json.dumps(
                        {
                            "index": index,
                            "text": item.text,
                            "links": item.links,
                            "attributes": item.attributes,
                        },
                        ensure_ascii=False,
                    )
                )
        sections.append("CANDIDATE GROUPS\n" + "\n".join(rows))
    sections.append("MAIN TEXT\n" + page.main_text)
    return "\n\n".join(sections)[:max_chars]
