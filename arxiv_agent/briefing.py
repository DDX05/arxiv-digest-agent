import re
from typing import Literal

from pydantic import BaseModel, Field

from .pdf_parser import Section

PRIORITY = [
    (0, ("abstract",)),
    (1, ("introduction",)),
    (2, ("conclusion", "discussion", "limitation", "future work", "broader impact")),
    (3, ("method", "approach", "model", "framework", "architecture", "algorithm", "proposed")),
    (4, ("experiment", "result", "evaluation", "analysis", "benchmark", "ablation")),
]
DEFAULT_PRIORITY = 5
NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*%?")


class Limitation(BaseModel):
    point: str = Field(min_length=5)
    source: Literal["stated", "inferred"]


class Briefing(BaseModel):
    why_it_matters: str = Field(min_length=40)
    problem: str = Field(min_length=15)
    method: list[str] = Field(min_length=2)
    key_results: list[str] = Field(min_length=1)
    limitations: list[Limitation] = Field(min_length=1)
    follow_up_questions: list[str] = Field(min_length=2)


def _priority(section: Section) -> int:
    if section.kind == "abstract":
        return 0
    if section.kind == "appendix":
        return 6
    title = section.title.lower()
    for level, words in PRIORITY:
        if any(word in title for word in words):
            return level
    return DEFAULT_PRIORITY


def build_summary_context(sections: list[Section], budget: int) -> str:
    usable = [(i, s) for i, s in enumerate(sections) if s.kind not in ("front", "references") and s.text.strip()]
    per_section_cap = max(1500, budget // 4)
    chosen: dict[int, str] = {}
    remaining = budget
    for index, section in sorted(usable, key=lambda pair: (_priority(pair[1]), pair[0])):
        if remaining <= 200:
            break
        cap = len(section.text) if section.kind == "abstract" else per_section_cap
        take = min(len(section.text), cap, remaining)
        excerpt = section.text[:take]
        if take < len(section.text):
            excerpt = excerpt.rsplit(" ", 1)[0] + " [...]"
        chosen[index] = f"## {section.title}\n{excerpt}"
        remaining -= take
    return "\n\n".join(chosen[i] for i in sorted(chosen))


def _normalize_number(value: str) -> str:
    return value.replace(",", "").rstrip("%")


def unverified_numbers(briefing: Briefing, source_text: str) -> list[str]:
    available = {_normalize_number(n) for n in NUMBER.findall(source_text)}
    missing = []
    for line in briefing.key_results:
        for number in NUMBER.findall(line):
            normalized = _normalize_number(number)
            if len(normalized.replace(".", "")) < 2:
                continue
            if normalized not in available and number not in missing:
                missing.append(number)
    return missing


def render_markdown(paper: dict, briefing: dict, report: dict, warnings: list[str], alternatives: list[dict]) -> str:
    authors = ", ".join(paper["authors"][:8]) + (" et al." if len(paper["authors"]) > 8 else "")
    lines = [
        f"# {paper['title']}",
        "",
        f"**Authors:** {authors}  ",
        f"**arXiv ID:** {paper['arxiv_id']}{paper.get('version', '')}  ",
        f"**Published:** {paper['published']} (last updated {paper['updated']})  ",
        f"**Categories:** {', '.join(paper['categories']) or 'not available'}  ",
        f"**Links:** [abstract]({paper['abs_url']}) · [PDF]({paper['pdf_url']})",
        "",
        "## Why this paper matters",
        briefing["why_it_matters"],
        "",
        "## Problem",
        briefing["problem"],
        "",
        "## Method",
        *[f"- {item}" for item in briefing["method"]],
        "",
        "## Key results",
        *[f"- {item}" for item in briefing["key_results"]],
        "",
        "## Limitations",
        *[
            f"- {item['point']} _({'stated by the authors' if item['source'] == 'stated' else 'inferred, not stated by the authors'})_"
            for item in briefing["limitations"]
        ],
        "",
        "## Questions worth asking",
        *[f"- {item}" for item in briefing["follow_up_questions"]],
        "",
        "## How this briefing was built",
        f"- Pages read: {report.get('pages_read', 0)} of {report.get('page_count', 0)}",
        f"- Sections found: {len(report.get('section_titles', []))}, references found: {report.get('reference_count', 0)}",
        f"- Chunks indexed for QA: {report.get('chunk_count', 0)}",
    ]
    if report.get("degraded"):
        lines.append("- **Only the abstract was usable.** Treat method, results and limitations as shallow.")
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    for warning in warnings:
        lines.append(f"- {warning}")
    if briefing.get("unverified_numbers"):
        joined = ", ".join(briefing["unverified_numbers"])
        lines.append(f"- Numbers in key results that did not appear in the text the model saw: {joined}. Check them.")
    if alternatives:
        lines += ["", "## Other candidates for this topic"]
        lines += [f"- {a['arxiv_id']}: {a['title']} (score {a['score']:.2f})" for a in alternatives]
    return "\n".join(lines) + "\n"
