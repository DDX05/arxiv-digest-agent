import json
import math
import operator
from datetime import date
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .arxiv_client import ArxivError, Paper, extract_arxiv_id
from .briefing import Briefing, build_summary_context, render_markdown, unverified_numbers
from .chunking import Chunk, chunk_sections
from .deps import Deps
from .llm import LLMError
from .pdf_parser import Paragraph, PDFParseError, Section, parse_pdf
from .prompts import BRIEFING_RETRY, BRIEFING_SYSTEM, BRIEFING_USER, QUERY_PLAN_SYSTEM
from .query import MAX_LEVEL, build_search_query, clean_plan, fallback_plan

LIMITATION_QUERY = "limitations weaknesses failure cases assumptions future work does not generalize"
LOW_RELEVANCE = 0.35


class IngestState(TypedDict, total=False):
    user_input: str
    refresh: bool
    mode: str
    arxiv_id: str
    topic: str
    plan: dict
    search_level: int
    search_query: str
    candidates: list[dict]
    alternatives: list[dict]
    paper: dict
    sections: list[dict]
    parse_report: dict
    chunks: list[dict]
    briefing: dict
    briefing_markdown: str
    output_dir: str
    cached: bool
    error: str
    warnings: Annotated[list[str], operator.add]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _recency_bonus(published: str) -> float:
    try:
        age_days = (date.today() - date.fromisoformat(published)).days
    except ValueError:
        return 0.0
    return 0.1 * max(0.0, 1.0 - age_days / 730)


def _source_warning(papers: list[Paper]) -> list[str]:
    sources = sorted({p.source for p in papers if p.source != "arXiv"})
    if not sources:
        return []
    return [f"The arXiv API refused the request, so paper metadata came from {', '.join(sources)}. arXiv categories are missing."]


def _coerce_briefing(raw: dict) -> dict:
    limitations = []
    for item in raw.get("limitations", []) or []:
        if isinstance(item, str):
            limitations.append({"point": item, "source": "inferred"})
        elif isinstance(item, dict):
            source = str(item.get("source", "inferred")).lower()
            limitations.append({"point": item.get("point", ""), "source": "stated" if source == "stated" else "inferred"})
    return {**raw, "limitations": limitations}


def _validation_problem(exc: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:4]]
    return "; ".join(parts)


def build_ingest_graph(deps: Deps):
    settings = deps.settings

    def understand_query(state: IngestState) -> dict:
        text = (state.get("user_input") or "").strip()
        if not text:
            return {"error": "Give me a topic or an arXiv ID."}
        arxiv_id = extract_arxiv_id(text)
        if arxiv_id:
            return {"mode": "paper", "arxiv_id": arxiv_id}
        try:
            plan = clean_plan(deps.llm.chat_json(QUERY_PLAN_SYSTEM, f"Topic: {text}", max_tokens=300), text)
            warnings = []
        except (LLMError, ValueError) as exc:
            plan = fallback_plan(text)
            warnings = [f"Query planner unavailable ({exc}); searched with plain keywords instead."]
        return {"mode": "topic", "topic": text, "plan": plan, "search_level": 0, "warnings": warnings}

    def retrieve(state: IngestState) -> dict:
        try:
            if state["mode"] == "paper":
                paper = deps.arxiv.fetch_by_id(state["arxiv_id"])
                if paper is None:
                    return {"candidates": []}
                return {"candidates": [paper.to_dict()], "paper": paper.to_dict(), "warnings": _source_warning([paper])}
            plan = state["plan"]
            level = state.get("search_level", 0)
            query = build_search_query(plan["keywords"], plan["categories"], level)
            sort_by = "submittedDate" if plan["prefer_recent"] and level == 0 else "relevance"
            papers = deps.arxiv.search(query, settings.arxiv_max_results, sort_by, fallback_text=state["topic"])
            return {
                "candidates": [p.to_dict() for p in papers],
                "search_query": query,
                "warnings": _source_warning(papers),
            }
        except ArxivError as exc:
            return {"error": f"arXiv could not be reached: {exc}"}

    def route_after_retrieve(state: IngestState) -> str:
        if state.get("error"):
            return "fail"
        if state.get("candidates"):
            return "check_cache" if state["mode"] == "paper" else "rank"
        if state["mode"] == "topic" and state.get("search_level", 0) < MAX_LEVEL:
            return "broaden"
        return "fail"

    def broaden(state: IngestState) -> dict:
        return {
            "search_level": state.get("search_level", 0) + 1,
            "warnings": [f"No results for `{state.get('search_query')}`, so the search was broadened."],
        }

    def rank(state: IngestState) -> dict:
        candidates = [Paper.from_dict(c) for c in state["candidates"]]
        vectors = deps.embedder.embed([state["topic"]] + [f"{p.title}. {p.abstract}" for p in candidates])
        query_vector = vectors[0]
        prefer_recent = state["plan"].get("prefer_recent", False)
        scored = []
        for paper, vector in zip(candidates, vectors[1:]):
            score = _cosine(query_vector, vector)
            if prefer_recent:
                score += _recency_bonus(paper.published)
            scored.append((paper, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        best, best_score = scored[0]
        warnings = []
        if best_score < LOW_RELEVANCE:
            warnings.append(
                f"Best match only scored {best_score:.2f} against the topic. The topic may be too vague; "
                "consider passing an arXiv ID from the candidates list."
            )
        alternatives = [{"arxiv_id": p.arxiv_id, "title": p.title, "score": s} for p, s in scored[1:5]]
        return {"paper": best.to_dict(), "arxiv_id": best.arxiv_id, "alternatives": alternatives, "warnings": warnings}

    def check_cache(state: IngestState) -> dict:
        arxiv_id = state["paper"]["arxiv_id"]
        if state.get("refresh") or not deps.sessions.exists(arxiv_id):
            return {"cached": False, "arxiv_id": arxiv_id}
        session = deps.sessions.load(arxiv_id)
        if not session.get("briefing") or deps.index.count(arxiv_id) != len(session.get("chunks", [])):
            return {"cached": False, "arxiv_id": arxiv_id}
        return {
            "cached": True,
            "arxiv_id": arxiv_id,
            "briefing": session["briefing"],
            "briefing_markdown": session["briefing_markdown"],
            "parse_report": session["parse_report"],
            "chunks": session["chunks"],
            "output_dir": session.get("output_dir", ""),
        }

    def fetch_parse(state: IngestState) -> dict:
        paper = Paper.from_dict(state["paper"])
        abstract = Section("Abstract", "abstract", [Paragraph(paper.abstract, 1)])
        try:
            pdf_path = deps.arxiv.download_pdf(paper, settings.pdf_dir)
            parsed = parse_pdf(pdf_path, settings.max_pdf_pages)
        except (ArxivError, PDFParseError) as exc:
            report = {"page_count": 0, "pages_read": 0, "char_count": 0, "section_titles": ["Abstract"],
                      "reference_count": 0, "degraded": True, "notes": [f"PDF unavailable: {exc}"]}
            return {
                "sections": [abstract.to_dict()],
                "parse_report": report,
                "warnings": ["Fell back to the arXiv abstract because the PDF could not be used."],
            }
        if parsed.degraded:
            return {
                "sections": [abstract.to_dict()],
                "parse_report": parsed.report(),
                "warnings": ["Fell back to the arXiv abstract because the PDF had almost no extractable text."],
            }
        body = [s for s in parsed.sections if s.kind != "abstract"]
        return {"sections": [s.to_dict() for s in [abstract, *body]], "parse_report": parsed.report()}

    def index(state: IngestState) -> dict:
        sections = [Section.from_dict(s) for s in state["sections"]]
        chunks = chunk_sections(sections, settings.chunk_chars, settings.chunk_overlap)
        try:
            deps.index.build(state["arxiv_id"], chunks)
        except Exception as exc:
            return {"error": f"Could not build the vector index: {exc}"}
        report = {**state["parse_report"], "chunk_count": len(chunks)}
        return {"chunks": [c.to_dict() for c in chunks], "parse_report": report}

    def summarize(state: IngestState) -> dict:
        paper = state["paper"]
        sections = [Section.from_dict(s) for s in state["sections"]]
        context = build_summary_context(sections, settings.summary_char_budget)
        chunks = [Chunk.from_dict(c) for c in state["chunks"]]
        hits = deps.retriever(state["arxiv_id"], chunks).search(LIMITATION_QUERY, k=4, include_references=False)
        evidence = "\n\n".join(f"({h.chunk.section}) {h.chunk.text}" for h in hits) or "None found."
        degraded = state["parse_report"].get("degraded", False)
        prompt = BRIEFING_USER.format(
            title=paper["title"],
            authors=", ".join(paper["authors"][:10]),
            degraded_note="NOTE: only the abstract was available. Keep every claim modest.\n" if degraded else "",
            context=context,
            limitation_evidence=evidence,
        )
        problem = None
        briefing = None
        for _ in range(2):
            message = prompt if problem is None else f"{prompt}\n\n{BRIEFING_RETRY.format(problem=problem)}"
            try:
                raw = deps.llm.chat_json(BRIEFING_SYSTEM, message, max_tokens=1600)
                briefing = Briefing.model_validate(_coerce_briefing(raw))
                break
            except ValidationError as exc:
                problem = _validation_problem(exc)
            except (ValueError, json.JSONDecodeError):
                problem = "the output was not a valid JSON object"
            except LLMError as exc:
                return {"error": f"Summarization failed: {exc}"}
        if briefing is None:
            return {"error": f"The model returned an unusable briefing twice ({problem})."}
        data = briefing.model_dump()
        data["unverified_numbers"] = unverified_numbers(briefing, f"{paper['abstract']}\n{context}\n{evidence}")
        return {"briefing": data}

    def save(state: IngestState) -> dict:
        paper = state["paper"]
        out_dir = settings.output_dir / paper["arxiv_id"].replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        warnings = list(dict.fromkeys(state.get("warnings", [])))
        markdown = render_markdown(paper, state["briefing"], state["parse_report"], warnings, state.get("alternatives", []))
        (out_dir / "briefing.md").write_text(markdown, encoding="utf-8")
        record = {"paper": paper, "briefing": state["briefing"], "parse_report": state["parse_report"], "warnings": warnings}
        (out_dir / "briefing.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        deps.sessions.save(
            {
                **record,
                "briefing_markdown": markdown,
                "chunks": state["chunks"],
                "output_dir": str(out_dir),
                "history": [],
            }
        )
        return {"briefing_markdown": markdown, "output_dir": str(out_dir)}

    def fail(state: IngestState) -> dict:
        if state.get("error"):
            return {"error": state["error"]}
        if state.get("mode") == "paper":
            return {"error": f"arXiv has no paper with ID {state['arxiv_id']}. Check the ID, e.g. 2401.12345."}
        tries = state.get("search_level", 0) + 1
        return {"error": f"No arXiv papers matched '{state.get('topic')}' after {tries} searches. Try other words or pass an arXiv ID."}

    def ok_or_fail(next_node: str):
        return lambda state: "fail" if state.get("error") else next_node

    graph = StateGraph(IngestState)
    for name, fn in [
        ("understand_query", understand_query),
        ("retrieve", retrieve),
        ("broaden", broaden),
        ("rank", rank),
        ("check_cache", check_cache),
        ("fetch_parse", fetch_parse),
        ("index", index),
        ("summarize", summarize),
        ("save", save),
        ("fail", fail),
    ]:
        graph.add_node(name, fn)

    graph.add_edge(START, "understand_query")
    graph.add_conditional_edges("understand_query", ok_or_fail("retrieve"), ["retrieve", "fail"])
    graph.add_conditional_edges("retrieve", route_after_retrieve, ["rank", "check_cache", "broaden", "fail"])
    graph.add_edge("broaden", "retrieve")
    graph.add_edge("rank", "check_cache")
    graph.add_conditional_edges("check_cache", lambda s: END if s.get("cached") else "fetch_parse", ["fetch_parse", END])
    graph.add_edge("fetch_parse", "index")
    graph.add_conditional_edges("index", ok_or_fail("summarize"), ["summarize", "fail"])
    graph.add_conditional_edges("summarize", ok_or_fail("save"), ["save", "fail"])
    graph.add_edge("save", END)
    graph.add_edge("fail", END)
    return graph.compile()
