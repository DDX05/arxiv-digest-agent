import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .chunking import Chunk
from .deps import Deps
from .llm import LLMError
from .prompts import ANSWER_STRICT_SUFFIX, ANSWER_SYSTEM, ANSWER_USER, CONDENSE_SYSTEM

CITATION = re.compile(r"\[(\d+)\]")
CITATION_GROUP = re.compile(
    r"[\[【]\s*(?:excerpts?\s*)?(\d+(?:\s*(?:,|;|–|-|and)\s*\d+)*)(?:\s*†[^\]】]*)?\s*[\]】]", re.I
)
EXCERPT_PAREN = re.compile(r"\(\s*excerpts?\s*(\d+(?:\s*(?:,|;|–|-|and)\s*\d+)*)\s*\)", re.I)
SENTENCE = re.compile(r"(?<=[.!?])\s+")
MAX_ANSWER_ATTEMPTS = 2


def _expand(group: str) -> list[int]:
    numbers: list[int] = []
    for part in re.split(r"\s*(?:,|;|and)\s*", group):
        bounds = re.split(r"\s*[–-]\s*", part)
        if len(bounds) == 2 and all(b.isdigit() for b in bounds):
            low, high = int(bounds[0]), int(bounds[1])
            numbers.extend(range(low, high + 1) if 0 < high - low < 20 else [low, high])
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def normalize_citations(text: str) -> str:
    def to_brackets(match: re.Match) -> str:
        return "".join(f"[{n}]" for n in _expand(match.group(1)))

    return EXCERPT_PAREN.sub(to_brackets, CITATION_GROUP.sub(to_brackets, text))


class QAState(TypedDict, total=False):
    arxiv_id: str
    title: str
    chunks: list[dict]
    history: list[dict]
    question: str
    standalone: str
    hits: list[dict]
    best_similarity: float
    attempts: int
    answer: str
    citations: list[dict]
    status: str
    rejected: list[str]


def _grounding(answer: str, hit_count: int) -> tuple[str, list[int]]:
    text = answer.strip()
    cited = [int(n) for n in CITATION.findall(text)]
    if "NOT_IN_PAPER" in text.upper() and not cited:
        return "not_in_paper", []
    if not cited or any(n < 1 or n > hit_count for n in cited):
        return "retry", []
    sentences = [s for s in SENTENCE.split(text) if len(s.split()) > 3]
    covered = sum(1 for s in sentences if CITATION.search(s))
    if sentences and covered / len(sentences) < 0.34:
        return "retry", []
    return "answered", sorted(set(cited))


def build_qa_graph(deps: Deps):
    settings = deps.settings

    def condense(state: QAState) -> dict:
        history = state.get("history", [])[-3:]
        question = state["question"].strip()
        if not history:
            return {"standalone": question, "attempts": 0, "rejected": []}
        transcript = "\n".join(f"User: {t['question']}\nAssistant: {t['answer'][:300]}" for t in history)
        try:
            rewritten = deps.llm.chat(CONDENSE_SYSTEM, f"History:\n{transcript}\n\nLatest question: {question}", max_tokens=120, temperature=0)
            rewritten = rewritten.strip().strip('"') or question
        except LLMError:
            rewritten = question
        return {"standalone": rewritten, "attempts": 0, "rejected": []}

    def retrieve(state: QAState) -> dict:
        chunks = [Chunk.from_dict(c) for c in state["chunks"]]
        hits = deps.retriever(state["arxiv_id"], chunks).search(state["standalone"], k=settings.qa_top_k)
        best = max((h.dense for h in hits), default=0.0)
        return {
            "hits": [{"chunk": h.chunk.to_dict(), "dense": h.dense, "lexical": h.lexical} for h in hits],
            "best_similarity": best,
        }

    def route_after_retrieve(state: QAState) -> str:
        if not state.get("hits") or state.get("best_similarity", 0.0) < settings.min_similarity:
            return "refuse"
        return "answer"

    def answer(state: QAState) -> dict:
        excerpts = "\n\n".join(
            f"[{i}] (Section: {h['chunk']['section']}, page {h['chunk']['page']})\n{h['chunk']['text']}"
            for i, h in enumerate(state["hits"], start=1)
        )
        prompt = ANSWER_USER.format(title=state.get("title", ""), excerpts=excerpts, question=state["standalone"])
        if state.get("attempts", 0) > 0:
            prompt += ANSWER_STRICT_SUFFIX
        try:
            text = deps.llm.chat(ANSWER_SYSTEM, prompt, max_tokens=400, temperature=0.1)
        except LLMError as exc:
            return {"answer": f"The language model is unavailable: {exc}", "status": "error", "attempts": state.get("attempts", 0) + 1}
        return {"answer": normalize_citations(text.strip()), "attempts": state.get("attempts", 0) + 1, "status": "pending"}

    def check_grounding(state: QAState) -> dict:
        if state.get("status") == "error":
            return {}
        status, cited = _grounding(state["answer"], len(state["hits"]))
        if status == "retry" and state.get("attempts", 0) >= MAX_ANSWER_ATTEMPTS:
            status = "ungrounded"
        citations = [
            {"n": n, "section": state["hits"][n - 1]["chunk"]["section"], "page": state["hits"][n - 1]["chunk"]["page"]}
            for n in cited
        ]
        update = {"status": status, "citations": citations}
        if status in ("retry", "ungrounded"):
            update["rejected"] = [*state.get("rejected", []), state["answer"]]
        return update

    def route_after_check(state: QAState) -> str:
        status = state.get("status")
        if status == "retry":
            return "answer"
        if status in ("not_in_paper", "ungrounded"):
            return "refuse"
        return END

    def refuse(state: QAState) -> dict:
        sections = list(dict.fromkeys(h["chunk"]["section"] for h in state.get("hits", [])))[:3]
        if state.get("status") == "ungrounded":
            message = "I couldn't tie an answer to specific passages of the paper, so I won't guess."
            status = "ungrounded"
        else:
            message = "The paper doesn't cover this, as far as the retrieved passages show."
            status = "not_in_paper"
        if sections:
            message += f" Closest sections: {', '.join(sections)}."
        return {"answer": message, "citations": [], "status": status}

    graph = StateGraph(QAState)
    graph.add_node("condense", condense)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("check_grounding", check_grounding)
    graph.add_node("refuse", refuse)
    graph.add_edge(START, "condense")
    graph.add_edge("condense", "retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve, ["answer", "refuse"])
    graph.add_edge("answer", "check_grounding")
    graph.add_conditional_edges("check_grounding", route_after_check, ["answer", "refuse", END])
    graph.add_edge("refuse", END)
    return graph.compile()
