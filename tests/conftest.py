import hashlib
import json
import math
import re
from pathlib import Path

import pymupdf
import pytest

from arxiv_agent.arxiv_client import Paper
from arxiv_agent.config import Settings
from arxiv_agent.deps import Deps
from arxiv_agent.session import SessionStore
from arxiv_agent.store import PaperIndex

SAMPLE_PAPER = Paper(
    arxiv_id="2401.00001",
    version="v1",
    title="Compressing the Key-Value Cache with Learned Token Eviction",
    authors=["Asha Rao", "Ben Ortiz"],
    abstract=(
        "Large language models store a key-value cache that grows with context length. "
        "We propose learned token eviction, which keeps a small budget of cached tokens and "
        "reduces cache memory by 72% with under 1 point loss on long-context benchmarks."
    ),
    published="2024-01-02",
    updated="2024-02-10",
    pdf_url="https://arxiv.org/pdf/2401.00001v1",
    abs_url="https://arxiv.org/abs/2401.00001v1",
    primary_category="cs.CL",
    categories=["cs.CL", "cs.LG"],
)

BODY = {
    "1 Introduction": [
        "Serving long prompts is expensive because the key-value cache grows linearly with sequence length. "
        "On a 7B model with 32k tokens the cache alone needs more memory than the weights.",
        "We ask whether most cached tokens can be dropped without hurting generation quality.",
    ],
    "2 Method": [
        "Learned token eviction trains a small scoring network on attention statistics. "
        "At each decoding step it keeps the top scoring tokens within a fixed budget of 512 tokens.",
        "The scorer has 1.2M parameters and adds 3% latency per step.",
    ],
    "3 Experiments": [
        "We evaluate on LongBench and a needle retrieval test with Llama-2-7B. "
        "Learned eviction reduces cache memory by 72% while the LongBench score drops from 41.3 to 40.6.",
        "Heuristic baselines such as keeping recent tokens lose 6.8 points at the same budget.",
    ],
    "4 Limitations": [
        "Our experiments only cover models up to 7B parameters and English benchmarks. "
        "The scorer must be retrained for every new model family.",
    ],
    "5 Conclusion": [
        "Learned eviction is a cheap way to shrink the key-value cache for long-context inference.",
    ],
    "References": [
        "[1] Vaswani et al. Attention is all you need. NeurIPS 2017.",
        "[2] Zhang et al. H2O: Heavy-hitter oracle for efficient generative inference. 2023.",
    ],
}


def make_pdf(path: Path, body: dict = BODY) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60
    page.insert_text((72, y), SAMPLE_PAPER.title, fontsize=16)
    y += 30
    page.insert_text((72, y), "Asha Rao, Ben Ortiz", fontsize=10)
    y += 40
    for heading, paragraphs in body.items():
        if y > 650:
            page = doc.new_page()
            y = 60
        page.insert_text((72, y), heading, fontsize=13)
        y += 22
        for paragraph in paragraphs:
            rect = pymupdf.Rect(72, y, 520, y + 110)
            page.insert_textbox(rect, paragraph, fontsize=10)
            y += 115
            if y > 700:
                page = doc.new_page()
                y = 60
    doc.save(path)
    return path


class HashEmbedder:
    dim = 384

    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vector[bucket] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


class ScriptedLLM:
    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def _next(self, system, user):
        self.calls.append((system, user))
        if not self.replies:
            raise AssertionError("LLM called more times than the test expected")
        reply = self.replies.pop(0)
        return reply(system, user) if callable(reply) else reply

    def chat(self, system, user, **kwargs):
        return self._next(system, user)

    def chat_json(self, system, user, **kwargs):
        from arxiv_agent.llm import parse_json_object

        reply = self._next(system, user)
        return reply if isinstance(reply, dict) else parse_json_object(reply)


class FakeArxiv:
    def __init__(self, papers=None, search_results=None, pdf_path=None, pdf_error=None):
        self.papers = {p.arxiv_id: p for p in (papers or [])}
        self.search_results = list(search_results or [])
        self.pdf_path = pdf_path
        self.pdf_error = pdf_error
        self.queries = []

    def fetch_by_id(self, arxiv_id):
        return self.papers.get(arxiv_id)

    def search(self, query, max_results=25, sort_by="relevance", fallback_text=None):
        self.queries.append(query)
        return self.search_results.pop(0) if self.search_results else []

    def download_pdf(self, paper, dest_dir):
        if self.pdf_error:
            raise self.pdf_error
        return self.pdf_path


GOOD_BRIEFING = {
    "why_it_matters": "The paper shows a learned way to drop most cached tokens during decoding, which cuts serving memory for long prompts.",
    "problem": "The key-value cache grows with context length and dominates memory for long prompts.",
    "method": ["Train a small scorer on attention statistics", "Keep the top 512 tokens at each step"],
    "key_results": ["Cache memory drops by 72%", "LongBench falls from 41.3 to 40.6"],
    "limitations": [{"point": "Only tested up to 7B parameters", "source": "stated"}],
    "follow_up_questions": ["How does it do on 70B models?", "What does the scorer cost to train?", "Does it work for code?"],
}


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        provider="groq",
        base_url="http://unused",
        api_key="test",
        model="test",
        embed_model="hash",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
        max_pdf_pages=60,
        summary_char_budget=8000,
        chunk_chars=400,
        chunk_overlap=80,
        arxiv_max_results=25,
        min_similarity=0.2,
        qa_top_k=4,
    )


@pytest.fixture
def make_deps(tmp_path):
    def factory(arxiv=None, llm=None):
        settings = make_settings(tmp_path)
        embedder = HashEmbedder()
        return Deps(
            settings=settings,
            arxiv=arxiv or FakeArxiv(),
            embedder=embedder,
            index=PaperIndex(settings.chroma_dir, embedder),
            sessions=SessionStore(settings.session_dir),
            llm_factory=lambda: llm or ScriptedLLM(),
        )

    return factory
