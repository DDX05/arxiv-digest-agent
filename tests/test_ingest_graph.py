import json

from arxiv_agent.arxiv_client import ArxivError, Paper
from arxiv_agent.graph import build_ingest_graph

from .conftest import GOOD_BRIEFING, SAMPLE_PAPER, FakeArxiv, ScriptedLLM, make_pdf


def run(graph, text, **extra):
    return graph.invoke({"user_input": text, "warnings": [], **extra})


def test_paper_id_runs_end_to_end(tmp_path, make_deps):
    arxiv = FakeArxiv(papers=[SAMPLE_PAPER], pdf_path=make_pdf(tmp_path / "p.pdf"))
    llm = ScriptedLLM([GOOD_BRIEFING])
    deps = make_deps(arxiv=arxiv, llm=llm)
    state = run(build_ingest_graph(deps), "https://arxiv.org/abs/2401.00001v1")

    assert not state.get("error")
    assert state["mode"] == "paper"
    assert state["briefing"]["limitations"][0]["source"] == "stated"
    assert state["briefing"]["unverified_numbers"] == []
    assert "## Limitations" in state["briefing_markdown"]
    saved = json.loads((tmp_path / "out" / "2401.00001" / "briefing.json").read_text())
    assert saved["paper"]["title"] == SAMPLE_PAPER.title
    assert deps.index.count("2401.00001") == len(state["chunks"])
    assert "4 Limitations" in llm.calls[0][1]


def test_second_run_uses_cache(tmp_path, make_deps):
    arxiv = FakeArxiv(papers=[SAMPLE_PAPER], pdf_path=make_pdf(tmp_path / "p.pdf"))
    deps = make_deps(arxiv=arxiv, llm=ScriptedLLM([GOOD_BRIEFING]))
    graph = build_ingest_graph(deps)
    run(graph, "2401.00001")
    second = run(graph, "2401.00001")
    assert second["cached"] is True
    assert second["briefing"]["problem"] == GOOD_BRIEFING["problem"]


def test_unknown_id_fails_cleanly(make_deps):
    state = run(build_ingest_graph(make_deps()), "2401.99999")
    assert "no paper with ID 2401.99999" in state["error"]


def test_topic_with_no_results_broadens_then_fails(make_deps):
    arxiv = FakeArxiv(search_results=[[], [], []])
    llm = ScriptedLLM([{"keywords": ["quantum gravity", "kv cache"], "categories": ["cs.CL"], "prefer_recent": False}])
    state = run(build_ingest_graph(make_deps(arxiv=arxiv, llm=llm)), "kv cache for quantum gravity")
    assert len(arxiv.queries) == 3
    assert arxiv.queries[0] != arxiv.queries[1] != arxiv.queries[2]
    assert "after 3 searches" in state["error"]
    assert sum("broadened" in w for w in state["warnings"]) == 2


def test_topic_ranks_candidates_by_abstract(tmp_path, make_deps):
    unrelated = Paper(**{**SAMPLE_PAPER.to_dict(), "arxiv_id": "2401.00002", "title": "Protein folding with diffusion",
                         "abstract": "We fold proteins with a diffusion model trained on structures."})
    arxiv = FakeArxiv(search_results=[[unrelated, SAMPLE_PAPER]], pdf_path=make_pdf(tmp_path / "p.pdf"))
    llm = ScriptedLLM([{"keywords": ["kv cache"], "categories": [], "prefer_recent": False}, GOOD_BRIEFING])
    state = run(build_ingest_graph(make_deps(arxiv=arxiv, llm=llm)), "key-value cache eviction for long context")
    assert state["paper"]["arxiv_id"] == "2401.00001"
    assert state["alternatives"][0]["arxiv_id"] == "2401.00002"


def test_planner_failure_falls_back_to_keywords(tmp_path, make_deps):
    def broken(system, user):
        raise ValueError("not json")

    arxiv = FakeArxiv(search_results=[[SAMPLE_PAPER]], pdf_path=make_pdf(tmp_path / "p.pdf"))
    state = run(build_ingest_graph(make_deps(arxiv=arxiv, llm=ScriptedLLM([broken, GOOD_BRIEFING]))), "kv cache eviction")
    assert not state.get("error")
    assert any("plain keywords" in w for w in state["warnings"])


def test_pdf_failure_falls_back_to_abstract(make_deps):
    arxiv = FakeArxiv(papers=[SAMPLE_PAPER], pdf_error=ArxivError("download did not return a PDF"))
    llm = ScriptedLLM([GOOD_BRIEFING])
    state = run(build_ingest_graph(make_deps(arxiv=arxiv, llm=llm)), "2401.00001")
    assert state["parse_report"]["degraded"] is True
    assert len(state["chunks"]) == 1
    assert "only the abstract was available" in llm.calls[0][1]
    assert "Only the abstract was usable" in state["briefing_markdown"]


def test_bad_briefing_is_retried_with_feedback(tmp_path, make_deps):
    missing_limits = {**GOOD_BRIEFING, "limitations": []}
    arxiv = FakeArxiv(papers=[SAMPLE_PAPER], pdf_path=make_pdf(tmp_path / "p.pdf"))
    llm = ScriptedLLM([missing_limits, GOOD_BRIEFING])
    state = run(build_ingest_graph(make_deps(arxiv=arxiv, llm=llm)), "2401.00001")
    assert not state.get("error")
    assert "limitations" in llm.calls[1][1]


def test_invented_numbers_are_flagged(tmp_path, make_deps):
    made_up = {**GOOD_BRIEFING, "key_results": ["Cache memory drops by 72%", "Throughput improves 4.7x on A100"]}
    arxiv = FakeArxiv(papers=[SAMPLE_PAPER], pdf_path=make_pdf(tmp_path / "p.pdf"))
    state = run(build_ingest_graph(make_deps(arxiv=arxiv, llm=ScriptedLLM([made_up]))), "2401.00001")
    assert state["briefing"]["unverified_numbers"] == ["4.7"]


def test_fallback_source_is_flagged_in_the_briefing(tmp_path, make_deps):
    from_s2 = Paper(**{**SAMPLE_PAPER.to_dict(), "source": "Semantic Scholar", "categories": []})
    arxiv = FakeArxiv(papers=[from_s2], pdf_path=make_pdf(tmp_path / "p.pdf"))
    state = run(build_ingest_graph(make_deps(arxiv=arxiv, llm=ScriptedLLM([GOOD_BRIEFING]))), "2401.00001")
    assert any("Semantic Scholar" in w for w in state["warnings"])
    assert "**Categories:** not available" in state["briefing_markdown"]
