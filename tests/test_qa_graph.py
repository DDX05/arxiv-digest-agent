import pytest

from arxiv_agent.graph import build_ingest_graph
from arxiv_agent.qa import build_qa_graph

from .conftest import GOOD_BRIEFING, SAMPLE_PAPER, FakeArxiv, ScriptedLLM, make_pdf


@pytest.fixture
def session(tmp_path, make_deps):
    arxiv = FakeArxiv(papers=[SAMPLE_PAPER], pdf_path=make_pdf(tmp_path / "p.pdf"))
    deps = make_deps(arxiv=arxiv, llm=ScriptedLLM([GOOD_BRIEFING]))
    build_ingest_graph(deps).invoke({"user_input": "2401.00001", "warnings": []})
    return deps.sessions.load("2401.00001")


def ask(make_deps, session, question, replies, history=None):
    llm = ScriptedLLM(replies)
    deps = make_deps(llm=llm)
    result = build_qa_graph(deps).invoke(
        {
            "arxiv_id": "2401.00001",
            "title": SAMPLE_PAPER.title,
            "chunks": session["chunks"],
            "history": history or [],
            "question": question,
        }
    )
    return result, llm


def test_grounded_answer_keeps_citations(session, make_deps):
    result, _ = ask(make_deps, session, "How much does learned eviction reduce cache memory on LongBench?",
                    ["It cuts cache memory by 72% while LongBench drops from 41.3 to 40.6 [1]."])
    assert result["status"] == "answered"
    assert result["citations"][0]["n"] == 1


def test_off_topic_question_is_refused_without_calling_llm(session, make_deps):
    result, llm = ask(make_deps, session, "Who won the 1998 football world cup final?", [])
    assert result["status"] == "not_in_paper"
    assert llm.calls == []


def test_model_saying_not_in_paper_is_respected(session, make_deps):
    result, _ = ask(make_deps, session, "What GPU memory bandwidth did the cache eviction experiments use?", ["NOT_IN_PAPER"])
    assert result["status"] == "not_in_paper"


def test_uncited_answer_is_retried_then_refused(session, make_deps):
    uncited = "The scorer is a transformer with twelve layers trained on web text."
    result, llm = ask(make_deps, session, "How is the eviction scorer network trained?", [uncited, uncited])
    assert result["status"] == "ungrounded"
    assert len(llm.calls) == 2
    assert "no valid excerpt citations" in llm.calls[1][1]


def test_out_of_range_citation_triggers_retry(session, make_deps):
    result, llm = ask(make_deps, session, "How big is the scorer network?",
                      ["It has 1.2M parameters [9].", "It has 1.2M parameters and adds 3% latency [1]."])
    assert result["status"] == "answered"
    assert len(llm.calls) == 2


def test_follow_up_is_condensed_with_history(session, make_deps):
    history = [{"question": "What is learned token eviction?", "answer": "A scorer that keeps top tokens [1]."}]
    result, llm = ask(make_deps, session, "How much latency does it add?",
                      ["How much latency does learned token eviction add per decoding step?",
                       "It adds 3% latency per step [1]."], history)
    assert result["standalone"].startswith("How much latency does learned token eviction")
    assert result["status"] == "answered"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Eight heads 【1†L3-L5】.", "Eight heads [1]."),
        ("Eight heads [1, 3].", "Eight heads [1][3]."),
        ("Eight heads [2-4].", "Eight heads [2][3][4]."),
        ("Eight heads (Excerpt 2).", "Eight heads [2]."),
        ("Eight heads [Excerpt 1].", "Eight heads [1]."),
        ("Eight heads (8 in total) [1].", "Eight heads (8 in total) [1]."),
    ],
)
def test_citation_styles_are_normalized(raw, expected):
    from arxiv_agent.qa import normalize_citations

    assert normalize_citations(raw) == expected


def test_gpt_oss_style_citations_are_accepted(session, make_deps):
    result, llm = ask(make_deps, session, "How big is the scorer network and what latency does it add?",
                      ["The scorer has 1.2M parameters and adds 3% latency per step 【1†L2-L3】."])
    assert result["status"] == "answered"
    assert len(llm.calls) == 1
    assert result["answer"].endswith("[1].")


def test_rejected_drafts_are_kept(session, make_deps):
    uncited = "The scorer is a transformer with twelve layers trained on web text."
    result, _ = ask(make_deps, session, "How is the eviction scorer network trained?", [uncited, uncited])
    assert result["rejected"] == [uncited, uncited]
