from arxiv_agent.chunking import chunk_sections
from arxiv_agent.pdf_parser import Paragraph, Section, parse_pdf

from .conftest import make_pdf


def test_parse_pdf_finds_sections_and_references(tmp_path):
    parsed = parse_pdf(make_pdf(tmp_path / "paper.pdf"))
    titles = [s.title for s in parsed.sections]
    assert "1 Introduction" in titles
    assert "4 Limitations" in titles
    kinds = {s.title: s.kind for s in parsed.sections}
    assert kinds["References"] == "references"
    assert kinds["2 Method"] == "body"
    assert len(parsed.references) == 2
    assert not parsed.degraded


def test_blank_pdf_is_flagged_degraded(tmp_path):
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc.save(tmp_path / "blank.pdf")
    parsed = parse_pdf(tmp_path / "blank.pdf")
    assert parsed.degraded


def test_page_limit_is_reported(tmp_path):
    parsed = parse_pdf(make_pdf(tmp_path / "paper.pdf"), max_pages=1)
    assert parsed.pages_read == 1
    assert any("skipped" in note for note in parsed.notes)


def test_chunks_respect_sections_and_size():
    long_text = " ".join(f"Sentence number {i} talks about caching." for i in range(60))
    sections = [
        Section("Front matter", "front", [Paragraph("Title and authors", 1)]),
        Section("2 Method", "body", [Paragraph(long_text, 3)]),
        Section("3 Results", "body", [Paragraph("Short result paragraph that is long enough to keep.", 5)]),
    ]
    chunks = chunk_sections(sections, target_chars=300, overlap_chars=60)
    assert all(c.section != "Front matter" for c in chunks)
    assert {c.section for c in chunks} == {"2 Method", "3 Results"}
    assert all(len(c.text) <= 380 for c in chunks)
    method = [c for c in chunks if c.section == "2 Method"]
    assert len(method) > 3
    last_sentence = method[0].text.split(". ")[-1]
    assert method[1].text.startswith(last_sentence.rstrip("."))
    assert chunks[-1].page == 5


def test_real_latex_pdf(tmp_path):
    from pathlib import Path

    parsed = parse_pdf(Path(__file__).parent / "fixtures" / "latex_two_column.pdf")
    titles = [s.title for s in parsed.sections]
    assert "3 Method > 3.1 Routing Network" in titles
    assert "2 Related Work" in titles
    kinds = {s.title: s.kind for s in parsed.sections}
    assert kinds["A Hyperparameters"] == "appendix"
    intro = next(s for s in parsed.sections if s.title == "1 Introduction").text
    assert "attention is quadratic" in intro
    assert len(parsed.references) == 3
    assert parsed.references[0].startswith("[1] A. Vaswani")
