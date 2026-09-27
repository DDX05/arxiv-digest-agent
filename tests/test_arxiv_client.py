import pytest

from arxiv_agent.arxiv_client import ArxivError, extract_arxiv_id, parse_feed

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <updated>2024-02-01T10:00:00Z</updated>
    <published>2024-01-22T18:59:59Z</published>
    <title>A Paper
      With a Wrapped Title</title>
    <summary>  First line of the abstract.
      Second line.  </summary>
    <author><name>Jane Doe</name></author>
    <author><name>John Roe</name></author>
    <link href="http://arxiv.org/abs/2401.12345v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v2" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""

ERROR_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_1234</id><title>Error</title></entry>
</feed>"""


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2401.12345", "2401.12345"),
        ("2401.12345v3", "2401.12345"),
        ("https://arxiv.org/abs/2401.12345", "2401.12345"),
        ("https://arxiv.org/pdf/2401.12345v2.pdf", "2401.12345"),
        ("arXiv:1706.03762", "1706.03762"),
        ("hep-th/9901001", "hep-th/9901001"),
        ("https://arxiv.org/abs/math.GT/0309136", "math.GT/0309136"),
        ("recent work on KV-cache compression for LLMs", None),
        ("papers from 2024 on retrieval", None),
    ],
)
def test_extract_arxiv_id(text, expected):
    assert extract_arxiv_id(text) == expected


def test_parse_feed_reads_metadata():
    paper = parse_feed(FEED)[0]
    assert paper.arxiv_id == "2401.12345"
    assert paper.version == "v2"
    assert paper.title == "A Paper With a Wrapped Title"
    assert paper.abstract == "First line of the abstract. Second line."
    assert paper.authors == ["Jane Doe", "John Roe"]
    assert paper.published == "2024-01-22"
    assert paper.pdf_url == "https://arxiv.org/pdf/2401.12345v2"
    assert paper.categories == ["cs.CL", "cs.LG"]
    assert paper.primary_category == "cs.CL"


def test_parse_feed_skips_error_entries():
    assert parse_feed(ERROR_FEED) == []


def test_parse_feed_rejects_garbage():
    with pytest.raises(ArxivError):
        parse_feed("<html>rate limited")


def _client(handler, monkeypatch):
    import httpx

    from arxiv_agent import arxiv_client

    monkeypatch.setattr(arxiv_client.time, "sleep", lambda seconds: None)
    return arxiv_client.ArxivClient(min_interval=0, transport=httpx.MockTransport(handler))


def test_406_on_export_host_falls_back_to_main_host(monkeypatch):
    import httpx

    seen = []

    def handler(request):
        seen.append(request.url.host)
        if request.url.host == "export.arxiv.org":
            return httpx.Response(406, content=b"")
        return httpx.Response(200, content=FEED.encode())

    paper = _client(handler, monkeypatch).fetch_by_id("2401.12345")
    assert paper.arxiv_id == "2401.12345"
    assert seen == ["export.arxiv.org"] * 3 + ["arxiv.org"]


def test_406_that_still_carries_a_feed_is_used(monkeypatch):
    import httpx

    handler = lambda request: httpx.Response(406, content=FEED.encode())
    assert _client(handler, monkeypatch).fetch_by_id("2401.12345").title == "A Paper With a Wrapped Title"


def test_every_host_refusing_raises_a_clear_error(monkeypatch):
    import httpx

    handler = lambda request: httpx.Response(406, content=b"")
    with pytest.raises(ArxivError, match="refused every attempt"):
        _client(handler, monkeypatch).fetch_by_id("2401.12345")


def test_requests_ask_for_atom(monkeypatch):
    import httpx

    captured = {}

    def handler(request):
        captured.update(request.headers)
        return httpx.Response(200, content=FEED.encode())

    _client(handler, monkeypatch).fetch_by_id("2401.12345")
    assert captured["accept"].startswith("application/atom+xml")


S2_SEARCH = {
    "total": 2,
    "data": [
        {
            "paperId": "abc",
            "externalIds": {"ArXiv": "2401.12345", "DOI": "10.1/x"},
            "title": "Learned Token Eviction",
            "abstract": "We evict tokens.",
            "authors": [{"authorId": "1", "name": "Asha Rao"}],
            "publicationDate": "2024-01-22",
            "year": 2024,
        },
        {"paperId": "def", "externalIds": {"DOI": "10.1/y"}, "title": "Not on arXiv", "abstract": "x", "authors": []},
    ],
}

OPENALEX_SEARCH = {
    "meta": {"count": 1},
    "results": [
        {
            "id": "https://openalex.org/W1",
            "display_name": "Cache Compression at Scale",
            "publication_date": "2024-05-01",
            "doi": None,
            "authorships": [{"author": {"display_name": "Ben Ortiz"}}],
            "abstract_inverted_index": {"We": [0], "compress": [1], "caches.": [2]},
            "locations": [
                {"source": {"id": "https://openalex.org/S4306400194"}, "landing_page_url": "https://arxiv.org/abs/2405.00001v2"}
            ],
        }
    ],
}


def test_search_falls_back_to_semantic_scholar(monkeypatch):
    import httpx

    def handler(request):
        if "arxiv.org" in request.url.host:
            return httpx.Response(406, content=b"")
        if request.url.host == "api.semanticscholar.org":
            assert request.url.params["query"] == "kv cache compression"
            return httpx.Response(200, json=S2_SEARCH)
        raise AssertionError(request.url)

    papers = _client(handler, monkeypatch).search('abs:"kv cache"', fallback_text="kv cache compression")
    assert [p.arxiv_id for p in papers] == ["2401.12345"]
    assert papers[0].source == "Semantic Scholar"
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2401.12345"
    assert papers[0].authors == ["Asha Rao"]


def test_search_falls_back_to_openalex_when_semantic_scholar_is_busy(monkeypatch):
    import httpx

    def handler(request):
        if "arxiv.org" in request.url.host:
            return httpx.Response(406, content=b"")
        if request.url.host == "api.semanticscholar.org":
            return httpx.Response(429, json={"message": "Too Many Requests"})
        return httpx.Response(200, json=OPENALEX_SEARCH)

    papers = _client(handler, monkeypatch).search("q", fallback_text="cache compression")
    assert papers[0].arxiv_id == "2405.00001"
    assert papers[0].abstract == "We compress caches."
    assert papers[0].source == "OpenAlex"


def test_fetch_by_id_falls_back_to_semantic_scholar(monkeypatch):
    import httpx

    def handler(request):
        if "arxiv.org" in request.url.host:
            return httpx.Response(406, content=b"")
        assert request.url.path.endswith("/paper/arXiv:2401.12345")
        return httpx.Response(200, json=S2_SEARCH["data"][0])

    paper = _client(handler, monkeypatch).fetch_by_id("2401.12345")
    assert paper.title == "Learned Token Eviction"


def test_openalex_reads_arxiv_doi():
    from arxiv_agent.arxiv_client import paper_from_openalex

    work = {**OPENALEX_SEARCH["results"][0], "doi": "https://doi.org/10.48550/arxiv.2311.04567", "locations": []}
    assert paper_from_openalex(work).arxiv_id == "2311.04567"


def test_search_without_fallback_text_still_raises(monkeypatch):
    import httpx

    handler = lambda request: httpx.Response(406, content=b"")
    with pytest.raises(ArxivError):
        _client(handler, monkeypatch).search("q")
