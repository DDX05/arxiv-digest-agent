import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

API_URLS = ("https://export.arxiv.org/api/query", "https://arxiv.org/api/query")
FEED_HEADERS = {
    "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
PDF_HEADERS = {"Accept": "application/pdf, */*;q=0.8"}
REFUSED = (403, 406, 415)
RETRYABLE = (429, 500, 502, 503, 504)
S2_URL = "https://api.semanticscholar.org/graph/v1/paper"
S2_FIELDS = "title,abstract,authors,externalIds,publicationDate,year"
OPENALEX_URL = "https://api.openalex.org/works"
OPENALEX_ARXIV = "S4306400194"
OPENALEX_SELECT = "id,display_name,publication_date,authorships,abstract_inverted_index,locations,doi"
ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(.+)$", re.I)
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"

NEW_STYLE_ID = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(v\d+)?(?!\d)")
OLD_STYLE_ID = re.compile(r"(?<![\w.-])([a-z]+(?:-[a-z]+)?(?:\.[A-Z]{2})?/\d{7})(v\d+)?(?!\d)")


class ArxivError(RuntimeError):
    pass


@dataclass
class Paper:
    arxiv_id: str
    version: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    updated: str
    pdf_url: str
    abs_url: str
    primary_category: str
    categories: list[str] = field(default_factory=list)
    source: str = "arXiv"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Paper":
        return cls(**data)


def extract_arxiv_id(text: str) -> str | None:
    text = text.strip()
    for pattern in (NEW_STYLE_ID, OLD_STYLE_ID):
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _https(url: str) -> str:
    return url.replace("http://", "https://", 1)


def _paper_stub(arxiv_id: str, title: str, authors: list[str], abstract: str, published: str, source: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        version="",
        title=_clean(title),
        authors=authors,
        abstract=_clean(abstract),
        published=published,
        updated=published,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        primary_category="",
        categories=[],
        source=source,
    )


def paper_from_semantic_scholar(item: dict) -> Paper | None:
    arxiv_id = (item.get("externalIds") or {}).get("ArXiv")
    if not arxiv_id or not item.get("title"):
        return None
    published = item.get("publicationDate") or (f"{item['year']}-01-01" if item.get("year") else "")
    authors = [a.get("name", "") for a in item.get("authors") or [] if a.get("name")]
    return _paper_stub(arxiv_id, item["title"], authors, item.get("abstract") or "", published, "Semantic Scholar")


def abstract_from_inverted_index(index: dict | None) -> str:
    if not index:
        return ""
    positions = sorted((pos, word) for word, spots in index.items() for pos in spots)
    return " ".join(word for _, word in positions)


def paper_from_openalex(work: dict) -> Paper | None:
    arxiv_id = None
    doi_match = ARXIV_DOI.search(work.get("doi") or "")
    if doi_match:
        arxiv_id = extract_arxiv_id(doi_match.group(1)) or doi_match.group(1)
    for location in work.get("locations") or []:
        if arxiv_id:
            break
        for url in (location.get("landing_page_url"), location.get("pdf_url")):
            if url and "arxiv.org" in url:
                arxiv_id = extract_arxiv_id(url.split("arxiv.org/", 1)[1])
                if arxiv_id:
                    break
    title = work.get("display_name") or work.get("title")
    if not arxiv_id or not title:
        return None
    authors = [
        a["author"]["display_name"]
        for a in work.get("authorships") or []
        if (a.get("author") or {}).get("display_name")
    ]
    abstract = abstract_from_inverted_index(work.get("abstract_inverted_index"))
    return _paper_stub(arxiv_id, title, authors, abstract, work.get("publication_date") or "", "OpenAlex")


def parse_feed(xml_text: str) -> list[Paper]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ArxivError("arXiv returned a response that is not valid XML") from exc
    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = entry.findtext(f"{ATOM}id", "")
        if "/api/errors" in raw_id:
            continue
        tail = raw_id.split("/abs/")[-1]
        version_match = re.search(r"(v\d+)$", tail)
        version = version_match.group(1) if version_match else ""
        arxiv_id = tail[: -len(version)] if version else tail
        pdf_url = ""
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        for link in entry.findall(f"{ATOM}link"):
            if link.get("title") == "pdf":
                pdf_url = _https(link.get("href", ""))
            elif link.get("rel") == "alternate":
                abs_url = _https(link.get("href", abs_url))
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}{version}"
        primary = entry.find(f"{ARXIV}primary_category")
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                version=version,
                title=_clean(entry.findtext(f"{ATOM}title")),
                authors=[_clean(a.findtext(f"{ATOM}name")) for a in entry.findall(f"{ATOM}author")],
                abstract=_clean(entry.findtext(f"{ATOM}summary")),
                published=entry.findtext(f"{ATOM}published", "")[:10],
                updated=entry.findtext(f"{ATOM}updated", "")[:10],
                pdf_url=pdf_url,
                abs_url=abs_url,
                primary_category=primary.get("term", "") if primary is not None else "",
                categories=[c.get("term", "") for c in entry.findall(f"{ATOM}category")],
            )
        )
    return papers


class ArxivClient:
    def __init__(
        self,
        min_interval: float = 3.0,
        timeout: float = 30.0,
        attempts: int = 3,
        transport: httpx.BaseTransport | None = None,
    ):
        self.min_interval = min_interval
        self.attempts = attempts
        self.http = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
            headers={"User-Agent": "arxiv-digest-agent/0.1", "Accept-Encoding": "gzip, deflate"},
        )
        self._last_call = 0.0

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last_call
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last_call = time.monotonic()

    def _get(self, url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            self._throttle()
            try:
                response = self.http.get(url, params=params, headers=headers)
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(3 * (attempt + 1))
                continue
            if response.status_code == 200:
                return response
            if response.status_code in REFUSED and b"<feed" in response.content[:2000]:
                return response
            last_error = ArxivError(f"{url} answered HTTP {response.status_code}")
            if response.status_code in REFUSED or response.status_code in RETRYABLE:
                time.sleep(4 * (attempt + 1))
                continue
            break
        raise ArxivError(str(last_error))

    def _query(self, params: dict) -> str:
        errors = []
        for url in API_URLS:
            try:
                return self._get(url, params, FEED_HEADERS).text
            except ArxivError as exc:
                errors.append(str(exc))
        raise ArxivError("arXiv API refused every attempt: " + " | ".join(errors))

    def _json(self, url: str, params: dict) -> dict | None:
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            try:
                response = self.http.get(url, params=params, headers={"Accept": "application/json"})
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(2 * (attempt + 1))
                continue
            if response.status_code == 200:
                return response.json()
            if response.status_code == 404:
                return None
            last_error = ArxivError(f"{url} answered HTTP {response.status_code}")
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(3 * (attempt + 1))
                continue
            break
        raise ArxivError(str(last_error))

    def _openalex_params(self, extra: dict) -> dict:
        params = {"select": OPENALEX_SELECT, **extra}
        key = os.getenv("OPENALEX_API_KEY")
        if key:
            params["api_key"] = key
        return params

    def _semantic_scholar_by_id(self, arxiv_id: str) -> Paper | None:
        data = self._json(f"{S2_URL}/arXiv:{arxiv_id}", {"fields": S2_FIELDS})
        return paper_from_semantic_scholar(data) if data else None

    def _openalex_by_id(self, arxiv_id: str) -> Paper | None:
        params = self._openalex_params({"filter": f"doi:https://doi.org/10.48550/arxiv.{arxiv_id}", "per_page": 1})
        data = self._json(OPENALEX_URL, params) or {}
        papers = [p for p in (paper_from_openalex(w) for w in data.get("results", [])) if p]
        return papers[0] if papers else None

    def _semantic_scholar_search(self, text: str, max_results: int) -> list[Paper]:
        params = {"query": text, "fields": S2_FIELDS, "limit": min(100, max_results * 2)}
        data = self._json(f"{S2_URL}/search", params) or {}
        papers = [p for p in (paper_from_semantic_scholar(i) for i in data.get("data") or []) if p]
        return papers[:max_results]

    def _openalex_search(self, text: str, max_results: int) -> list[Paper]:
        params = self._openalex_params(
            {"search": text, "filter": f"locations.source.id:{OPENALEX_ARXIV}", "per_page": min(100, max_results)}
        )
        data = self._json(OPENALEX_URL, params) or {}
        return [p for p in (paper_from_openalex(w) for w in data.get("results") or []) if p]

    def fetch_by_id(self, arxiv_id: str) -> Paper | None:
        try:
            papers = parse_feed(self._query({"id_list": arxiv_id, "max_results": 1}))
            return papers[0] if papers else None
        except ArxivError as exc:
            refusal = exc
        failures = [str(refusal)]
        for lookup in (self._semantic_scholar_by_id, self._openalex_by_id):
            try:
                paper = lookup(arxiv_id)
            except (ArxivError, ValueError, KeyError, TypeError) as exc:
                failures.append(f"{lookup.__name__}: {exc}")
                continue
            if paper:
                return paper
        raise ArxivError(" | ".join(failures))

    def search(
        self, query: str, max_results: int = 25, sort_by: str = "relevance", fallback_text: str | None = None
    ) -> list[Paper]:
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        try:
            return parse_feed(self._query(params))
        except ArxivError as exc:
            refusal = exc
        if not fallback_text:
            raise refusal
        failures = [str(refusal)]
        for finder in (self._semantic_scholar_search, self._openalex_search):
            try:
                papers = finder(fallback_text, max_results)
            except (ArxivError, ValueError, KeyError, TypeError) as exc:
                failures.append(f"{finder.__name__}: {exc}")
                continue
            if papers:
                return papers
        raise ArxivError(" | ".join(failures))

    def download_pdf(self, paper: Paper, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / f"{paper.arxiv_id.replace('/', '_')}{paper.version}.pdf"
        if target.exists() and target.stat().st_size > 0:
            return target
        response = self._get(paper.pdf_url, headers=PDF_HEADERS)
        if not response.content.startswith(b"%PDF"):
            raise ArxivError("download did not return a PDF (the paper may be withdrawn or HTML-only)")
        target.write_bytes(response.content)
        return target
