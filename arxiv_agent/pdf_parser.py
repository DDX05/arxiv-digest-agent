import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "related works", "background", "preliminaries",
    "method", "methods", "methodology", "approach", "our approach", "model", "framework",
    "experiments", "experimental setup", "experimental results", "evaluation", "results",
    "analysis", "discussion", "limitations", "limitation", "conclusion", "conclusions",
    "conclusion and future work", "future work", "broader impact", "ethics statement",
    "acknowledgments", "acknowledgements", "acknowledgment", "references", "bibliography", "appendix",
}
REFERENCE_HEADINGS = {"references", "bibliography"}
NUMBERED = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.)\s+[A-Z].{1,90}$")
LETTERED = re.compile(r"^[A-H](?:\.\d+)*\.?\s+[A-Z].{1,90}$")
WATERMARK = re.compile(r"^arXiv:\d{4}\.\d{4,5}")
PAGE_NUMBER = re.compile(r"^\d{1,3}$")
BOLD_FONT = re.compile(r"Bold|BX|Bd|Black|Heavy|Semibold", re.I)
REFERENCE_START = re.compile(r"^(\[\d+\]|\d{1,3}\.\s|\[[A-Za-z+]+\d{2,4}[a-z]?\])")
MIN_CHARS_PER_PAGE = 300


class PDFParseError(RuntimeError):
    pass


@dataclass
class Paragraph:
    text: str
    page: int


@dataclass
class Section:
    title: str
    kind: str
    paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Section":
        return cls(data["title"], data["kind"], [Paragraph(**p) for p in data["paragraphs"]])


@dataclass
class ParsedPaper:
    sections: list[Section]
    references: list[str]
    page_count: int
    pages_read: int
    char_count: int
    degraded: bool
    notes: list[str]

    def report(self) -> dict:
        return {
            "page_count": self.page_count,
            "pages_read": self.pages_read,
            "char_count": self.char_count,
            "section_titles": [s.title for s in self.sections],
            "reference_count": len(self.references),
            "degraded": self.degraded,
            "notes": self.notes,
        }


@dataclass
class _Block:
    text: str
    page: int
    max_size: float
    all_bold: bool
    line_count: int


def _normalize_heading(text: str) -> str:
    stripped = re.sub(r"^(?:\d+(?:\.\d+)*|[A-H](?:\.\d+)*|[IVX]+)\.?\s+", "", text)
    return re.sub(r"[^a-z ]", "", stripped.lower()).strip()


def _read_blocks(doc: pymupdf.Document, pages: int) -> tuple[list[_Block], float]:
    blocks: list[_Block] = []
    size_weight: Counter = Counter()
    for page_index in range(pages):
        page = doc[page_index]
        data = page.get_text("dict", flags=pymupdf.TEXT_DEHYPHENATE | pymupdf.TEXT_PRESERVE_WHITESPACE)
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            lines, sizes, bold_flags = [], [], []
            for line in block.get("lines", []):
                if tuple(round(v) for v in line.get("dir", (1, 0))) != (1, 0):
                    continue
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                lines.append(text)
                for span in spans:
                    if not span.get("text", "").strip():
                        continue
                    sizes.append(span["size"])
                    bold_flags.append(bool(span["flags"] & 16) or bool(BOLD_FONT.search(span.get("font", ""))))
                    size_weight[round(span["size"], 1)] += len(span["text"])
            if not lines:
                continue
            blocks.append(_Block(_join_lines(lines), page_index + 1, max(sizes), all(bold_flags), len(lines)))
    body_size = size_weight.most_common(1)[0][0] if size_weight else 10.0
    return blocks, body_size


def _join_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        if text.endswith("-") and len(text) > 1 and text[-2].isalpha() and line[:1].islower():
            text = text[:-1] + line
        else:
            text = f"{text} {line}" if text else line
    return re.sub(r"\s+", " ", text).strip()


def _split_references(paragraphs: list[Paragraph]) -> list[str]:
    texts = [p.text for p in paragraphs]
    numbered = sum(1 for t in texts if REFERENCE_START.match(t))
    if numbered >= 2 and numbered >= len(texts) // 3:
        entries: list[str] = []
        for text in texts:
            if REFERENCE_START.match(text) or not entries:
                entries.append(text)
            else:
                entries[-1] = f"{entries[-1]} {text}"
    else:
        entries = texts
    return [e for e in entries if len(e) > 20]


def _drop_furniture(blocks: list[_Block], pages: int) -> list[_Block]:
    if pages < 4:
        return [b for b in blocks if not WATERMARK.match(b.text) and not PAGE_NUMBER.match(b.text)]
    signature = Counter()
    for block in blocks:
        if len(block.text) < 80:
            signature[re.sub(r"\d+", "#", block.text.lower())] += 1
    repeated = {sig for sig, count in signature.items() if count >= max(3, pages * 0.4)}
    kept = []
    for block in blocks:
        if WATERMARK.match(block.text) or PAGE_NUMBER.match(block.text):
            continue
        if len(block.text) < 80 and re.sub(r"\d+", "#", block.text.lower()) in repeated:
            continue
        kept.append(block)
    return kept


def _is_heading(block: _Block, body_size: float, in_back_matter: bool) -> bool:
    text = block.text
    if block.line_count > 2 or len(text) > 100 or len(text) < 3:
        return False
    if text.endswith((".", ",", ";", ":")):
        return False
    numbered = bool(NUMBERED.match(text)) or (in_back_matter and bool(LETTERED.match(text)))
    normalized = _normalize_heading(text)
    emphasized = block.max_size >= body_size + 0.8 or block.all_bold
    if normalized in KNOWN_HEADINGS or normalized.startswith("appendix"):
        return emphasized or text.isupper() or numbered
    return numbered and emphasized


def _section_kind(title: str, current_kind: str) -> str:
    normalized = _normalize_heading(title)
    if normalized in REFERENCE_HEADINGS:
        return "references"
    if normalized.startswith("appendix"):
        return "appendix"
    if current_kind in ("references", "appendix"):
        return "appendix"
    if normalized == "abstract":
        return "abstract"
    return "body"


def parse_pdf(path: Path, max_pages: int = 60) -> ParsedPaper:
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise PDFParseError(f"could not open PDF: {exc}") from exc
    with doc:
        if doc.page_count == 0:
            raise PDFParseError("PDF has no pages")
        pages = min(doc.page_count, max_pages)
        blocks, body_size = _read_blocks(doc, pages)
        page_count = doc.page_count

    notes: list[str] = []
    if page_count > max_pages:
        notes.append(f"Read the first {max_pages} of {page_count} pages; later pages were skipped.")
    blocks = _drop_furniture(blocks, pages)
    char_count = sum(len(b.text) for b in blocks)
    degraded = char_count / pages < MIN_CHARS_PER_PAGE
    if degraded:
        notes.append("Very little extractable text; the PDF is probably scanned or image-based.")

    sections: list[Section] = [Section("Front matter", "front")]
    top_level: dict[str, str] = {}
    for block in blocks:
        in_back_matter = sections[-1].kind in ("references", "appendix")
        if _is_heading(block, body_size, in_back_matter):
            title = block.text
            number = re.match(r"^(\d+)(\.\d+)*\.?\s", title)
            if number and not number.group(2):
                top_level[number.group(1)] = title
            elif number and number.group(1) in top_level:
                title = f"{top_level[number.group(1)]} > {title}"
            sections.append(Section(title, _section_kind(block.text, sections[-1].kind)))
        else:
            sections[-1].paragraphs.append(Paragraph(block.text, block.page))
    sections = [s for s in sections if s.paragraphs]

    if len([s for s in sections if s.kind == "body"]) == 0 and not degraded:
        notes.append("No section headings detected; the paper is treated as one long section.")
        for section in sections:
            if section.kind == "front":
                section.kind = "body"
                section.title = "Full text"

    references = _split_references([p for s in sections if s.kind == "references" for p in s.paragraphs])
    return ParsedPaper(sections, references, page_count, pages, char_count, degraded, notes)
