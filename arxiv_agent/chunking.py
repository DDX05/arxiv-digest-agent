import re
from dataclasses import asdict, dataclass

from .pdf_parser import Section

SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"])")
SKIP_KINDS = {"front"}


@dataclass
class Chunk:
    chunk_id: str
    text: str
    section: str
    kind: str
    page: int
    order: int

    @property
    def embed_text(self) -> str:
        return f"Section: {self.section}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def _units(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    units, current = [], ""
    for sentence in SENTENCE_BREAK.split(text):
        if len(sentence) > limit:
            if current:
                units.append(current)
                current = ""
            units.extend(sentence[i : i + limit] for i in range(0, len(sentence), limit))
        elif len(current) + len(sentence) + 1 > limit and current:
            units.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        units.append(current)
    return units


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return ""
    sentences = SENTENCE_BREAK.split(text)
    tail = ""
    for sentence in reversed(sentences):
        if len(tail) + len(sentence) > overlap:
            break
        tail = f"{sentence} {tail}".strip()
    return tail or text[-overlap:].split(" ", 1)[-1]


def chunk_sections(sections: list[Section], target_chars: int = 1000, overlap_chars: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(body: str, section: Section, page: int) -> None:
        body = body.strip()
        if len(body) < 40:
            return
        order = len(chunks)
        chunks.append(Chunk(f"c{order:04d}", body, section.title, section.kind, page, order))

    for section in sections:
        if section.kind in SKIP_KINDS:
            continue
        limit = target_chars * 2 if section.kind == "references" else target_chars
        buffer, buffer_page = "", None
        for paragraph in section.paragraphs:
            for unit in _units(paragraph.text, limit):
                if buffer_page is None:
                    buffer_page = paragraph.page
                candidate = f"{buffer}\n\n{unit}" if buffer else unit
                if len(candidate) > limit and buffer:
                    emit(buffer, section, buffer_page)
                    carried = _tail(buffer, overlap_chars) if section.kind != "references" else ""
                    buffer = f"{carried} {unit}".strip()
                    buffer_page = paragraph.page
                else:
                    buffer = candidate
        if buffer:
            emit(buffer, section, buffer_page or 1)
    return chunks
