from dataclasses import dataclass, field

from .arxiv_client import ArxivClient
from .chunking import Chunk
from .config import Settings
from .embeddings import Embedder, SentenceTransformerEmbedder
from .llm import LLM
from .retrieval import HybridRetriever
from .session import SessionStore
from .store import PaperIndex


@dataclass
class Deps:
    settings: Settings
    arxiv: ArxivClient
    embedder: Embedder
    index: PaperIndex
    sessions: SessionStore
    llm_factory: object
    _llm: object = None
    _retrievers: dict = field(default_factory=dict)

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = self.llm_factory()
        return self._llm

    def retriever(self, arxiv_id: str, chunks: list[Chunk]) -> HybridRetriever:
        key = (arxiv_id, len(chunks))
        if key not in self._retrievers:
            self._retrievers[key] = HybridRetriever(arxiv_id, chunks, self.index)
        return self._retrievers[key]


def build_deps(settings: Settings) -> Deps:
    embedder = SentenceTransformerEmbedder(settings.embed_model)
    return Deps(
        settings=settings,
        arxiv=ArxivClient(),
        embedder=embedder,
        index=PaperIndex(settings.chroma_dir, embedder),
        sessions=SessionStore(settings.session_dir),
        llm_factory=lambda: LLM(settings),
    )
