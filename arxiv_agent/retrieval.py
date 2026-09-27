import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .store import PaperIndex

STOPWORDS = set(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it
    its of on or our paper that the their them then there these they this those to was we were what when where
    which while who why will with would you your about does do authors author they them paper papers""".split()
)
REFERENCE_HINTS = re.compile(r"\b(cite|cites|cited|citation|references?|prior work|related work|bibliography)\b", re.I)
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower()) if t not in STOPWORDS]


@dataclass
class Hit:
    chunk: Chunk
    dense: float
    lexical: float
    fused: float


class HybridRetriever:
    def __init__(self, arxiv_id: str, chunks: list[Chunk], index: PaperIndex):
        self.arxiv_id = arxiv_id
        self.index = index
        self.chunks = {c.chunk_id: c for c in chunks}
        self.ordered = chunks
        corpus = [tokenize(c.embed_text) or ["_"] for c in chunks]
        self.bm25 = BM25Okapi(corpus) if chunks else None

    @staticmethod
    def wants_references(query: str) -> bool:
        return bool(REFERENCE_HINTS.search(query))

    def search(self, query: str, k: int = 6, pool: int = 20, include_references: bool | None = None) -> list[Hit]:
        if not self.ordered:
            return []
        if include_references is None:
            include_references = self.wants_references(query)
        allowed = {cid for cid, c in self.chunks.items() if include_references or c.kind != "references"}

        dense = [(cid, s) for cid, s in self.index.search(self.arxiv_id, query, pool, include_references) if cid in allowed]
        dense_scores = dict(dense)

        lexical_scores: dict[str, float] = {}
        tokens = tokenize(query)
        if tokens and self.bm25 is not None:
            scores = self.bm25.get_scores(tokens)
            ranked = sorted(
                ((c.chunk_id, float(score)) for c, score in zip(self.ordered, scores) if c.chunk_id in allowed),
                key=lambda pair: pair[1],
                reverse=True,
            )
            lexical_scores = {cid: score for cid, score in ranked[:pool] if score > 0}

        fused: dict[str, float] = {}
        for rank, (cid, _) in enumerate(dense):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, cid in enumerate(sorted(lexical_scores, key=lexical_scores.get, reverse=True)):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            Hit(self.chunks[cid], dense_scores.get(cid, 0.0), lexical_scores.get(cid, 0.0), fused[cid])
            for cid in top
        ]
