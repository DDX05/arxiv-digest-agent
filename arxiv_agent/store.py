import re
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from .chunking import Chunk
from .embeddings import Embedder


def collection_name(arxiv_id: str) -> str:
    return "paper_" + re.sub(r"[^A-Za-z0-9]", "_", arxiv_id)


class PaperIndex:
    def __init__(self, persist_dir: Path, embedder: Embedder):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )

    def _collection(self, arxiv_id: str):
        return self.client.get_or_create_collection(
            collection_name(arxiv_id), embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def build(self, arxiv_id: str, chunks: list[Chunk], batch_size: int = 64) -> str:
        name = collection_name(arxiv_id)
        if name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(name)
        collection = self._collection(arxiv_id)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=self.embedder.embed([c.embed_text for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[{"section": c.section, "kind": c.kind, "page": c.page, "order": c.order} for c in batch],
            )
        return name

    def count(self, arxiv_id: str) -> int:
        name = collection_name(arxiv_id)
        if name not in [c.name for c in self.client.list_collections()]:
            return 0
        return self._collection(arxiv_id).count()

    def search(self, arxiv_id: str, query: str, k: int, include_references: bool = False) -> list[tuple[str, float]]:
        collection = self._collection(arxiv_id)
        total = collection.count()
        if total == 0:
            return []
        where = None if include_references else {"kind": {"$ne": "references"}}
        result = collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=min(k, total),
            where=where,
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [(chunk_id, 1.0 - distance) for chunk_id, distance in zip(ids, distances)]
