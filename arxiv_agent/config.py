import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "default_model": "qwen2.5:7b-instruct",
    },
}


@dataclass(frozen=True)
class Settings:
    provider: str
    base_url: str
    api_key: str | None
    model: str
    embed_model: str
    data_dir: Path
    output_dir: Path
    max_pdf_pages: int
    summary_char_budget: int
    chunk_chars: int
    chunk_overlap: int
    arxiv_max_results: int
    min_similarity: float
    qa_top_k: int

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def session_dir(self) -> Path:
        return self.data_dir / "sessions"


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def load_settings() -> Settings:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDERS)}, got '{provider}'")
    spec = PROVIDERS[provider]
    key_env = spec["key_env"]
    return Settings(
        provider=provider,
        base_url=os.getenv("LLM_BASE_URL", spec["base_url"]),
        api_key=os.getenv(key_env) if key_env else None,
        model=os.getenv("LLM_MODEL", spec["default_model"]),
        embed_model=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        data_dir=Path(os.getenv("DATA_DIR", ".data")),
        output_dir=Path(os.getenv("OUTPUT_DIR", "outputs")),
        max_pdf_pages=_int("MAX_PDF_PAGES", 60),
        summary_char_budget=_int("SUMMARY_CHAR_BUDGET", 18000),
        chunk_chars=_int("CHUNK_CHARS", 1000),
        chunk_overlap=_int("CHUNK_OVERLAP", 150),
        arxiv_max_results=_int("ARXIV_MAX_RESULTS", 25),
        min_similarity=_float("MIN_SIMILARITY", 0.28),
        qa_top_k=_int("QA_TOP_K", 6),
    )
