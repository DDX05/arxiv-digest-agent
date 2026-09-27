import json
from datetime import datetime, timezone
from pathlib import Path


def _safe(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


class SessionStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, arxiv_id: str) -> Path:
        return self.directory / f"{_safe(arxiv_id)}.json"

    def exists(self, arxiv_id: str) -> bool:
        return self.path(arxiv_id).exists()

    def load(self, arxiv_id: str) -> dict:
        path = self.path(arxiv_id)
        if not path.exists():
            raise FileNotFoundError(f"No saved session for {arxiv_id}. Run: python -m arxiv_agent brief {arxiv_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, session: dict) -> Path:
        session["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = self.path(session["paper"]["arxiv_id"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path

    def append_turn(self, arxiv_id: str, turn: dict) -> None:
        session = self.load(arxiv_id)
        session.setdefault("history", []).append(turn)
        self.save(session)

    def list(self) -> list[dict]:
        rows = []
        for path in sorted(self.directory.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"arxiv_id": data["paper"]["arxiv_id"], "title": data["paper"]["title"], "turns": len(data.get("history", []))})
        return rows
