import re

FILLER = set(
    """a an and are about any for from in into is of on or the to with what which how recent latest new
    current work works paper papers research study studies approach approaches method methods using based
    find show me some state art sota survey llm llms model models""".split()
)
RECENCY = re.compile(r"\b(recent|latest|new|newest|current|this year|2025|2026)\b", re.I)
CATEGORY = re.compile(r"^[a-z\-]+(?:\.[A-Za-z\-]+)?$")
MAX_LEVEL = 2


def fallback_plan(topic: str) -> dict:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", topic)
    keywords = [w.lower() for w in words if w.lower() not in FILLER and len(w) > 2]
    return {"keywords": keywords[:4] or [topic.strip()], "categories": [], "prefer_recent": bool(RECENCY.search(topic))}


def clean_plan(raw: dict, topic: str) -> dict:
    keywords = [str(k).strip().strip('"') for k in raw.get("keywords", []) if str(k).strip()]
    categories = [str(c).strip() for c in raw.get("categories", []) if CATEGORY.match(str(c).strip())]
    if not keywords:
        return fallback_plan(topic)
    return {
        "keywords": keywords[:4],
        "categories": categories[:3],
        "prefer_recent": bool(raw.get("prefer_recent")) or bool(RECENCY.search(topic)),
    }


def _term(field: str, phrase: str) -> str:
    phrase = phrase.replace('"', "")
    return f'{field}:"{phrase}"' if " " in phrase else f"{field}:{phrase}"


def build_search_query(keywords: list[str], categories: list[str], level: int) -> str:
    if level == 0:
        query = " AND ".join(_term("abs", k) for k in keywords)
        if categories:
            cats = " OR ".join(f"cat:{c}" for c in categories)
            query = f"({query}) AND ({cats})"
        return query
    if level == 1:
        return " AND ".join(_term("all", k) for k in keywords[:2])
    words = sorted({w for k in keywords for w in k.split() if w.lower() not in FILLER})
    return " OR ".join(f"all:{w}" for w in words) or " OR ".join(_term("all", k) for k in keywords)
