from arxiv_agent.query import build_search_query, clean_plan, fallback_plan


def test_fallback_plan_drops_filler_and_spots_recency():
    plan = fallback_plan("recent work on KV-cache compression for LLMs")
    assert plan["keywords"] == ["kv-cache", "compression"]
    assert plan["prefer_recent"] is True


def test_clean_plan_filters_bad_categories():
    plan = clean_plan({"keywords": ["kv cache", "compression"], "categories": ["cs.CL", "not a category!"]}, "x")
    assert plan["categories"] == ["cs.CL"]


def test_query_levels_get_broader():
    strict = build_search_query(["kv cache", "compression"], ["cs.CL"], 0)
    assert strict == '(abs:"kv cache" AND abs:compression) AND (cat:cs.CL)'
    assert build_search_query(["kv cache", "compression"], ["cs.CL"], 1) == 'all:"kv cache" AND all:compression'
    assert build_search_query(["kv cache", "compression"], [], 2) == "all:cache OR all:compression OR all:kv"
