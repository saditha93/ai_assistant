from app.auth import User
from app.retrieval.bm25 import BM25, tokenize
from app.retrieval.documents import catalog, load_corpus, split_sections
from app.retrieval.search import build_filter, hybrid_search, matches_filter

VIEWER = User("v", "Viewer", "viewer", "retail")
ADMIN = User("a", "Admin", "admin", "platform")


def test_split_sections_by_heading():
    body = "Intro line\n\n## Summary\nshort\n\n## Root Cause\nthe pool ran out"
    assert split_sections(body) == [("Overview", "Intro line"), ("Summary", "short"), ("Root Cause", "the pool ran out")]


def test_corpus_chunks_have_attribution_metadata():
    chunks = load_corpus()
    assert len({c["doc_id"] for c in chunks}) >= 25
    for c in chunks:
        assert c["id"].startswith(c["doc_id"] + "#")
        assert {"title", "section", "department", "document_type", "access_level", "created_date"} <= set(c)
    assert all(d["sections"] for d in catalog(list(chunks)))


def test_bm25_ranks_relevant_text_first():
    texts = ["TLS certificate expired on the gateway", "database connection pool exhausted", "holiday calendar"]
    ranking = BM25(texts).search("connection pools exhausted", k=3)
    assert ranking[0][0] == 1
    assert tokenize("Payments failures") == ["payment", "failure"]


def test_sparse_vectors_are_pinecone_shaped():
    bm25 = BM25(["alpha beta", "beta gamma"])
    vec = bm25.encode_query("beta gamma")
    assert len(vec["indices"]) == len(vec["values"]) == 2
    assert abs(sum(vec["values"]) - 1) < 1e-6
    assert all(0 <= i < 2**32 for i in vec["indices"])


def test_access_filter_follows_role():
    restricted = next(c for c in load_corpus() if c["access_level"] == "restricted")
    ns = [restricted["namespace"]]
    assert not matches_filter(restricted, VIEWER, ns, {})
    assert matches_filter(restricted, ADMIN, ns, {})
    assert build_filter(VIEWER, {})["$and"][0] == {"access_level": {"$in": ["public", "internal"]}}


def test_date_and_type_filters():
    f = build_filter(ADMIN, {"document_types": ["incident"], "since": "2025-09-26"})
    assert {"document_type": {"$in": ["incident"]}} in f["$and"]
    old = next(c for c in load_corpus() if c["doc_id"] == "INC-2025-019")
    assert not matches_filter(old, ADMIN, ["engineering"], {"since": "2025-09-26"})


async def test_search_falls_back_to_keywords_without_pinecone():
    result = await hybrid_search("Key management HSM design", VIEWER)
    assert result["mode"] == "keyword_fallback" and result["notes"]
    assert all(h["access_level"] in ("public", "internal") for h in result["hits"])
    assert not any(h["doc_id"] == "ARCH-004" for h in result["hits"])  # restricted doc


async def test_planted_injection_is_quarantined():
    result = await hybrid_search("NorthGate vendor sync notes maintenance mode status badge", ADMIN)
    hit = next(h for h in result["hits"] if h["doc_id"] == "MTG-2026-011" and h.get("quarantined"))
    assert "ignore all previous instructions" not in hit["text"].lower()
    assert "evil-analytics" not in hit["text"]
    assert result["injection_flags"]
