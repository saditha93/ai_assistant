import asyncio
import datetime as dt
from functools import lru_cache

from langsmith import traceable
from pinecone import AsyncIndex, AsyncPinecone

from app import guardrails
from app.auth import User
from app.config import settings
from app.llm import embed_query
from app.observability import log
from app.resilience import DependencyDown, pinecone_breaker
from app.retrieval.bm25 import BM25
from app.retrieval.documents import NAMESPACES, index_text, load_corpus

DENSE_FIELD, SPARSE_FIELD = "_values", "_sparse_values"
METADATA_FIELDS = ("doc_id", "title", "section", "text", "document_type", "department",
                   "access_level", "created_date", "created_ts", "owner", "namespace")


@lru_cache
def local_index() -> BM25:
    """BM25 fitted on the whole corpus. Also provides the IDF values for sparse vectors, so
    ingestion and querying always use the same vocabulary weights."""
    return BM25([index_text(c) for c in load_corpus()])


_pinecone: dict = {}


async def get_index() -> AsyncIndex:
    if "index" not in _pinecone:
        _pinecone["client"] = AsyncPinecone(api_key=settings.pinecone_api_key, timeout=10)
        _pinecone["index"] = await _pinecone["client"].index(settings.pinecone_index)
    return _pinecone["index"]


def build_filter(user: User, filters: dict) -> dict:
    clauses = [{"access_level": {"$in": user.access_levels}}]
    if filters.get("document_types"):
        clauses.append({"document_type": {"$in": filters["document_types"]}})
    if filters.get("departments"):
        clauses.append({"department": {"$in": filters["departments"]}})
    if filters.get("since"):
        since = dt.date.fromisoformat(filters["since"])
        clauses.append({"created_ts": {"$gte": int(dt.datetime(since.year, since.month, since.day).timestamp())}})
    return {"$and": clauses}


def matches_filter(chunk: dict, user: User, namespaces: list[str], filters: dict) -> bool:
    """The same rules as build_filter, for chunks we hold locally."""
    if chunk["access_level"] not in user.access_levels or chunk["namespace"] not in namespaces:
        return False
    if filters.get("document_types") and chunk["document_type"] not in filters["document_types"]:
        return False
    if filters.get("departments") and chunk["department"] not in filters["departments"]:
        return False
    return not (filters.get("since") and chunk["created_date"] < filters["since"])


def _sparse_dot(q: dict, d: dict | None) -> float:
    if not d:
        return 0.0
    weights = dict(zip(d["indices"], d["values"], strict=True))
    return sum(v * weights.get(i, 0.0) for i, v in zip(q["indices"], q["values"], strict=True))


async def _pinecone_search(query: str, namespaces: list[str], flt: dict, k: int) -> list[dict]:
    if not settings.has_pinecone:
        raise DependencyDown("Pinecone is not configured")
    alpha = settings.hybrid_alpha
    dense = await embed_query(query)
    sparse = local_index().encode_query(query)
    index = await get_index()

    async def one(ns: str):
        return await index.query(
            namespace=ns, top_k=k, filter=flt, include_metadata=True, include_values=True,
            vector=[v * alpha for v in dense],
            sparse_vector={"indices": sparse["indices"], "values": [v * (1 - alpha) for v in sparse["values"]]}
            if sparse["indices"] else None,
        )

    responses = await asyncio.gather(*(one(ns) for ns in namespaces))
    hits = []
    for response in responses:
        for m in response.matches:
            dense_score = sum(a * b for a, b in zip(dense, m.values or [], strict=False))
            sparse_values = m.sparse_values and {"indices": m.sparse_values.indices, "values": m.sparse_values.values}
            sparse_score = _sparse_dot(sparse, sparse_values)
            hits.append({**m.metadata, "id": m.id, "dense_score": round(dense_score, 4),
                         "sparse_score": round(sparse_score, 4), "score": round(m.score, 4)})
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


def keyword_search(query: str, user: User, namespaces: list[str], filters: dict, k: int) -> list[dict]:
    corpus = load_corpus()
    allowed = {i for i, c in enumerate(corpus) if matches_filter(c, user, namespaces, filters)}
    return [{**corpus[i], "dense_score": None, "sparse_score": round(s, 4), "score": round(s, 4)}
            for i, s in local_index().search(query, k, allowed)]


async def _rerank(query: str, hits: list[dict], top_n: int) -> list[dict]:
    client = _pinecone["client"]
    result = await client.inference.rerank(
        model=settings.rerank_model, query=query, top_n=top_n, return_documents=False,
        documents=[{"id": h["id"], "text": index_text(h)} for h in hits],
    )
    return [{**hits[r.index], "rerank_score": round(r.score, 4)} for r in result.data]


@traceable(run_type="retriever", name="hybrid_search")
async def hybrid_search(
    query: str, user: User, namespaces: list[str] | None = None, filters: dict | None = None,
    top_k: int | None = None,
) -> dict:
    top_k = top_k or settings.retrieval_top_k
    namespaces = [ns for ns in (namespaces or NAMESPACES) if ns in NAMESPACES] or NAMESPACES
    filters = filters or {}
    result = {"query": query, "namespaces": namespaces, "filters": filters, "mode": "hybrid",
              "reranked": False, "notes": [], "injection_flags": []}
    try:
        hits = await pinecone_breaker.call(
            _pinecone_search, query, namespaces, build_filter(user, filters), top_k * 3, timeout=20
        )
    except Exception as exc:
        log.warning("vector_search_failed", error=str(exc))
        result["mode"] = "keyword_fallback"
        result["notes"].append(f"Vector search unavailable ({exc}). Used local keyword search instead.")
        hits = keyword_search(query, user, namespaces, filters, top_k * 3)

    if result["mode"] == "hybrid" and len(hits) > 1:
        try:
            hits = await pinecone_breaker.call(_rerank, query, hits[:20], top_k, timeout=10)
            result["reranked"] = True
        except Exception as exc:
            result["notes"].append(f"Reranker unavailable ({exc}). Kept hybrid order.")

    clean_hits = []
    for hit in hits[:top_k]:
        hit, flags = guardrails.sanitize_chunk(hit)
        if flags:
            result["injection_flags"].append({"chunk": hit["id"], "labels": flags})
        clean_hits.append(hit)
    result["hits"] = clean_hits
    return result
