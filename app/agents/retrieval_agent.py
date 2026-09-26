from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from app.agents.state import merge_evidence
from app.auth import User
from app.config import settings
from app.llm import llm, llm_available
from app.retrieval.search import hybrid_search

REWRITE_PROMPT = """Rewrite this search query so it matches how internal bank documents (runbooks, incident
reports, policies, architecture docs) would phrase it. Use specific technical terms and synonyms.
Return only the new query, nothing else.

Query: {query}"""


def _is_weak(result: dict) -> bool:
    if not result["hits"]:
        return True
    if result["reranked"]:
        return result["hits"][0].get("rerank_score", 0) < settings.min_rerank_score
    return False


def _event(result: dict, status: str) -> dict:
    return {
        "type": "retrieval", "status": status, "query": result["query"], "mode": result["mode"],
        "reranked": result["reranked"], "namespaces": result["namespaces"], "notes": result["notes"],
        "injection_flags": result["injection_flags"],
        "hits": [{k: h.get(k) for k in ("id", "title", "section", "access_level", "dense_score", "sparse_score",
                                        "score", "rerank_score")} for h in result["hits"]],
    }


async def retrieval_agent(state: dict, runtime: Runtime[User]) -> dict:
    user, emit = runtime.context, runtime.stream_writer
    plan = state["plan"]
    query = plan["standalone_question"]
    filters = {k: plan.get(k) for k in ("document_types", "departments", "since") if plan.get(k)}

    emit({"type": "retrieval", "status": "searching", "query": query, "namespaces": plan["namespaces"],
          "filters": filters})
    result = await hybrid_search(query, user, plan["namespaces"], filters)
    emit(_event(result, "done"))
    hits, notes = result["hits"], list(result["notes"])

    if _is_weak(result):
        new_query = query
        if llm_available():
            try:
                prompt = HumanMessage(REWRITE_PROMPT.format(query=query))
                reply = await llm(name="query_rewrite", light=True).ainvoke([prompt])
                new_query = reply.text.strip().strip('"')[:300] or query
            except Exception as exc:
                notes.append(f"Query rewrite failed ({str(exc)[:80]}).")
        emit({"type": "retrieval", "status": "retrying", "query": new_query,
              "reason": "weak results; rewriting the query and searching all namespaces without filters"})
        retry = await hybrid_search(new_query, user)
        emit(_event(retry, "done"))
        hits = merge_evidence(hits, retry["hits"])
        notes += [n for n in retry["notes"] if n not in notes]

    summary = {"mode": result["mode"], "reranked": result["reranked"], "hit_count": len(hits),
               "injection_flags": result["injection_flags"]}
    return {"evidence": hits, "retrieval": summary, "notes": notes}
