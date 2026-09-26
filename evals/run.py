"""Retrieval evaluation: recall@5 and MRR over evals/golden.yaml.

    uv run python -m evals.run

Always measures local BM25. With Pinecone and Gemini keys it also measures hybrid search
with and without the reranker, so the value of each layer is visible.
"""

import asyncio
from pathlib import Path

import yaml

from app.auth import User
from app.config import settings
from app.retrieval import search

ADMIN = User("eval", "Eval", "admin", "platform")  # sees every document
K = 5


def score(ranked_doc_ids: list[str], expected: list[str]) -> tuple[float, float]:
    docs = list(dict.fromkeys(ranked_doc_ids))  # chunk hits -> distinct documents, in order
    hit = any(d in expected for d in docs[:K])
    rank = next((i + 1 for i, d in enumerate(docs) if d in expected), None)
    return float(hit), 1 / rank if rank else 0.0


async def main() -> None:
    golden = yaml.safe_load((Path(__file__).parent / "golden.yaml").read_text())
    namespaces = search.NAMESPACES

    async def bm25(q):
        return [h["doc_id"] for h in search.keyword_search(q, ADMIN, namespaces, {}, 20)]

    async def hybrid(q):
        hits = await search._pinecone_search(q, namespaces, search.build_filter(ADMIN, {}), 20)
        return [h["doc_id"] for h in hits]

    async def hybrid_rerank(q):
        return [h["doc_id"] for h in (await search.hybrid_search(q, ADMIN, top_k=10))["hits"]]

    systems = {"bm25": bm25}
    if settings.has_pinecone and settings.has_llm:
        systems |= {"hybrid": hybrid, "hybrid+rerank": hybrid_rerank}
    else:
        print("Pinecone/Gemini keys not set: evaluating BM25 only.\n")

    print(f"{'system':<15} recall@{K}   MRR")
    for name, fn in systems.items():
        recalls, rrs = [], []
        for item in golden:
            r, rr = score(await fn(item["question"]), item["expected"])
            recalls.append(r)
            rrs.append(rr)
        print(f"{name:<15} {sum(recalls) / len(recalls):>8.2f}   {sum(rrs) / len(rrs):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
