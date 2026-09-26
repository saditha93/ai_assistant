import argparse
import asyncio
import re
from collections import defaultdict

from pinecone import AsyncPinecone

from app.auth import User
from app.config import settings
from app.llm import embed_documents
from app.retrieval.documents import index_text, load_corpus
from app.retrieval.search import DENSE_FIELD, METADATA_FIELDS, SPARSE_FIELD, hybrid_search, local_index

# Free tier allows 100 embedded texts per minute, and each text counts as one request.
EMBED_BATCH = 20


async def embed_with_retry(texts: list[str], attempts: int = 6) -> list[list[float]]:
    for attempt in range(attempts):
        try:
            return await embed_documents(texts)
        except Exception as exc:
            if "RESOURCE_EXHAUSTED" not in str(exc) or attempt == attempts - 1:
                raise
            delay = re.search(r"retry in ([\d.]+)s", str(exc))
            wait = float(delay.group(1)) + 2 if delay else 30
            print(f"    embedding quota reached, waiting {wait:.0f}s")
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


async def ensure_index(pc: AsyncPinecone, recreate: bool) -> None:
    exists = await pc.has_index(settings.pinecone_index)
    if exists and recreate:
        print(f"Deleting index {settings.pinecone_index}")
        await pc.delete_index(settings.pinecone_index)
        exists = False
    if not exists:
        print(f"Creating hybrid index {settings.pinecone_index}")
        await pc.indexes.create(
            name=settings.pinecone_index,
            schema={"fields": {
                DENSE_FIELD: {"type": "dense_vector", "dimension": settings.embed_dim, "metric": "dotproduct"},
                SPARSE_FIELD: {"type": "sparse_vector"},
            }},
            deployment={"deployment_type": "managed", "cloud": "aws", "region": settings.pinecone_region},
        )


async def ingest(recreate: bool = False) -> None:
    if not (settings.has_pinecone and settings.has_llm):
        raise SystemExit("PINECONE_API_KEY and GOOGLE_API_KEY are both needed to ingest.")

    chunks = list(load_corpus())
    bm25 = local_index()
    by_namespace = defaultdict(list)
    for c in chunks:
        by_namespace[c["namespace"]].append(c)

    async with AsyncPinecone(api_key=settings.pinecone_api_key) as pc:
        await ensure_index(pc, recreate)
        index = await pc.index(settings.pinecone_index)
        for namespace, items in by_namespace.items():
            for start in range(0, len(items), EMBED_BATCH):
                batch = items[start:start + EMBED_BATCH]
                texts = [index_text(c) for c in batch]
                dense = await embed_with_retry(texts)
                records = [
                    {"id": c["id"], "values": vec, "sparse_values": bm25.encode_doc(text),
                     "metadata": {k: c[k] for k in METADATA_FIELDS}}
                    for c, vec, text in zip(batch, dense, texts, strict=True)
                ]
                await index.upsert(vectors=records, namespace=namespace, show_progress=False)
            print(f"  {namespace}: {len(items)} chunks")
        await index.close()
    print(f"Upserted {len(chunks)} chunks from {len({c['doc_id'] for c in chunks})} documents.")

    await asyncio.sleep(5)
    admin = User("ingest", "Ingest check", "admin", "platform")
    result = await hybrid_search("payment gateway timeout root cause", admin)
    print(f"Smoke query mode={result['mode']} reranked={result['reranked']}")
    for h in result["hits"][:3]:
        print(f"  {h['id']:<20} dense={h['dense_score']} sparse={h['sparse_score']} {h['title']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true")
    asyncio.run(ingest(parser.parse_args().recreate))
