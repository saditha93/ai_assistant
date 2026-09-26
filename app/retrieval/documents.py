import datetime as dt
import re
from functools import lru_cache
from pathlib import Path

import yaml

from app.config import settings

NAMESPACE_BY_TYPE = {
    "incident": "engineering",
    "architecture": "engineering",
    "runbook": "engineering",
    "policy": "governance",
    "product_spec": "product",
    "meeting_notes": "product",
}
NAMESPACES = sorted(set(NAMESPACE_BY_TYPE.values()))

MAX_CHUNK_CHARS = 1500


def parse_document(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    _, front, body = raw.split("---", 2)
    meta = yaml.safe_load(front)
    meta["created_date"] = str(meta["created_date"])
    return meta, body.strip()


def split_sections(body: str) -> list[tuple[str, str]]:
    """Return (heading, text) pairs. Text before the first heading is called 'Overview'."""
    sections = []
    for block in re.split(r"^## ", body, flags=re.MULTILINE):
        block = block.strip()
        if not block:
            continue
        heading, _, text = block.partition("\n")
        if text.strip():
            sections.append((heading.strip(), text.strip()))
        elif not sections:
            sections.append(("Overview", heading.strip()))
    return sections


def _split_long(text: str) -> list[str]:
    """Split an oversized section on paragraph boundaries."""
    parts, current = [], ""
    for para in text.split("\n\n"):
        if current and len(current) + len(para) > MAX_CHUNK_CHARS:
            parts.append(current)
            current = ""
        current = f"{current}\n\n{para}" if current else para
    return parts + [current] if current else parts


def chunk_document(meta: dict, body: str) -> list[dict]:
    created = dt.date.fromisoformat(meta["created_date"])
    base = {
        "doc_id": meta["doc_id"],
        "title": meta["title"],
        "document_type": meta["document_type"],
        "department": meta["department"],
        "access_level": meta["access_level"],
        "created_date": meta["created_date"],
        "created_ts": int(dt.datetime(created.year, created.month, created.day).timestamp()),
        "owner": meta.get("owner", ""),
        "namespace": NAMESPACE_BY_TYPE[meta["document_type"]],
    }
    chunks = []
    for heading, text in split_sections(body):
        for part in _split_long(text):
            chunks.append({**base, "id": f"{meta['doc_id']}#{len(chunks) + 1}", "section": heading, "text": part})
    return chunks


@lru_cache
def load_corpus(folder: Path | None = None) -> tuple[dict, ...]:
    folder = folder or settings.data_dir / "docs"
    chunks = []
    for path in sorted(folder.rglob("*.md")):
        chunks.extend(chunk_document(*parse_document(path)))
    return tuple(chunks)


def index_text(chunk: dict) -> str:
    """What we embed and keyword-index: title and section add useful context to short sections."""
    return f"{chunk['title']} | {chunk['section']}\n{chunk['text']}"


def catalog(chunks: list[dict]) -> list[dict]:
    """One line per document with its sections, without any content. The research agent
    looks at this first instead of reading documents."""
    docs: dict[str, dict] = {}
    for c in chunks:
        entry = docs.setdefault(
            c["doc_id"],
            {k: c[k] for k in ("doc_id", "title", "document_type", "department", "created_date")}
            | {"sections": []},
        )
        if c["section"] not in entry["sections"]:
            entry["sections"].append(c["section"])
    return list(docs.values())
