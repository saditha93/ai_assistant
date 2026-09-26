"""A small BM25 implementation.

We use it for three things:
  * sparse vectors for Pinecone hybrid search (encode_doc / encode_query),
  * local keyword search when Pinecone is unavailable,
  * ranking a user's past questions for long-term memory.

Pinecone needs integer token ids, so tokens are hashed with crc32 (stable across
processes, unlike Python's hash()). Query weights are IDF values normalised to sum
to 1, which keeps sparse scores in a similar range to cosine similarity.
"""

import math
import re
import zlib
from collections import Counter

STOPWORDS = set(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was were "
    "what when where which who why will with do does did we you our your can all any about into last "
    "me my show tell give list".split()
)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    # ponytail: plural stripping only; swap in a real stemmer if recall on verb forms matters
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _token_id(token: str) -> int:
    return zlib.crc32(token.encode())


def _to_sparse(weights: dict[str, float]) -> dict:
    merged: dict[int, float] = {}
    for token, w in weights.items():
        idx = _token_id(token)
        merged[idx] = merged.get(idx, 0.0) + w
    return {"indices": list(merged), "values": [round(v, 6) for v in merged.values()]}


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        counts = [Counter(tokenize(t)) for t in texts]
        lengths = [sum(c.values()) for c in counts]
        self.avg_len = (sum(lengths) / len(lengths)) if lengths else 1.0
        n = len(counts)
        df = Counter(tok for c in counts for tok in c)
        self.idf = {tok: math.log(1 + (n - f + 0.5) / (f + 0.5)) for tok, f in df.items()}
        self.doc_weights = [self._tf_weights(c, length) for c, length in zip(counts, lengths, strict=True)]

    def _tf_weights(self, counts: Counter, length: int) -> dict[str, float]:
        norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
        return {tok: tf * (self.k1 + 1) / (tf + norm) for tok, tf in counts.items()}

    def query_weights(self, query: str) -> dict[str, float]:
        weights = {t: self.idf[t] for t in set(tokenize(query)) if t in self.idf}
        total = sum(weights.values()) or 1.0
        return {t: w / total for t, w in weights.items()}

    def search(self, query: str, k: int = 10, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        """Return (position, score) for the best k texts. `allowed` restricts to some positions."""
        q = self.query_weights(query)
        scored = []
        for i, doc in enumerate(self.doc_weights):
            if allowed is not None and i not in allowed:
                continue
            score = sum(w * doc.get(t, 0.0) for t, w in q.items())
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda s: s[1], reverse=True)
        return scored[:k]

    def encode_doc(self, text: str) -> dict:
        tokens = tokenize(text)
        return _to_sparse(self._tf_weights(Counter(tokens), len(tokens)))

    def encode_query(self, query: str) -> dict:
        return _to_sparse(self.query_weights(query))
