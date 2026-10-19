"""
Keyword side of hybrid search: BM25 as a Qdrant sparse vector, computed locally.

How it works (the same idea as fastembed's "Qdrant/bm25", without downloading anything):

  document  ->  tokens  ->  one sparse dimension per token, value = BM25 term-frequency part
                              tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_len))
  query     ->  tokens  ->  value 1.0 per distinct token

The IDF half of BM25 is done by Qdrant itself: the collection's sparse vector is
created with `modifier=IDF`, so Qdrant keeps document frequencies per token and
multiplies them in at query time. That keeps IDF correct as documents are added
or deleted, with no corpus statistics stored here.

Why it matters for course Q&A: error messages, CLI flags, package names and ids
("ERR_CONNECTION_REFUSED", "pg_hba.conf", "--no-cache-dir") are exactly what dense
embeddings blur and keyword search nails.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from functools import lru_cache

K1 = 1.2
B = 0.75
AVG_LEN = 256.0

_TOKEN = re.compile(r"[a-z0-9]+(?:[._\-/][a-z0-9]+)*")

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her here
    hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off
    on once only or other our ours ourselves out over own same she should so some such than that the their theirs
    them themselves then there these they this those through to too under until up very was we were what when where
    which while who whom why will with would you your yours yourself yourselves hi hello thanks thank please""".split()
)


@lru_cache(maxsize=1)
def _stemmer():
    try:  # nltk ships with llama-index; PorterStemmer needs no data download
        from nltk.stem import PorterStemmer

        return PorterStemmer().stem
    except Exception:  # noqa: BLE001
        return lambda w: w


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Compound tokens (pg_hba.conf, gpt-4o-mini) are kept whole
    AND split into parts, so both 'pg_hba.conf' and 'hba' match."""
    stem = _stemmer()
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        parts = re.split(r"[._\-/]", tok)
        if len(parts) > 1:
            out.append(tok)
        for p in parts:
            if p and p not in STOPWORDS and (len(p) > 1 or p.isdigit()):
                out.append(stem(p) if p.isalpha() else p)
    return out


def token_id(token: str) -> int:
    """Stable 32-bit id (Python's hash() is salted per process, so it can't be used)."""
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big")


def _merge(pairs: dict[int, float]) -> tuple[list[int], list[float]]:
    items = sorted(pairs.items())
    return [i for i, _ in items], [v for _, v in items]


def encode_document(text: str) -> tuple[list[int], list[float]]:
    tokens = tokenize(text)
    if not tokens:
        return [], []
    doc_len = len(tokens)
    norm = K1 * (1 - B + B * doc_len / AVG_LEN)
    weights: dict[int, float] = {}
    for tok, tf in Counter(tokens).items():
        idx = token_id(tok)
        weights[idx] = weights.get(idx, 0.0) + tf * (K1 + 1) / (tf + norm)
    return _merge(weights)


def encode_query(text: str) -> tuple[list[int], list[float]]:
    return _merge({token_id(t): 1.0 for t in set(tokenize(text))})


# Signatures LlamaIndex's QdrantVectorStore expects for sparse_doc_fn / sparse_query_fn
def sparse_doc_fn(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    encoded = [encode_document(t) for t in texts]
    return [e[0] for e in encoded], [e[1] for e in encoded]


def sparse_query_fn(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    encoded = [encode_query(t) for t in texts]
    return [e[0] for e in encoded], [e[1] for e in encoded]


def fastembed_sparse_fns(model_name: str):
    """Alternative: a fastembed sparse model (Qdrant/bm25, prithivida/Splade_PP_en_v1, ...).
    Downloads the model from Hugging Face on first use."""
    from fastembed import SparseTextEmbedding

    model = SparseTextEmbedding(model_name=model_name)

    def _doc(texts: list[str]):
        embs = list(model.embed(texts))
        return [e.indices.tolist() for e in embs], [e.values.tolist() for e in embs]

    def _query(texts: list[str]):
        embs = list(model.query_embed(texts))
        return [e.indices.tolist() for e in embs], [e.values.tolist() for e in embs]

    return _doc, _query


def requires_idf(provider: str, model_name: str) -> bool:
    if provider == "bm25":
        return True
    return "bm25" in model_name.lower()
