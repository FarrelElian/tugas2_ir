"""Uji kebenaran implementasi.

1. Korpus mini yang hasilnya bisa dihitung terpisah:
   - skor TF-IDF via inverted index = cosine dari matriks penuh,
   - skor BM25 via inverted index = rumus langsung per dokumen,
   - ekspresi Boolean = operasi himpunan.
2. Pada korpus asli: TF-IDF via index dibandingkan dengan perhitungan matriks penuh (NumPy).
Jika ada yang salah, program berhenti dengan AssertionError.
"""
import math

import numpy as np

from .boolean_retrieval import BooleanRetriever
from .bm25 import BM25Retriever
from .indexing import InvertedIndex
from .tfidf_retrieval import TfidfRetriever


def toy_tests():
    toy = {"d1": ["a", "a", "b"], "d2": ["b", "c"], "d3": ["c", "c", "c", "d"], "d4": ["a", "d", "d"]}
    ix = InvertedIndex(toy)
    tf, bm = TfidfRetriever(ix), BM25Retriever(ix, k1=1.2, b=0.75)
    bl = BooleanRetriever(ix, lambda w: [w.lower()])
    q = ["a", "c"]

    # (1) TF-IDF vs matriks penuh
    vocab = sorted(ix.postings)
    M = np.zeros((ix.N, len(vocab)))
    for i, d in enumerate(ix.doc_ids):
        for j, t in enumerate(vocab):
            c = ix.get(t).get(d, 0)
            if c:
                M[i, j] = (1 + math.log10(c)) * math.log10(ix.N / ix.df[t])
    qv = np.array([math.log10(ix.N / ix.df[t]) if t in q else 0.0 for t in vocab])
    cos_full = (M @ qv) / (np.linalg.norm(M, axis=1) * np.linalg.norm(qv))
    sc = tf.score(q)
    for i, d in enumerate(ix.doc_ids):
        assert abs(sc.get(d, 0.0) - cos_full[i]) < 1e-9, f"TF-IDF beda di {d}"

    # (2) BM25 vs rumus langsung
    def direct(d, k1=1.2, b=0.75):
        s = 0.0
        for t in set(q):
            n = sum(1 for x in toy.values() if t in x)
            idf = math.log(1 + (len(toy) - n + 0.5) / (n + 0.5))
            c = toy[d].count(t)
            avgdl = sum(map(len, toy.values())) / len(toy)
            s += idf * c * (k1 + 1) / (c + k1 * (1 - b + b * len(toy[d]) / avgdl))
        return s
    bs = bm.score(q)
    for d in toy:
        assert abs(bs.get(d, 0.0) - direct(d)) < 1e-9, f"BM25 beda di {d}"

    # (3) Boolean vs himpunan
    A, C, D, U = set(ix.get("a")), set(ix.get("c")), set(ix.get("d")), set(toy)
    assert set(bl.evaluate("a AND c")) == A & C
    assert set(bl.evaluate("a OR c")) == A | C
    assert set(bl.evaluate("NOT a")) == U - A
    assert set(bl.evaluate("a AND (c OR d)")) == A & (C | D)
    assert set(bl.evaluate("a c")) == A & C                        # AND implisit
    assert set(bl.evaluate("NOT (a OR c)")) == U - (A | C)
    for bad in ["a AND", "(a OR c", "AND a", "a )"]:
        try:
            bl.evaluate(bad)
            raise AssertionError(f"query tidak valid tidak ditolak: {bad!r}")
        except ValueError:
            pass
    print("Uji korpus mini LULUS: TF-IDF = matriks penuh, BM25 = rumus langsung, Boolean = himpunan, query salah ditolak.")


def corpus_tfidf_check(engine, query="harga beras naik"):
    ix, tf = engine.index, engine.tfidf
    vocab = sorted(ix.postings)
    v_idx = {t: i for i, t in enumerate(vocab)}
    pos = {d: i for i, d in enumerate(ix.doc_ids)}
    M = np.zeros((ix.N, len(vocab)))
    for t, plist in ix.postings.items():
        for d in plist:
            M[pos[d], v_idx[t]] = tf.w[t][d]
    qtoks = engine.pre(query)
    qv = np.zeros(len(vocab))
    for t, w in tf.query_vector(qtoks).items():
        qv[v_idx[t]] = w
    den = np.linalg.norm(M, axis=1) * np.linalg.norm(qv)
    cos_dense = np.divide(M @ qv, den, out=np.zeros(ix.N), where=den > 0)
    sc = tf.score(qtoks)
    err = max(abs(sc.get(d, 0.0) - cos_dense[pos[d]]) for d in ix.doc_ids)
    print(f"Korpus asli, query {query!r}: selisih maksimum skor TF-IDF (index vs matriks penuh) = {err:.2e}")
    assert err < 1e-9
    return err


def run_all(engine=None):
    toy_tests()
    if engine is not None:
        corpus_tfidf_check(engine)


if __name__ == "__main__":
    # Demo modul: python -m src.sanity_tests
    from .search_engine import SearchEngine
    run_all(SearchEngine())
