"""Tahap 3-4 - TF-IDF, Vector Space Model, cosine similarity.

Setiap dokumen dan query = vektor berdimensi |vocabulary|, tiap dimensi = bobot satu term.

    TF  (sublinear) : 1 + log10(tf)            bila tf > 0, selain itu 0
    IDF             : log10(N / df)
    TF-IDF          : w(t,d) = (1 + log10 tf(t,d)) * log10(N / df(t))
    Cosine          : cos(q,d) = sum_t w(t,q)*w(t,d) / (||q|| * ||d||)

Skor dihitung term-at-a-time lewat inverted index (hanya dokumen yang memuat term query yang disentuh).
Term dengan df = N mendapat idf = 0 sehingga tidak memengaruhi skor.
"""
import math
from collections import Counter, defaultdict

import pandas as pd


class TfidfRetriever:
    def __init__(self, index):
        self.index = index
        self.idf = {t: math.log10(index.N / df) for t, df in index.df.items()}
        self.w = defaultdict(dict)                 # term -> {doc: bobot}   (untuk menghitung skor)
        self.doc_vec = defaultdict(dict)           # doc  -> {term: bobot}  (representasi vektor dokumen)
        sq = defaultdict(float)
        for t, plist in index.postings.items():
            for d, tf in plist.items():
                wt = (1 + math.log10(tf)) * self.idf[t]
                self.w[t][d] = wt
                self.doc_vec[d][t] = wt
                sq[d] += wt * wt
        self.norm = {d: math.sqrt(sq.get(d, 0.0)) for d in index.doc_ids}      # ||d||

    def query_vector(self, q_tokens):
        """Vektor query (sparse): term -> bobot, dengan rumus bobot yang sama seperti dokumen."""
        qtf = Counter(t for t in q_tokens if t in self.idf)
        return {t: (1 + math.log10(c)) * self.idf[t] for t, c in qtf.items()}

    def score(self, q_tokens):
        """Cosine similarity query terhadap semua dokumen kandidat -> {doc_id: skor}."""
        qv = self.query_vector(q_tokens)
        qnorm = math.sqrt(sum(v * v for v in qv.values()))
        if qnorm == 0:
            return {}
        acc = defaultdict(float)                   # akumulasi dot product
        for t, wq in qv.items():
            for d, wd in self.w[t].items():
                acc[d] += wq * wd
        return {d: s / (self.norm[d] * qnorm) for d, s in acc.items() if self.norm[d] > 0 and s > 0}

    def explain(self, q_tokens, doc_id, n_top_terms=8):
        """Perhitungan langkah demi langkah untuk satu dokumen (untuk laporan)."""
        qv = self.query_vector(q_tokens)
        if not qv:
            return None
        qnorm = math.sqrt(sum(v * v for v in qv.values()))
        qtf = Counter(t for t in q_tokens if t in self.idf)
        q_table = pd.DataFrame([{
            "term": t, "tf_q": qtf[t], "df": self.index.df[t],
            "idf = log10(N/df)": self.idf[t], "w_q = (1+log10 tf)*idf": qv[t],
        } for t in sorted(qv)])
        rows, dot = [], 0.0
        for t in sorted(qv):
            wd = self.w[t].get(doc_id, 0.0)
            dot += qv[t] * wd
            rows.append({"term": t, "tf_d": self.index.get(t).get(doc_id, 0),
                         "w_d = (1+log10 tf)*idf": wd, "w_q": qv[t], "w_q*w_d": qv[t] * wd})
        nd = self.norm[doc_id]
        cos = dot / (nd * qnorm) if nd > 0 else 0.0
        top = sorted(self.doc_vec[doc_id].items(), key=lambda x: -x[1])[:n_top_terms]
        return {"query_table": q_table, "doc_table": pd.DataFrame(rows), "dot": dot,
                "norm_d": nd, "norm_q": qnorm, "cosine": cos,
                "check": self.score(q_tokens).get(doc_id, 0.0),
                "n_dims": len(self.doc_vec[doc_id]), "top_terms": top}


if __name__ == "__main__":
    # Demo modul: python -m src.tfidf_retrieval
    from .preprocessing import load_and_clean, default_preprocessor, build_docs_tokens
    from .indexing import InvertedIndex
    from .utils import rank_docs
    df, _ = load_and_clean()
    pre = default_preprocessor()
    tf = TfidfRetriever(InvertedIndex(build_docs_tokens(df, pre)))
    q = pre("harga beras naik")
    sc = tf.score(q)
    print("token query:", q)
    for i, d in enumerate(rank_docs(sc)[:5], 1):
        print(f"{i}. {d}  cosine={sc[d]:.4f}")
