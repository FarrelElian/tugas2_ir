"""Tahap 5 - BM25 (Okapi).

    score(q,d) = sum_{t in q}  IDF(t) * tf*(k1+1) / ( tf + k1*(1 - b + b*|d|/avgdl) )
    IDF(t)     = ln( 1 + (N - df + 0,5) / (df + 0,5) )

Nilai awal sesuai ketentuan: k1 = 1,2 ; b = 0,75.
 - tf saturation : komponen tf dibatasi k1+1 sehingga pengulangan term tidak menaikkan skor secara linear.
 - length norm   : b mengatur hukuman untuk dokumen yang lebih panjang dari rata-rata (b=0 -> panjang diabaikan).
 - IDF           : term yang jarang muncul bernilai lebih tinggi. Varian 'ln(1+...)' tidak pernah negatif.
Setiap term query unik dihitung sekali.
"""
import math
from collections import defaultdict

import pandas as pd

from . import config


def tf_component(tf, dl, avgdl, k1, b):
    """Komponen TF BM25 untuk satu term pada satu dokumen."""
    if not tf:
        return 0.0
    return tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


class BM25Retriever:
    def __init__(self, index, k1=config.K1, b=config.B):
        self.index, self.k1, self.b = index, k1, b
        self.idf = {t: math.log(1 + (index.N - df + 0.5) / (df + 0.5)) for t, df in index.df.items()}

    def score(self, q_tokens):
        acc = defaultdict(float)
        ix = self.index
        for t in set(q_tokens):
            if t not in ix.postings:
                continue
            idf = self.idf[t]
            for d, tf in ix.postings[t].items():
                acc[d] += idf * tf_component(tf, ix.doc_len[d], ix.avgdl, self.k1, self.b)
        return dict(acc)


def saturation_table(k1_list=(0.5, 1.2, 2.0), tfs=(1, 2, 3, 5, 10, 20, 30)):
    """Komponen TF BM25 pada dl = avgdl untuk beberapa tf: terlihat jenuh mendekati k1+1."""
    rows = []
    for k1 in k1_list:
        row = {"k1": k1, "batas atas (k1+1)": k1 + 1}
        for tf in tfs:
            row[f"tf={tf}"] = tf_component(tf, 1, 1, k1, 0.75)
        rows.append(row)
    return pd.DataFrame(rows)


def length_table(k1=config.K1, tf=3, b_list=(0, 0.5, 0.75, 1.0), mults=(0.25, 0.5, 1, 2, 4)):
    """Komponen TF BM25 untuk tf tetap, panjang dokumen berbeda (kelipatan avgdl) dan b berbeda."""
    return pd.DataFrame(
        {f"dl = {m}x avgdl": [tf_component(tf, m, 1, k1, b) for b in b_list] for m in mults},
        index=[f"b = {b}" for b in b_list])


if __name__ == "__main__":
    # Demo modul: python -m src.bm25
    from .preprocessing import load_and_clean, default_preprocessor, build_docs_tokens
    from .indexing import InvertedIndex
    from .utils import rank_docs
    df, _ = load_and_clean()
    pre = default_preprocessor()
    bm = BM25Retriever(InvertedIndex(build_docs_tokens(df, pre)))
    q = pre("harga beras naik")
    sc = bm.score(q)
    print(f"k1={bm.k1} b={bm.b} | token query: {q}")
    for i, d in enumerate(rank_docs(sc)[:5], 1):
        print(f"{i}. {d}  skor={sc[d]:.4f}")
    print("\nSaturasi TF:\n", saturation_table().round(3).to_string(index=False))
