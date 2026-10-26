"""Mesin pencari sederhana: menyatukan preprocessing, index, dan tiga model retrieval.

Alur data dari query sampai Top-k:

  query pengguna
    -> preprocessing (pipeline SAMA dengan dokumen)        -> token query
    -> ambil postings tiap term dari inverted index         -> dokumen kandidat
    -> hitung skor (cosine TF-IDF / BM25) atau himpunan Boolean
    -> urutkan skor menurun                                 -> ranking
    -> ambil Top-k dan tampilkan judul, URL, skor, cuplikan

Pemakaian terminal:  python -m src.search_engine          (interaktif)
                     python -m src.search_engine --demo   (contoh tanpa input)
Pemakaian web     :  streamlit run app.py
"""
import argparse
import math
import re

import pandas as pd

from . import config
from .boolean_retrieval import BooleanRetriever
from .bm25 import BM25Retriever
from .indexing import InvertedIndex
from .preprocessing import build_docs_tokens, default_preprocessor, load_and_clean
from .tfidf_retrieval import TfidfRetriever
from .utils import rank_docs

OPERATOR_RE = re.compile(r"\b(AND|OR|NOT)\b")


class SearchEngine:
    METHODS = ("tfidf", "bm25", "boolean")

    def __init__(self, data_path=None, k1=config.K1, b=config.B):
        self.df, self.clean_log = load_and_clean(data_path)
        self.info = self.df.set_index("doc_id")
        self.pre = default_preprocessor()
        self.docs_tokens = build_docs_tokens(self.df, self.pre)
        self.index = InvertedIndex(self.docs_tokens)
        self.boolean = BooleanRetriever(self.index, self.pre)
        self.tfidf = TfidfRetriever(self.index)
        self.bm25 = BM25Retriever(self.index, k1, b)

    # ---------- pemrosesan query ----------
    def analyze_query(self, query, method="bm25"):
        """Token query. Untuk Boolean, operator AND/OR/NOT dibuang dulu dari token."""
        if method == "boolean":
            query = OPERATOR_RE.sub(" ", query)
        return self.pre(query)

    def diagnose_empty(self, query, q_tokens):
        if not query.strip():
            return "Query kosong. Ketik minimal satu kata."
        if not q_tokens:
            return ("Query hanya berisi kata yang dibuang saat preprocessing (stopword/angka/simbol). "
                    "Coba kata yang lebih spesifik.")
        unknown = [t for t in q_tokens if t not in self.index.postings]
        if len(unknown) == len(set(q_tokens)):
            return f"Tidak ada dokumen yang memuat term {sorted(set(unknown))}. Coba kata lain atau sinonimnya."
        return "Tidak ada dokumen yang cocok dengan query ini."

    # ---------- pencarian ----------
    def search(self, query, method="bm25", k=config.TOP_K):
        """Kembalikan dict: ranked (semua dokumen cocok, terurut), scores, tokens, message (jika kosong/error)."""
        if method not in self.METHODS:
            raise ValueError(f"method harus salah satu dari {self.METHODS}")
        q_tokens = self.analyze_query(query, method)
        error = None
        if method == "boolean":
            try:
                ranked = self.boolean.evaluate(query)
            except ValueError as e:
                ranked, error = [], str(e)
            scores = {d: 1.0 for d in ranked}
        else:
            retr = self.tfidf if method == "tfidf" else self.bm25
            scores = retr.score(q_tokens)
            ranked = rank_docs(scores)

        message = None
        if error:
            message = f"Query Boolean tidak valid: {error}"
        elif not ranked:
            message = self.diagnose_empty(query, q_tokens)
        return {"query": query, "method": method, "tokens": q_tokens, "ranked": ranked,
                "scores": scores, "message": message, "k": k}

    # ---------- tampilan ----------
    def make_snippet(self, doc_id, q_tokens, max_chars=260):
        """Cuplikan = kalimat dokumen yang memuat paling banyak term query (kalimat diproses dengan pipeline sama)."""
        content = self.info.loc[doc_id, "content"]
        qset, best, best_hits = set(q_tokens), None, 0
        for s in re.split(r"(?<=[.!?])\s+", content):
            if len(s) < 25:
                continue
            hits = len(set(self.pre(s)) & qset)
            if hits > best_hits:
                best, best_hits = s, hits
        best = (best or content).strip()
        return best[:max_chars].rstrip() + (" ..." if len(best) > max_chars else "")

    def result_rows(self, res, k=None):
        """Top-k sebagai daftar dict: rank, doc_id, title, url, score, snippet."""
        k = res["k"] if k is None else k
        rows = []
        for i, d in enumerate(res["ranked"][:k], 1):
            rows.append({"rank": i, "doc_id": d, "title": self.info.loc[d, "title"],
                         "url": self.info.loc[d, "url"],
                         "score": None if res["method"] == "boolean" else res["scores"][d],
                         "snippet": self.make_snippet(d, res["tokens"])})
        return rows

    def format_results(self, res, k=None):
        k = res["k"] if k is None else k
        out = ["=" * 90,
               f"Query: {res['query']!r} | metode: {res['method'].upper()} | token query: {res['tokens']}"]
        if res["message"]:
            out += ["", ">> " + res["message"], ""]
            return "\n".join(out)
        rows = self.result_rows(res, k)
        out.append(f"Dokumen cocok: {len(res['ranked'])} | ditampilkan: Top-{len(rows)}")
        for r in rows:
            sk = "cocok (Boolean tidak berskor)" if r["score"] is None else f"{r['score']:.4f}"
            out += ["", f"{r['rank']}. {r['title']}", f"   URL     : {r['url']}",
                    f"   Skor    : {sk}   [{r['doc_id']}]", f"   Cuplikan: {r['snippet']}"]
        out.append("")
        return "\n".join(out)

    def explain_doc(self, query, doc_id):
        """Penguraian skor satu dokumen per term: kontribusi cosine TF-IDF dan BM25 berdampingan."""
        q_tokens = self.pre(query)
        terms = sorted(set(t for t in q_tokens if t in self.index.postings))
        qv = self.tfidf.query_vector(q_tokens)
        qnorm = math.sqrt(sum(v * v for v in qv.values())) or 1.0
        dl = self.index.doc_len[doc_id]
        len_norm = 1 - self.bm25.b + self.bm25.b * dl / self.index.avgdl
        rows = []
        for t in terms:
            tf = self.index.get(t).get(doc_id, 0)
            nd = self.tfidf.norm[doc_id]
            cos_c = (qv.get(t, 0.0) * self.tfidf.w[t].get(doc_id, 0.0)) / (nd * qnorm) if nd > 0 else 0.0
            tf_comp = tf * (self.bm25.k1 + 1) / (tf + self.bm25.k1 * len_norm) if tf else 0.0
            rows.append({"term": t, "tf": tf, "df": self.index.df[t],
                         "idf_tfidf": self.tfidf.idf[t], "kontrib_cosine": cos_c,
                         "idf_bm25": self.bm25.idf[t], "komponen_tf_bm25": tf_comp,
                         "kontrib_bm25": self.bm25.idf[t] * tf_comp})
        return pd.DataFrame(rows), dl, len_norm


# ---------------------------------------------------------------------
def run_cli(engine):
    print("Mesin Pencari Sederhana - ketik 'keluar' untuk berhenti.")
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in {"keluar", "exit", "quit"}:
            print("Selesai.")
            break
        method = input("Metode [tfidf/bm25/boolean/both] (default bm25): ").strip().lower() or "bm25"
        if method not in {"tfidf", "bm25", "boolean", "both"}:
            print("Metode tidak dikenal, memakai bm25.")
            method = "bm25"
        ks = input("Top-k (default 5): ").strip()
        k = int(ks) if ks.isdigit() and int(ks) > 0 else 5
        for m in (("tfidf", "bm25") if method == "both" else (method,)):
            print(engine.format_results(engine.search(query, m, k)))


def run_demo(engine):
    for q, m in [("harga beras naik", "tfidf"), ("harga beras naik", "bm25"),
                 ("beras AND harga AND naik", "boolean"),
                 ("kucing terbang", "bm25"), ("dan yang di", "bm25")]:
        print(engine.format_results(engine.search(q, m, 3)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mesin pencari sederhana (terminal)")
    ap.add_argument("--demo", action="store_true", help="jalankan contoh query tanpa input")
    args = ap.parse_args()
    eng = SearchEngine()
    run_demo(eng) if args.demo else run_cli(eng)
