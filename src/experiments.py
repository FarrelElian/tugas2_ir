"""Eksperimen 1-3 dan analisis lanjutan. Semua fungsi mencetak hasilnya ke terminal (dan mengembalikan DataFrame)."""
import math

import matplotlib
matplotlib.use("Agg")                       # tanpa jendela grafik: grafik disimpan ke berkas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .bm25 import BM25Retriever, length_table, saturation_table, tf_component
from .evaluation import average_precision, ndcg_at_k
from .indexing import InvertedIndex
from .preprocessing import Preprocessor, STOPWORDS_PURE, doc_text
from .tfidf_retrieval import TfidfRetriever
from .utils import rank_docs, section, show, spearman


# =============================================================== Eksperimen 1
def experiment1(engine, results, queries, k=5):
    """Top-k dan skor untuk setiap query: TF-IDF+cosine berdampingan dengan BM25."""
    info = engine.info
    for q in queries:
        r = results[q["id"]]
        rows = []
        for i in range(k):
            row = {"rank": i + 1}
            for m, name in (("tfidf", "TF-IDF"), ("bm25", "BM25")):
                if i < len(r[m]):
                    d = r[m][i]
                    row[f"{name} judul"] = f"{d}: {info.loc[d, 'title'][:50]}"
                    row[f"{name} skor"] = round(r["s_" + m][d], 4)
                else:
                    row[f"{name} judul"], row[f"{name} skor"] = "-", None
            rows.append(row)
        print(f"\n{q['id']} | {q['text']!r} | token: {r['tokens']}")
        show(pd.DataFrame(rows), max_colwidth=58)


def export_topk(engine, results, queries, path, k=config.TOP_K):
    rows = []
    for q in queries:
        qt = results[q["id"]]["tokens"]
        for m in ["tfidf", "bm25"]:
            for rnk, d in enumerate(results[q["id"]][m][:k], 1):
                rows.append({"query_id": q["id"], "query": q["text"], "method": m, "rank": rnk, "doc_id": d,
                             "score": round(results[q["id"]]["s_" + m][d], 5), "title": engine.info.loc[d, "title"],
                             "url": engine.info.loc[d, "url"], "snippet": engine.make_snippet(d, qt)})
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


# =============================================================== Analisis BM25 (TF, IDF, panjang, saturasi)
def bm25_analysis(engine, out_png):
    ix, bm = engine.index, engine.bm25

    print("(1) TF SATURATION: komponen TF BM25 (dl = avgdl) jenuh mendekati batas atas k1+1")
    show(saturation_table(), decimals=3)

    tfs = np.arange(1, 31)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k1 in (0.5, 1.2, 2.0):
        f = tfs * (k1 + 1) / (tfs + k1)
        ax[0].plot(tfs, f, label=f"BM25 k1={k1}")
        ax[0].axhline(k1 + 1, ls=":", lw=0.8, color="gray")
        ax[1].plot(tfs, f / f[0], label=f"BM25 k1={k1}")
    ax[1].plot(tfs, tfs, "k--", label="TF mentah (linear)")
    ax[1].plot(tfs, 1 + np.log10(tfs), "k:", label="1 + log10(tf) (TF-IDF)")
    ax[0].set_title("Komponen TF BM25 jenuh di k1+1"); ax[0].set_xlabel("tf"); ax[0].legend()
    ax[1].set_title("Dinormalisasi terhadap tf=1"); ax[1].set_xlabel("tf"); ax[1].set_ylim(0, 8); ax[1].legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)
    print(f"Grafik disimpan: {out_png}")

    t = engine.pre("beras")[0]
    plist = sorted(ix.get(t).items(), key=lambda x: x[1])
    picks = [plist[int(i)] for i in np.linspace(0, len(plist) - 1, 6)]
    rows = [{"doc_id": d, "tf": tf, "dl": ix.doc_len[d],
             "komponen_tf_bm25": tf_component(tf, ix.doc_len[d], ix.avgdl, bm.k1, bm.b),
             "1+log10(tf)": 1 + math.log10(tf), "tf mentah": tf} for d, tf in picks]
    print(f"\nContoh nyata, term '{t}' pada dokumen dengan tf berbeda (k1={bm.k1}, b={bm.b}):")
    show(pd.DataFrame(rows), decimals=3)

    print("\n(2) PENGARUH PANJANG DOKUMEN: tf = 3, panjang dokumen berbeda, b berbeda")
    print("    (makin besar b, dokumen panjang makin dihukum; b = 0 -> panjang diabaikan)")
    show(length_table(bm.k1), index=True, decimals=3)

    print("\n(3) PENGARUH IDF: term jarang bernilai lebih tinggi daripada term umum")
    rows = []
    for w in ["harga", "pangan", "beras", "cabai", "bulog", "satgas", "sidak", "impor"]:
        tk = engine.pre(w)
        if tk and tk[0] in ix.df:
            t = tk[0]
            rows.append({"kata": w, "term": t, "df": ix.df[t], "df/N": ix.df[t] / ix.N,
                         "idf_tfidf": engine.tfidf.idf[t], "idf_bm25": bm.idf[t]})
    show(pd.DataFrame(rows).sort_values("df", ascending=False), decimals=3)


# =============================================================== Eksperimen 2
def compare_rankings(engine, results, queries, qrels, qid, k=10):
    r = results[qid]
    q = next(x for x in queries if x["id"] == qid)
    grades = qrels.get(qid, {})
    rk_tf = {d: i for i, d in enumerate(r["tfidf"], 1)}
    rk_bm = {d: i for i, d in enumerate(r["bm25"], 1)}
    big = engine.index.N + 1
    cand = list(dict.fromkeys(r["tfidf"][:k] + r["bm25"][:k]))
    qt = set(engine.pre(q["text"]))
    rows = []
    for d in cand:
        rows.append({"doc_id": d, "rank_tfidf": rk_tf.get(d, big), "rank_bm25": rk_bm.get(d, big),
                     "geser (tfidf-bm25)": rk_tf.get(d, big) - rk_bm.get(d, big),
                     "dl": engine.index.doc_len[d],
                     "tf_term_query": sum(engine.index.get(t).get(d, 0) for t in qt),
                     "label": grades.get(d, "?"), "judul": engine.info.loc[d, "title"][:55]})
    out = pd.DataFrame(rows)
    out["_m"] = out[["rank_tfidf", "rank_bm25"]].min(axis=1)
    return out.sort_values("_m").drop(columns="_m").reset_index(drop=True)


def experiment2(engine, results, queries, qrels, per_query):
    """Perbedaan ranking VSM cosine vs BM25: overlap, selisih AP, perbandingan peringkat, penguraian skor."""
    ov = pd.DataFrame([{"query_id": q["id"],
                        "overlap_top10 (Jaccard)": _jaccard(results[q["id"]]["tfidf"][:10], results[q["id"]]["bm25"][:10])}
                       for q in queries]).set_index("query_id")
    ap = per_query.pivot(index="query_id", columns="method", values="AP")
    comp = ov.join((ap["bm25"] - ap["tfidf"]).rename("AP BM25 - TFIDF")).reindex([q["id"] for q in queries])
    comp["|selisih AP|"] = comp["AP BM25 - TFIDF"].abs()
    show(comp.sort_values("|selisih AP|", ascending=False), index=True, decimals=3)
    print("Rata-rata Jaccard top-10: %.3f" % ov["overlap_top10 (Jaccard)"].mean())

    qmap = {q["id"]: q for q in queries}
    for qid in comp["|selisih AP|"].dropna().sort_values(ascending=False).index[:2]:
        q = qmap[qid]
        print("\n" + "-" * 100)
        print(f"{qid} | {q['text']!r} | token: {results[qid]['tokens']} | "
              f"AP BM25={ap.loc[qid, 'bm25']:.3f} | AP TF-IDF={ap.loc[qid, 'tfidf']:.3f}")
        cmp_df = compare_rankings(engine, results, queries, qrels, qid)
        show(cmp_df, max_colwidth=55)
        both = cmp_df[(cmp_df["rank_tfidf"] <= engine.index.N) & (cmp_df["rank_bm25"] <= engine.index.N)]
        src = both if len(both) else cmp_df
        d_top = src.iloc[src["geser (tfidf-bm25)"].abs().argmax()]["doc_id"]
        tbl, dl, ln = engine.explain_doc(q["text"], d_top)
        print(f"\nDokumen dengan pergeseran peringkat terbesar: {d_top} (dl={dl}, avgdl={engine.index.avgdl:.1f}, "
              f"faktor panjang={ln:.3f}) - penguraian skor per term:")
        show(tbl, decimals=4)
    return comp


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else float("nan")


# =============================================================== Eksperimen 3
def experiment3(engine, results, queries, short_queries=config.SHORT_QUERIES, k=5):
    """Query pendek dan spesifik: pengaruh tf dan panjang dokumen pada cosine vs BM25."""
    ix, info = engine.index, engine.info
    summ = []
    for sq in short_queries:
        qt = engine.pre(sq)
        s_tf, s_bm = engine.tfidf.score(qt), engine.bm25.score(qt)
        r_tf, r_bm = rank_docs(s_tf), rank_docs(s_bm)
        print("\n" + "-" * 100)
        print(f"Query {sq!r} -> token {qt} | dokumen cocok: {len(r_bm)} | df term: {[ix.df.get(t, 0) for t in qt]}")
        if not r_bm:
            print(">>", engine.diagnose_empty(sq, qt))
            continue
        terms = set(qt)

        def rows(rk, sc):
            return pd.DataFrame([{"rank": i, "doc_id": d, "tf": sum(ix.get(t).get(d, 0) for t in terms),
                                  "dl": ix.doc_len[d], "skor": sc[d], "judul": info.loc[d, "title"][:50]}
                                 for i, d in enumerate(rk[:k], 1)])
        print("TF-IDF + cosine, Top-5:"); show(rows(r_tf, s_tf))
        print("BM25, Top-5:");            show(rows(r_bm, s_bm))
        docs = list(s_bm)
        dls = [ix.doc_len[d] for d in docs]
        summ.append({"query": sq, "cocok": len(docs),
                     "overlap_top5": _jaccard(r_tf[:5], r_bm[:5]),
                     "rata2_dl_top5_tfidf": np.mean([ix.doc_len[d] for d in r_tf[:5]]),
                     "rata2_dl_top5_bm25": np.mean([ix.doc_len[d] for d in r_bm[:5]]),
                     "avgdl_korpus": ix.avgdl,
                     "spearman(dl,skor)_tfidf": spearman(dls, [s_tf.get(d, 0) for d in docs]),
                     "spearman(dl,skor)_bm25": spearman(dls, [s_bm[d] for d in docs])})
    print("\nRINGKASAN EKSPERIMEN 3")
    s3 = pd.DataFrame(summ).set_index("query") if summ else pd.DataFrame()
    show(s3, index=True, decimals=3)

    print("\nProfil panjang dokumen untuk 10 query uji:")
    rows = []
    for q in queries:
        r = results[q["id"]]
        docs = list(r["s_bm25"])
        dls = [ix.doc_len[d] for d in docs]
        rows.append({"query_id": q["id"],
                     "rata2_dl_top10_tfidf": np.mean([ix.doc_len[d] for d in r["tfidf"][:10]]) if r["tfidf"] else np.nan,
                     "rata2_dl_top10_bm25": np.mean([ix.doc_len[d] for d in r["bm25"][:10]]) if r["bm25"] else np.nan,
                     "spearman(dl,skor)_tfidf": spearman(dls, [r["s_tfidf"].get(d, 0) for d in docs]),
                     "spearman(dl,skor)_bm25": spearman(dls, [r["s_bm25"][d] for d in docs])})
    lp = pd.DataFrame(rows).set_index("query_id")
    show(lp, index=True, decimals=3)
    print(f"avgdl korpus = {ix.avgdl:.1f} | rata-rata korelasi -> TF-IDF: "
          f"{lp['spearman(dl,skor)_tfidf'].mean():.3f} | BM25: {lp['spearman(dl,skor)_bm25'].mean():.3f}")
    print("\nCara membaca: spearman(dl,skor) > 0 = dokumen panjang cenderung diuntungkan; < 0 = dokumen pendek diuntungkan.")
    print("Interpretasi untuk laporan harus ditulis dari angka di atas (bukan disalin dari sini).")
    return s3, lp


# =============================================================== Analisis lanjutan
def sensitivity_grid(engine, queries, qrels, thr=1):
    """MAP BM25 untuk kombinasi (k1, b)."""
    k1_grid, b_grid = [0.5, 1.2, 1.5, 2.0], [0.25, 0.5, 0.75, 1.0]
    grid = pd.DataFrame(index=[f"k1={k}" for k in k1_grid], columns=[f"b={b}" for b in b_grid], dtype=float)
    for k in k1_grid:
        for b in b_grid:
            bm = BM25Retriever(engine.index, k1=k, b=b)
            aps = []
            for q in queries:
                rel = {d for d, g in qrels.get(q["id"], {}).items() if g >= thr}
                if rel:
                    aps.append(average_precision(rank_docs(bm.score(engine.pre(q["text"]))), rel))
            grid.loc[f"k1={k}", f"b={b}"] = float(np.mean(aps)) if aps else float("nan")
    show(grid, index=True)
    print(f"Nilai yang dipakai: k1={config.K1}, b={config.B} (default tugas). Dengan ~10 query selisih kecil antar sel "
          "tidak bermakna; jangan menyetel parameter lalu melaporkannya sebagai performa umum.")
    return grid


def ablation(engine, queries, qrels):
    """Pengaruh stemming/stopword (termasuk 'naik' dibuang vs dipertahankan). Pool dibangun dari konfigurasi default,
    sehingga konfigurasi lain sedikit dirugikan: baca sebagai indikasi, bukan bukti."""
    configs = [
        ("token saja (tanpa stopword/stem)",                 dict(stem=False, stop=False)),
        ("stopword Sastrawi murni ('naik' dibuang)",         dict(stem=False, stop=True, stopwords=STOPWORDS_PURE)),
        ("stopword, 'naik' dipertahankan",                   dict(stem=False, stop=True)),
        ("stem + stopword murni ('naik' dibuang)",           dict(stem=True, stop=True, stopwords=STOPWORDS_PURE)),
        ("stem + stopword, 'naik' dipertahankan (DEFAULT)",  dict(stem=True, stop=True)),
    ]
    rows = []
    for name, kw in configs:
        p = Preprocessor(min_len=config.MIN_TOKEN_LEN, **kw)
        ix = InvertedIndex({r["doc_id"]: p(doc_text(r)) for _, r in engine.df.iterrows()})
        retr = {"tfidf": TfidfRetriever(ix).score, "bm25": BM25Retriever(ix, config.K1, config.B).score}
        res = {"konfigurasi": name, "vocab": len(ix.postings)}
        for label, fn in retr.items():
            aps, nd = [], []
            for q in queries:
                g = qrels.get(q["id"], {})
                rel = {d for d, v in g.items() if v >= 1}
                if rel:
                    rk = rank_docs(fn(p(q["text"])))
                    aps.append(average_precision(rk, rel))
                    nd.append(ndcg_at_k(rk, g, 10))
            res[f"MAP {label}"] = float(np.mean(aps)) if aps else float("nan")
            res[f"nDCG@10 {label}"] = float(np.nanmean(nd)) if nd else float("nan")
        rows.append(res)
    out = pd.DataFrame(rows)
    show(out, max_colwidth=52)
    return out


def inspect_query(engine, results, queries, qrels, qid, method="bm25", k=10):
    q = next(x for x in queries if x["id"] == qid)
    grades = qrels.get(qid, {})
    ranked = results[qid][method]
    print(f"{qid} | {q['text']!r} | metode: {method} | token query: {results[qid]['tokens']}")
    show(pd.DataFrame([{"rank": i, "doc_id": d, "label": grades.get(d, "belum dinilai"),
                        "judul": engine.info.loc[d, "title"][:80]} for i, d in enumerate(ranked[:k], 1)]))
    missed = [d for d, g in grades.items() if g >= 1 and d not in ranked[:k]]
    print(f"Dokumen relevan yang TIDAK masuk top-{k}: {missed}\n")


def plot_summary(summary, path, title):
    cols = [c for c in ["P@5", "F1@5", "P@10", "F1@10", "MAP", "MRR", "nDCG@10"] if c in summary.columns]
    ax = summary[cols].T.plot(kind="bar", figsize=(9, 4), rot=0)
    ax.set_ylim(0, 1.05); ax.set_ylabel("skor"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
