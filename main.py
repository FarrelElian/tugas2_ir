"""main.py - Menjalankan SELURUH pipeline Tugas 2 dan MENCETAK setiap hasilnya ke terminal.

    python main.py                    (dataset di data/dataset_crawling_ir.csv)
    python main.py --data path.csv    (dataset di lokasi lain)

Semua output terminal juga disimpan ke hasil/run_log.txt, sehingga hasil eksekusi bisa dibaca
tanpa menjalankan ulang program. Berkas hasil lain (CSV, grafik) disimpan di hasil/, index/, data/, evaluation/.

Tahapan:  2 pembersihan data -> 3 preprocessing -> 4 inverted index -> 5 Boolean -> 6 TF-IDF/VSM/cosine
          -> 7 BM25 -> 8 uji kebenaran -> 9 mesin pencari -> 10 Eksperimen 1 -> 11 pooling dan label
          -> 12 evaluasi -> 13 Eksperimen 2 -> 14 Eksperimen 3 -> 15 analisis lanjutan -> 16 ekspor
"""
import argparse
import datetime
import math
import platform
import sys

import numpy as np
import pandas as pd

from src import config, sanity_tests
from src.bm25 import BM25Retriever
from src.evaluation import (METHODS, boolean_table, coverage_table, run_eval, sig_table)
from src.experiments import (ablation, bm25_analysis, experiment1, experiment2, experiment3, export_topk,
                             inspect_query, plot_summary, sensitivity_grid)
from src.indexing import index_example, save_index_json
from src.preprocessing import (Preprocessor, STOPWORDS, STOPWORDS_PURE, doc_text, save_preprocessed,
                               stem_word)
from src.qrels import build_pool, build_results, load_labels, load_queries
from src.search_engine import SearchEngine, run_demo
from src.utils import Tee, rank_docs, section, show

H = config.HASIL_DIR


def print_tfidf_explanation(engine, query, doc_ids):
    q_tokens = engine.pre(query)
    print(f"Query: {query!r} -> token: {q_tokens}")
    for i, d in enumerate(doc_ids):
        e = engine.tfidf.explain(q_tokens, d)
        if e is None:
            print("Tidak ada term query yang dikenal index.")
            return
        if i == 0:
            print("\nBobot query (w_q):")
            show(e["query_table"])
            print(f"||q|| = {e['norm_q']:.4f}")
        print(f"\nDokumen {d} | {engine.info.loc[d, 'title'][:80]}")
        show(e["doc_table"])
        print(f"  sum(w_q*w_d) = {e['dot']:.4f} | ||d|| = {e['norm_d']:.4f} | "
              f"cosine = {e['dot']:.4f} / ({e['norm_d']:.4f} x {e['norm_q']:.4f}) = {e['cosine']:.4f} "
              f"| cek retriever: {e['check']:.4f}")
        print(f"  vektor dokumen: {e['n_dims']} dimensi tak-nol dari {len(engine.index.postings)}; bobot terbesar: "
              + ", ".join(f"{t}={w:.3f}" for t, w in e["top_terms"]))


def main():
    ap = argparse.ArgumentParser(description="Tugas 2 - pipeline lengkap")
    ap.add_argument("--data", default=None, help="path dataset_crawling_ir.csv (default: data/dataset_crawling_ir.csv)")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):                      # agar karakter non-ASCII aman di terminal Windows
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    H.mkdir(exist_ok=True)
    log = open(H / "run_log.txt", "w", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, log)
    try:
        run(args)
    except FileNotFoundError as e:                         # mis. dataset belum disalin ke data/
        print(f"\nERROR: {e}")
        sys.exit(1)
    finally:
        sys.stdout = real_stdout
        log.close()
    print(f"\nLog lengkap tersimpan di: {H / 'run_log.txt'}")


def run(args):
    # ------------------------------------------------------------------ header
    section("TUGAS 2 - RETRIEVAL DATA DAN MESIN PENCARI SEDERHANA | LOG EKSEKUSI")
    print("Waktu eksekusi :", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Python / OS    :", platform.python_version(), "/", platform.platform())
    print("pandas / numpy :", pd.__version__, "/", np.__version__)
    print(f"Konfigurasi    : stemming={config.USE_STEMMING} stopword={config.REMOVE_STOPWORDS} "
          f"keep_words={sorted(config.KEEP_WORDS)} k1={config.K1} b={config.B} top_k={config.TOP_K} pool={config.POOL_DEPTH}")

    # ------------------------------------------------------------------ 2. data
    section("2. KORPUS DAN PEMBERSIHAN DATA")
    engine = SearchEngine(args.data)
    df, info, pre, index = engine.df, engine.info, engine.pre, engine.index
    show(pd.DataFrame(engine.clean_log, columns=["langkah", "sisa dokumen"]))
    print("\nKolom:", list(df.columns))
    show(df[["doc_id", "title", "url"]].head())

    # ------------------------------------------------------------------ 3. preprocessing
    section("3. PREPROCESSING")
    print("Tahap: case folding -> cleaning (URL, angka, tanda baca) -> tokenisasi -> stopword -> stemming (Sastrawi)")
    print("'naik' di daftar stopword Sastrawi murni :", "naik" in STOPWORDS_PURE)
    print("'naik' dibuang oleh sistem ini            :", "naik" in STOPWORDS, "(KEEP_WORDS: sinyal arah harga)")

    cand = [d for d, t in engine.docs_tokens.items() if "beras" in t and "naik" in t]
    sid = cand[0] if cand else df.iloc[0]["doc_id"]
    s = pre.steps(doc_text({"title": info.loc[sid, "title"], "content": info.loc[sid, "content"]})[:380])
    print(f"\nContoh dokumen {sid}: {info.loc[sid, 'title']}")
    print("\nSEBELUM (teks asli)   :", s["raw"])
    print("\nCASE FOLDING+CLEANING :", s["cleaned"])
    print("\nTOKENISASI            :", s["tokens"][:25])
    print("\nSETELAH STOPWORD      :", s["no_stopword"][:25])
    print("\nSETELAH STEMMING      :", s["stemmed"][:25])
    print(f"\nJumlah token: tokenisasi = {len(s['tokens'])} -> stopword = {len(s['no_stopword'])} -> akhir = {len(s['stemmed'])}")

    nostop = Preprocessor(stem=False, stop=False, min_len=config.MIN_TOKEN_LEN)
    nostem = Preprocessor(stem=False, stop=True, min_len=config.MIN_TOKEN_LEN)
    raw_vocab, nostem_vocab, raw_sets, tot_raw = set(), set(), {}, 0
    for _, r in df.iterrows():
        t = doc_text(r)
        a, b = nostop(t), nostem(t)
        raw_vocab.update(a); nostem_vocab.update(b); tot_raw += len(a); raw_sets[r["doc_id"]] = set(a)
    final_vocab = set(t for toks in engine.docs_tokens.values() for t in toks)
    lens = pd.Series({d: len(t) for d, t in engine.docs_tokens.items()})
    print("\nUkuran vocabulary per tahap:")
    show(pd.DataFrame({"Tahap": ["Tokenisasi saja", "+ stopword removal", "+ stemming (akhir)"],
                       "Ukuran vocabulary": [len(raw_vocab), len(nostem_vocab), len(final_vocab)]}))
    print(f"Token total (tanpa filter) = {tot_raw} | token total (akhir) = {int(lens.sum())}")
    print(f"Panjang dokumen (token): mean={lens.mean():.1f} median={lens.median():.0f} min={lens.min()} max={lens.max()}")
    print("Dokumen kosong setelah preprocessing:", int((lens == 0).sum()))

    print("\nDampak stemming - variasi kata 'naik' disatukan menjadi satu term:")
    variants = ["naik", "kenaikan", "menaikkan", "dinaikkan", "naikkan"]
    show(pd.DataFrame([{"kata asli": w, "stem": stem_word(w),
                        "dokumen memuat kata asli (tanpa stemming)": sum(w in s_ for s_ in raw_sets.values())}
                       for w in variants]))
    print("Setelah stemming, dokumen yang memuat term 'naik':", sum("naik" in set(t) for t in engine.docs_tokens.values()))
    print("Query 'harga beras naik' diproses menjadi:", pre("harga beras naik"))

    out = save_preprocessed(df, engine.docs_tokens, pre)
    print(f"\nDisimpan: {config.PREPROCESSED_PATH} {out.shape}")

    # ------------------------------------------------------------------ 4. index
    section("4. INVERTED INDEX")
    print(f"Jumlah dokumen (N) = {index.N} | vocabulary = {len(index.postings)} | "
          f"avgdl = {index.avgdl:.1f} token | total postings = {sum(index.df.values())}")
    top_df = pd.Series(index.df).sort_values(ascending=False).head(12).rename("df").to_frame()
    top_df["df/N"] = top_df["df"] / index.N
    print("\n12 term dengan df tertinggi:")
    show(top_df, index=True, decimals=3)
    print("\nContoh dictionary, postings list, TF dan DF:")
    show(index_example(index, pre, ["beras", "cabai", "bulog", "inflasi", "naik", "satgas", "sidak"]))
    save_index_json(index, pre)
    pd.DataFrame([{"term": t, "df": index.df[t], "postings": " ".join(f"{d}:{c}" for d, c in sorted(p.items()))}
                  for t, p in sorted(index.postings.items())]).to_csv(
        config.INDEX_PATH.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"\nDisimpan: {config.INDEX_PATH} dan {config.INDEX_PATH.with_suffix('.csv')}")

    # ------------------------------------------------------------------ 5. boolean
    section("5. BOOLEAN RETRIEVAL (AND, OR, NOT, tanda kurung; prioritas NOT > AND > OR)")
    for q in ["beras AND harga", "beras AND harga AND naik", "cabai AND (naik OR turun)",
              "beras AND NOT cabai", "inflasi OR bulog"]:
        res = engine.boolean.evaluate(q)
        print(f"{q:<32} -> {len(res):>3} dokumen | {res[:6]}{' ...' if len(res) > 6 else ''}")

    # ------------------------------------------------------------------ 6. tf-idf
    section("6. TF-IDF, VECTOR SPACE MODEL, COSINE SIMILARITY")
    print("TF      = 1 + log10(tf)         IDF = log10(N/df)         w(t,d) = TF x IDF")
    print("cosine  = sum(w_q * w_d) / (||q|| * ||d||)\n")
    ex_q = "harga beras naik"
    ex_rank = rank_docs(engine.tfidf.score(pre(ex_q)))
    print_tfidf_explanation(engine, ex_q, [ex_rank[0], ex_rank[1], ex_rank[min(20, len(ex_rank) - 1)]])

    # ------------------------------------------------------------------ 7. bm25
    section("7. BM25 DAN ANALISISNYA (TF, IDF, PANJANG DOKUMEN, SATURASI)")
    print("score = sum IDF(t) * tf*(k1+1) / (tf + k1*(1 - b + b*|d|/avgdl)) ; IDF = ln(1 + (N-df+0,5)/(df+0,5))")
    print(f"Parameter: k1 = {engine.bm25.k1}, b = {engine.bm25.b}\n")
    bm25_analysis(engine, H / "bm25_saturasi.png")

    # ------------------------------------------------------------------ 8. sanity
    section("8. UJI KEBENARAN IMPLEMENTASI")
    sanity_tests.run_all(engine)

    # ------------------------------------------------------------------ 9. search engine
    section("9. MESIN PENCARI SEDERHANA - contoh penggunaan (antarmuka web: streamlit run app.py)")
    run_demo(engine)

    # ------------------------------------------------------------------ 10. exp 1
    section("10. EKSPERIMEN 1 - Top-5 dan skor, TF-IDF+cosine vs BM25 untuk 10 query uji")
    queries = load_queries()
    results = build_results(engine, queries)
    for q in queries:
        r = results[q["id"]]
        print(f"{q['id']:<4} {q['text']:<30} token={r['tokens']} | boolean: {len(r['boolean']):>3} dok | "
              f"tfidf/bm25 cocok: {len(r['s_tfidf']):>3}/{len(r['s_bm25']):>3}")
    experiment1(engine, results, queries, k=5)
    export_topk(engine, results, queries, H / "hasil_ranking_topk.csv")
    print(f"\nDisimpan: {H / 'hasil_ranking_topk.csv'} (top-10 lengkap dengan URL dan cuplikan)")

    # ------------------------------------------------------------------ 11. qrels
    section("11. POOLING DAN RELEVANCE JUDGMENT")
    pool = build_pool(results, queries)
    qrels, mode, source, template, n_missing = load_labels(engine, queries, pool)
    n_new = int((template["status"] == "BARU").sum())
    print(f"Sumber label: {source} | QRELS_MODE = {mode}")
    print(f"Template v2: {len(template)} baris | BARU (perlu dilabeli): {n_new} | sudah berlabel: {len(template) - n_new}")
    sm = template.assign(is_new=(template["status"] == "BARU")).groupby("query_id").agg(
        baris=("doc_id", "size"), BARU=("is_new", "sum")).reindex([q["id"] for q in queries])
    show(sm, index=True)
    print(f"Disimpan: {config.QRELS_V2_TEMPLATE}")
    if n_new:
        print(f"Langkah: isi 'relevance' (0/1/2) pada baris status=BARU -> simpan sebagai "
              f"{config.QRELS_V2_PATH.name} -> jalankan ulang python main.py")
    if mode == "DEMO-OTOMATIS":
        print("\n" + "!" * 90)
        print("MODE DEMO: label otomatis berbasis kata kunci (MELINGKAR). JANGAN laporkan angka evaluasi di bawah.")
        print("!" * 90)
    show(pd.DataFrame({
        "dinilai": {qid: len(g) for qid, g in qrels.items()},
        "relevan (>=1)": {qid: sum(v >= 1 for v in g.values()) for qid, g in qrels.items()},
        "sangat relevan (=2)": {qid: sum(v == 2 for v in g.values()) for qid, g in qrels.items()},
    }).reindex([q["id"] for q in queries]).T, index=True)

    # ------------------------------------------------------------------ 12. evaluation
    section(f"12. EVALUASI | mode label: {mode}")
    per_query, summary = run_eval(queries, results, qrels, thr=1)
    per_query_strict, summary_strict = run_eval(queries, results, qrels, thr=2, warn=False)
    print("Ringkasan - relevansi LONGGAR (grade >= 1), rata-rata atas query:")
    show(summary, index=True)
    print("\nRingkasan - relevansi KETAT (grade = 2):")
    show(summary_strict, index=True)

    print("\nCakupan penilaian (persen top-10 yang sudah berlabel; sisanya dianggap tidak relevan):")
    cov = coverage_table(queries, results, qrels)
    show(cov.T, index=True, decimals=2)
    print("Rata-rata cakupan -> TF-IDF: %.3f | BM25: %.3f" % (cov["tfidf"].mean(), cov["bm25"].mean()))
    if (cov < 1).any().any():
        print("Ada dokumen top-10 yang belum berlabel (lihat bagian 11): metrik bisa meremehkan metode terkait.")

    print("\nBoolean retrieval (set-based; presisi hanya pada dokumen hasil yang berlabel):")
    bool_df = boolean_table(queries, results, qrels)
    show(bool_df)
    print("Rata-rata Boolean -> Precision(berlabel): %.4f | Recall: %.4f | F1: %.4f" % (
        bool_df["Precision(berlabel)"].mean(), bool_df["Recall"].mean(), bool_df["F1"].mean()))

    print("\nPer-query AP, nDCG@10, P@10, F1@10 (TF-IDF vs BM25):")
    pv = per_query[per_query["method"].isin(["tfidf", "bm25"])].pivot(
        index="query_id", columns="method", values=["AP", "nDCG@10", "P@10", "F1@10"])
    show(pv.reindex([q["id"] for q in queries if q["id"] in pv.index]), index=True, decimals=3)

    print("\nUji signifikansi (bootstrap berpasangan atas query, 5.000 resampling; CI memuat 0 = selisih tidak bermakna):")
    print("Relevansi longgar:")
    sig = sig_table(per_query)
    show(sig, index=True)
    if not per_query_strict.empty:
        print("Relevansi ketat:")
        show(sig_table(per_query_strict), index=True)
    plot_summary(summary, H / "perbandingan_metode.png", f"TF-IDF vs BM25 vs Boolean->BM25 ({mode[:22]})")
    print(f"\nGrafik disimpan: {H / 'perbandingan_metode.png'}")

    # ------------------------------------------------------------------ 13. exp 2
    section("13. EKSPERIMEN 2 - Perbedaan ranking VSM cosine vs BM25")
    experiment2(engine, results, queries, qrels, per_query)

    # ------------------------------------------------------------------ 14. exp 3
    section("14. EKSPERIMEN 3 - Query pendek dan spesifik (pengaruh tf dan panjang dokumen)")
    s3, _ = experiment3(engine, results, queries)

    # ------------------------------------------------------------------ 15. analisis
    section("15. ANALISIS LANJUTAN")
    print("(a) Sensitivitas parameter BM25 (MAP, relevansi longgar):")
    sensitivity_grid(engine, queries, qrels)
    print("\n(b) Ablasi preprocessing (pool dari konfigurasi default -> konfigurasi lain sedikit dirugikan):")
    ablation(engine, queries, qrels)
    print("\n(c) Analisis kegagalan: dua query dengan AP BM25 terendah")
    if len(per_query):
        for qid in per_query[per_query["method"] == "bm25"].sort_values("AP").head(2)["query_id"]:
            inspect_query(engine, results, queries, qrels, qid, "bm25")

    # ------------------------------------------------------------------ 16. ekspor
    section("16. EKSPOR HASIL")
    qrows = []
    for q, row in zip(queries, load_queries_rows(queries)):
        g = qrels.get(q["id"], {})
        qrows.append({**row,
                      "n_dinilai": len(g), "n_relevan": sum(v >= 1 for v in g.values()),
                      "n_sangat_relevan": sum(v == 2 for v in g.values()),
                      "doc_relevan": " ".join(sorted(d for d, v in g.items() if v >= 1)),
                      "doc_sangat_relevan": " ".join(sorted(d for d, v in g.items() if v == 2))})
    pd.DataFrame(qrows).to_csv(config.QUERIES_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame([{"query_id": qid, "doc_id": d, "relevance": v} for qid, g in qrels.items()
                  for d, v in sorted(g.items())]).to_csv(config.QRELS_FINAL_PATH, index=False, encoding="utf-8-sig")
    per_query.to_csv(H / "metrik_per_query.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(H / "metrik_ringkasan_longgar.csv", encoding="utf-8-sig")
    if not summary_strict.empty:
        summary_strict.to_csv(H / "metrik_ringkasan_ketat.csv", encoding="utf-8-sig")
    bool_df.to_csv(H / "metrik_boolean.csv", index=False, encoding="utf-8-sig")
    sig.to_csv(H / "metrik_signifikansi_bootstrap.csv", encoding="utf-8-sig")
    s3.to_csv(H / "eksperimen3_query_pendek.csv", encoding="utf-8-sig")
    for p in [config.PREPROCESSED_PATH, config.INDEX_PATH, config.QUERIES_PATH, config.QRELS_FINAL_PATH,
              config.QRELS_V2_TEMPLATE, H / "hasil_ranking_topk.csv", H / "metrik_per_query.csv",
              H / "metrik_ringkasan_longgar.csv", H / "metrik_boolean.csv", H / "metrik_signifikansi_bootstrap.csv",
              H / "eksperimen3_query_pendek.csv", H / "perbandingan_metode.png", H / "bm25_saturasi.png"]:
        print("  ", p)

    # ------------------------------------------------------------------ ringkasan
    section("RINGKASAN")
    print(f"Dokumen: {index.N} | vocabulary: {len(index.postings)} | mode label: {mode}")
    if not summary.empty:
        print(f"MAP  -> TF-IDF: {summary.loc['tfidf', 'MAP']:.4f} | BM25: {summary.loc['bm25', 'MAP']:.4f}")
        print(f"nDCG@10 -> TF-IDF: {summary.loc['tfidf', 'nDCG@10']:.4f} | BM25: {summary.loc['bm25', 'nDCG@10']:.4f}")
        r = sig.loc["AP"]
        print(f"Selisih AP (BM25 - TF-IDF) = {r['selisih']:.4f}, CI95 [{r['CI95_bawah']:.4f}, {r['CI95_atas']:.4f}] "
              f"-> {'TIDAK ' if r['CI95_bawah'] <= 0 <= r['CI95_atas'] else ''}bermakna pada data ini")
    if mode != "MANUAL":
        print("CATATAN: hasil evaluasi di atas BELUM final (mode label bukan MANUAL lengkap).")


def load_queries_rows(queries):
    """Kolom asli queries.csv (query_id, query_text, boolean_query, must) agar berkas tetap bisa dibaca ulang."""
    return [{"query_id": q["id"], "query_text": q["text"], "boolean_query": q["boolean"],
             "must": ";".join(q["must"])} for q in queries]


if __name__ == "__main__":
    main()
