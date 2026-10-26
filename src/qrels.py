"""Query uji, pooling, dan relevance judgment (qrels).

Pooling: dokumen yang dinilai = gabungan top-POOL_DEPTH TF-IDF, BM25, dan Boolean (dipotong).
Jika preprocessing berubah, pool berubah. Modul ini memakai ulang label lama: template v2 berisi semua
pasangan pool baru; pasangan yang sudah pernah dilabeli terisi otomatis (status 'lama'),
yang belum dinilai ditandai 'BARU' -> hanya baris BARU yang perlu diisi.
"""
import glob
import os
from collections import defaultdict

import pandas as pd

from . import config


# ---------------------------------------------------------------- query
def load_queries(path=None):
    """Baca evaluation/queries.csv (dibuat dari DEFAULT_QUERIES jika belum ada)."""
    path = config.QUERIES_PATH if path is None else path
    if not os.path.exists(path):
        pd.DataFrame(config.DEFAULT_QUERIES,
                     columns=["query_id", "query_text", "boolean_query", "must"]).to_csv(path, index=False, encoding="utf-8-sig")
    q = pd.read_csv(path)
    out = []
    for r in q.itertuples(index=False):
        must = str(getattr(r, "must", "")) if pd.notna(getattr(r, "must", None)) else ""
        out.append({"id": str(r.query_id), "text": str(r.query_text), "boolean": str(r.boolean_query),
                    "must": [w for w in must.split(";") if w]})
    return out


def build_results(engine, queries):
    """Jalankan TF-IDF, BM25, Boolean, dan Boolean->BM25 untuk semua query."""
    from .utils import rank_docs
    results = {}
    for q in queries:
        qt = engine.pre(q["text"])
        s_tf, s_bm = engine.tfidf.score(qt), engine.bm25.score(qt)
        bl = engine.boolean.evaluate(q["boolean"])
        bl_set = set(bl)
        results[q["id"]] = {
            "tfidf": rank_docs(s_tf), "bm25": rank_docs(s_bm), "boolean": bl,
            "boolean_bm25": [d for d in rank_docs(s_bm) if d in bl_set],     # filter Boolean, urut BM25
            "s_tfidf": s_tf, "s_bm25": s_bm, "tokens": qt,
        }
    return results


# ---------------------------------------------------------------- pooling
def pooled_docs(results, qid, depth=config.POOL_DEPTH):
    r = results[qid]
    pool = []
    for d in r["tfidf"][:depth] + r["bm25"][:depth]:
        if d not in pool:
            pool.append(d)
    bl = sorted(r["boolean"], key=lambda d: (-r["s_bm25"].get(d, 0.0), d))[:depth]
    for d in bl:
        if d not in pool:
            pool.append(d)
    return pool


def build_pool(results, queries, depth=config.POOL_DEPTH):
    return {q["id"]: pooled_docs(results, q["id"], depth) for q in queries}


# ---------------------------------------------------------------- berkas label
def read_qrels_file(path):
    try:
        q = pd.read_csv(path)
    except Exception:
        return None
    if not {"query_id", "doc_id", "relevance"} <= set(q.columns):
        return None
    q = q.copy()
    q["query_id"] = q["query_id"].astype(str).str.strip()
    q["doc_id"] = q["doc_id"].astype(str).str.strip()
    q["relevance"] = pd.to_numeric(q["relevance"], errors="coerce")
    return q


def labels_from_df(q):
    lab = q.dropna(subset=["relevance"])
    assert lab["relevance"].isin([0, 1, 2]).all(), "relevance harus 0, 1, atau 2"
    return {(r.query_id, r.doc_id): int(r.relevance) for r in lab.itertuples(index=False)}


def find_old_qrels():
    """Cari berkas label lama (*qrels*.csv di folder evaluation/ dan root proyek) yang kolom relevance-nya terisi."""
    skip = {config.QRELS_V2_TEMPLATE.name, config.QRELS_V2_PATH.name, config.QRELS_FINAL_PATH.name}
    best = None
    for folder in (config.ROOT / "evaluation", config.ROOT):
        for p in sorted(glob.glob(str(folder / "*qrels*.csv"))):
            if os.path.basename(p) in skip:
                continue
            q = read_qrels_file(p)
            if q is None:
                continue
            n = int(q["relevance"].notna().sum())
            if n and (best is None or n > best[0]):
                best = (n, p, q)
    return best


def build_template_v2(engine, queries, pool, old_labels, path=None):
    """Tulis qrels_template_v2.csv: label lama terisi, pasangan baru kosong (status BARU)."""
    path = config.QRELS_V2_TEMPLATE if path is None else path
    info = engine.info
    qmap = {q["id"]: q for q in queries}
    qorder = {q["id"]: i for i, q in enumerate(queries)}

    def row(q, d, status, rel):
        return {"query_id": q["id"], "query_text": q["text"], "doc_id": d, "status": status,
                "title": info.loc[d, "title"], "url": info.loc[d, "url"],
                "snippet": info.loc[d, "content"][:300], "relevance": rel}

    rows, seen = [], set()
    for q in queries:
        for d in pool[q["id"]]:
            key = (q["id"], d)
            seen.add(key)
            rows.append(row(q, d, "lama" if key in old_labels else "BARU", old_labels.get(key, "")))
    for (qid, d), rel in old_labels.items():
        if (qid, d) not in seen and qid in qmap and d in info.index:
            rows.append(row(qmap[qid], d, "lama (di luar pool baru)", rel))

    t = pd.DataFrame(rows)
    t["_o"] = t["query_id"].map(qorder)
    t["_s"] = (t["status"] != "BARU").astype(int)
    t = t.sort_values(["_o", "_s", "doc_id"]).drop(columns=["_o", "_s"]).reset_index(drop=True)
    t.to_csv(path, index=False, encoding="utf-8-sig")
    return t


def demo_labels(engine, queries, pool):
    """Label OTOMATIS berbasis kata kunci -> MELINGKAR, hanya agar program bisa jalan tanpa label manual."""
    tok_sets = {d: set(t) for d, t in engine.docs_tokens.items()}
    out = {}
    for q in queries:
        must = {t for w in q["must"] for t in engine.pre(w)}
        for d in pool[q["id"]]:
            hit = len(must & tok_sets[d])
            out[(q["id"], d)] = 2 if must and hit == len(must) else (1 if hit > 0 else 0)
    return out


def load_labels(engine, queries, pool):
    """Pilih sumber label dan buat template v2. Kembalikan (QRELS, mode, sumber, template_df, n_belum_dilabeli)."""
    old = find_old_qrels()
    old_labels = labels_from_df(old[2]) if old else {}
    v2_exists = os.path.exists(config.QRELS_V2_PATH)
    if v2_exists:
        v2 = read_qrels_file(config.QRELS_V2_PATH)
        if v2 is not None:
            old_labels.update(labels_from_df(v2))           # label v2 (lebih baru) menimpa label lama

    template = build_template_v2(engine, queries, pool, old_labels)

    if v2_exists:
        v2 = read_qrels_file(config.QRELS_V2_PATH)
        if v2 is None:
            raise ValueError(f"{config.QRELS_V2_PATH.name} harus punya kolom query_id, doc_id, relevance")
        labels, source = labels_from_df(v2), str(config.QRELS_V2_PATH.name)
    elif old_labels:
        labels, source = dict(old_labels), os.path.basename(old[1])
    else:
        labels, source = demo_labels(engine, queries, pool), "demo"

    qrels = defaultdict(dict)
    for (qid, d), g in labels.items():
        qrels[qid][d] = g
    pairs = [(q["id"], d) for q in queries for d in pool[q["id"]]]
    n_missing = sum(1 for p in pairs if p not in labels)

    if source == "demo":
        mode = "DEMO-OTOMATIS"
    elif n_missing:
        mode = f"MANUAL-BELUM-LENGKAP ({n_missing} pasangan pool belum berlabel)"
    else:
        mode = "MANUAL"
    return dict(qrels), mode, source, template, n_missing
