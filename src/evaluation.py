"""Evaluasi retrieval: P@k, R@k, F1@k, AP/MAP, MRR, nDCG, metrik Boolean, bootstrap berpasangan.

Relevansi biner memakai ambang (thr): 1 = longgar (grade >= 1), 2 = ketat (grade = 2).
nDCG memakai grade 0/1/2 dengan gain 2^g - 1.
Dokumen yang tidak ada di qrels dianggap TIDAK relevan (lihat coverage_table untuk pengaruhnya).
"""
import math

import numpy as np
import pandas as pd

from . import config

METHODS = ["tfidf", "bm25", "boolean_bm25"]


def precision_at_k(ranked, rel, k):
    return sum(d in rel for d in ranked[:k]) / k


def recall_at_k(ranked, rel, k):
    return sum(d in rel for d in ranked[:k]) / len(rel) if rel else float("nan")


def f1(p, r):
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def average_precision(ranked, rel):
    if not rel:
        return float("nan")
    hits, s = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in rel:
            hits += 1
            s += hits / i
    return s / len(rel)


def reciprocal_rank(ranked, rel):
    for i, d in enumerate(ranked, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, grades, k):
    dcg = sum((2 ** grades.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else float("nan")


def eval_ranked(ranked, grades, ks=config.EVAL_KS, thr=1):
    rel = {d for d, g in grades.items() if g >= thr}
    out = {}
    for k in ks:
        p, r = precision_at_k(ranked, rel, k), recall_at_k(ranked, rel, k)
        out[f"P@{k}"], out[f"R@{k}"] = p, r
        out[f"F1@{k}"] = f1(p, r) if not math.isnan(r) else float("nan")
    out["AP"] = average_precision(ranked, rel)
    out["RR"] = reciprocal_rank(ranked, rel)
    out[f"nDCG@{max(ks)}"] = ndcg_at_k(ranked, grades, max(ks))
    return out


def eval_boolean(result, grades, thr=1):
    """Boolean tidak berperingkat -> ukur pada himpunan. Presisi HANYA pada dokumen hasil yang berlabel
    (dokumen tak berlabel tidak dianggap salah, kalau tidak presisi akan bias ke bawah)."""
    rel = {d for d, g in grades.items() if g >= thr}
    got = set(result)
    judged = got & set(grades)
    tp = len(got & rel)
    p = tp / len(judged) if judged else float("nan")
    r = tp / len(rel) if rel else float("nan")
    return {"n_hasil": len(got), "n_berlabel": len(judged), "Precision(berlabel)": p, "Recall": r,
            "F1": f1(p, r) if not (math.isnan(p) or math.isnan(r)) else float("nan")}


def run_eval(queries, results, qrels, thr=1, methods=METHODS, ks=config.EVAL_KS, warn=True):
    """Metrik per query dan ringkasan (rata-rata) untuk setiap metode berperingkat."""
    rows = []
    for q in queries:
        grades = qrels.get(q["id"], {})
        if not any(g >= thr for g in grades.values()):
            if warn:
                print(f"PERINGATAN: {q['id']} tidak punya dokumen relevan (ambang {thr}) -> dilewati")
            continue
        for m in methods:
            rows.append({"query_id": q["id"], "method": m, **eval_ranked(results[q["id"]][m], grades, ks, thr)})
    pq = pd.DataFrame(rows)
    if pq.empty:
        return pq, pd.DataFrame()
    cols = [c for c in pq.columns if c not in ("query_id", "method")]
    summ = pq.groupby("method")[cols].mean().reindex(methods).rename(columns={"AP": "MAP", "RR": "MRR"})
    return pq, summ


def boolean_table(queries, results, qrels, thr=1):
    rows = []
    for q in queries:
        g = qrels.get(q["id"], {})
        if any(v >= thr for v in g.values()):
            rows.append({"query_id": q["id"], "boolean": q["boolean"], **eval_boolean(results[q["id"]]["boolean"], g, thr)})
    return pd.DataFrame(rows)


def coverage_table(queries, results, qrels, k=10):
    """Persentase top-k yang sudah berlabel; sisanya dianggap tidak relevan."""
    rows = []
    for q in queries:
        g = qrels.get(q["id"], {})
        row = {"query_id": q["id"]}
        for m in ("tfidf", "bm25"):
            top = results[q["id"]][m][:k]
            row[m] = sum(d in g for d in top) / max(len(top), 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("query_id")


def paired_bootstrap(a, b, n=5000, seed=0):
    """Bootstrap berpasangan atas query: selisih rata-rata a-b dengan CI 95%."""
    a, b = np.array(a, float), np.array(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    d = a - b
    if len(d) == 0:
        return {k: float("nan") for k in ["n_query", "rata2_BM25", "rata2_TFIDF", "selisih", "CI95_bawah",
                                           "CI95_atas", "menang_BM25", "seri", "menang_TFIDF"]}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_query": len(d), "rata2_BM25": a.mean(), "rata2_TFIDF": b.mean(), "selisih": d.mean(),
            "CI95_bawah": lo, "CI95_atas": hi,
            "menang_BM25": int((d > 1e-9).sum()), "seri": int((abs(d) <= 1e-9).sum()),
            "menang_TFIDF": int((d < -1e-9).sum())}


def sig_table(pq, ks=config.EVAL_KS):
    rows = {}
    for met in ["AP", f"nDCG@{max(ks)}", f"P@{max(ks)}", f"F1@{max(ks)}"]:
        a = pq[pq.method == "bm25"].set_index("query_id")[met]
        b = pq[pq.method == "tfidf"].set_index("query_id")[met].reindex(a.index)
        rows[met] = paired_bootstrap(a.values, b.values)
    return pd.DataFrame(rows).T
