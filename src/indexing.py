"""Tahap 2 - Inverted index.

Struktur: term -> {doc_id: tf}. Kunci kamus = term, nilainya = postings list (ID dokumen + frekuensi).
Dari sini dihitung df (jumlah dokumen yang memuat term), panjang dokumen, dan avgdl.
"""
import json
from collections import Counter, defaultdict

import pandas as pd

from . import config


class InvertedIndex:
    def __init__(self, docs_tokens):
        self.doc_ids = list(docs_tokens)
        self.N = len(self.doc_ids)
        self.doc_len = {}
        postings = defaultdict(dict)
        for d, toks in docs_tokens.items():
            self.doc_len[d] = len(toks)
            for t, c in Counter(toks).items():
                postings[t][d] = c
        self.postings = dict(postings)
        self.df = {t: len(p) for t, p in self.postings.items()}
        self.avgdl = sum(self.doc_len.values()) / self.N

    def get(self, term):
        return self.postings.get(term, {})


def index_example(index, pre, words, max_docs=6):
    """Contoh dictionary, postings list, TF dan DF untuk beberapa kata."""
    rows = []
    for w in words:
        t = pre(w)
        if not t:
            rows.append({"kata": w, "term": "(dibuang stopword)"})
            continue
        t = t[0]
        p = index.get(t)
        by_tf = sorted(p.items(), key=lambda x: (-x[1], x[0]))
        rows.append({
            "kata": w, "term": t, "DF": len(p), "total TF": sum(p.values()),
            "postings (doc_id)": "[" + ", ".join(sorted(p)[:max_docs]) + (", ..." if len(p) > max_docs else "") + "]",
            "TF tertinggi (doc:tf)": ", ".join(f"{d}:{c}" for d, c in by_tf[:3]),
        })
    return pd.DataFrame(rows)


def save_index_json(index, pre, path=None):
    path = config.INDEX_PATH if path is None else path
    payload = {
        "meta": {"N": index.N, "vocab_size": len(index.postings), "avgdl": index.avgdl,
                 "stemming": pre.stem, "stopword": pre.stop, "keep_words": sorted(config.KEEP_WORDS)},
        "doc_len": index.doc_len,
        "index": {t: {"df": index.df[t], "postings": p} for t, p in sorted(index.postings.items())},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


if __name__ == "__main__":
    # Demo modul: python -m src.indexing
    from .preprocessing import load_and_clean, default_preprocessor, build_docs_tokens
    df, _ = load_and_clean()
    pre = default_preprocessor()
    ix = InvertedIndex(build_docs_tokens(df, pre))
    print(f"N={ix.N} | vocabulary={len(ix.postings)} | avgdl={ix.avgdl:.1f}")
    print(index_example(ix, pre, ["beras", "cabai", "bulog", "naik"]).to_string(index=False))
