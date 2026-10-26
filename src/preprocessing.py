"""Tahap 1 - Prapemrosesan.

Urutan: pembersihan data -> case folding -> normalisasi/cleaning -> tokenisasi
        -> stopword removal -> stemming (Sastrawi).
Pipeline yang SAMA dipakai untuk dokumen dan query agar token keduanya bisa dicocokkan.
"""
import hashlib
import re

import pandas as pd

from . import config

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
except ImportError:                                   # Sastrawi belum terpasang
    StemmerFactory = None
    StopWordRemoverFactory = None

# Stopword domain: kata sumber/satuan yang muncul di hampir semua berita dan tidak membawa topik.
CUSTOM_STOPWORDS = {
    "detik", "detikcom", "detikfinance", "com", "rp", "ribu", "juta", "kg", "kilogram",
    "per", "kata", "ujar", "tutur", "jelas", "menurut", "mengatakan", "dikutip",
}
FALLBACK_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "dengan", "adalah",
    "dalam", "juga", "akan", "atau", "oleh", "karena", "sebagai", "ada", "tidak", "sudah",
    "saat", "bisa", "lebih", "masih", "para", "telah", "hari", "satu", "kami", "kita", "naik",
}

BASE_STOPWORDS = (set(StopWordRemoverFactory().get_stop_words())
                  if StopWordRemoverFactory is not None else FALLBACK_STOPWORDS)
STOPWORDS_PURE = BASE_STOPWORDS | CUSTOM_STOPWORDS            # tanpa pengecualian (untuk ablasi)
# 'naik' ada di daftar Sastrawi, padahal pada berita harga ia sinyal utama (lawan 'turun'):
STOPWORDS = STOPWORDS_PURE - set(config.KEEP_WORDS)           # yang dipakai sistem

_stemmer = StemmerFactory().create_stemmer() if StemmerFactory is not None else None
_stem_cache = {}                                              # stemming mahal -> simpan hasilnya
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def stem_word(w):
    if _stemmer is None:
        return w
    if w not in _stem_cache:
        _stem_cache[w] = _stemmer.stem(w)
    return _stem_cache[w]


class Preprocessor:
    def __init__(self, stem=True, stop=True, min_len=2, stopwords=None):
        self.stem = stem and (_stemmer is not None)
        self.stop = stop
        self.min_len = min_len
        self.stopwords = STOPWORDS if stopwords is None else stopwords
        if stem and _stemmer is None:
            print("PERINGATAN: Sastrawi tidak terpasang -> stemming dinonaktifkan (pip install PySastrawi)")

    def clean(self, text):
        text = str(text).lower()                      # 1) case folding
        text = URL_RE.sub(" ", text)                  # 2) normalisasi: hapus URL,
        text = re.sub(r"\d+", " ", text)              #    angka,
        text = re.sub(r"[^a-z\s]", " ", text)         #    tanda baca dan simbol
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text):
        return [t for t in self.clean(text).split(" ") if len(t) >= self.min_len]     # 3) tokenisasi

    def __call__(self, text):
        toks = self.tokenize(text)
        if self.stop:
            toks = [t for t in toks if t not in self.stopwords]                        # 4) stopword
        if self.stem:
            toks = [stem_word(t) for t in toks]                                        # 5) stemming
            if self.stop:                                  # hasil stem bisa menjadi stopword
                toks = [t for t in toks if t not in self.stopwords and len(t) >= self.min_len]
        return toks

    def steps(self, text):
        """Hasil tiap tahap, untuk contoh sebelum/sesudah pada laporan."""
        raw = str(text)
        tokens = self.tokenize(raw)
        no_stop = [t for t in tokens if t not in self.stopwords] if self.stop else tokens
        stemmed = [stem_word(t) for t in no_stop] if self.stem else no_stop
        return {"raw": raw, "cleaned": self.clean(raw), "tokens": tokens,
                "no_stopword": no_stop, "stemmed": stemmed}


def default_preprocessor():
    return Preprocessor(stem=config.USE_STEMMING, stop=config.REMOVE_STOPWORDS, min_len=config.MIN_TOKEN_LEN)


def doc_text(row, title_repeat=None):
    n = config.TITLE_REPEAT if title_repeat is None else title_repeat
    return " ".join([row["title"]] * n + [row["content"]])


def load_and_clean(path=None):
    """Muat CSV Tugas 1 lalu buang nilai kosong dan duplikat. doc_id TIDAK dinomori ulang
    agar label relevansi yang sudah dibuat tetap cocok."""
    path = config.DATA_PATH if path is None else path
    try:
        raw = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}\n"
            "Salin dataset_crawling_ir.csv (hasil Tugas 1) ke folder data/ lalu jalankan ulang.")
    missing = [c for c in ["doc_id", "title", "url", "content"] if c not in raw.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ada di dataset: {missing}")

    log = [("dokumen awal", len(raw))]
    df = raw.dropna(subset=["doc_id"]).copy()
    df["doc_id"] = df["doc_id"].astype(str).str.strip()
    for c in ["title", "content", "url"]:
        df[c] = df[c].fillna("").astype(str).map(lambda s: re.sub(r"\s+", " ", s).strip())

    df = df[(df["doc_id"] != "") & (df["title"] != "") & (df["content"] != "")]
    log.append(("buang doc_id/judul/isi kosong", len(df)))
    df = df.drop_duplicates(subset=["doc_id"]);                    log.append(("buang duplikat doc_id", len(df)))
    df = df.drop_duplicates(subset=["url"]);                       log.append(("buang duplikat URL", len(df)))
    df = df.loc[~df["title"].str.casefold().duplicated()];         log.append(("buang duplikat judul", len(df)))
    key = df["content"].str[:500].str.casefold().map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
    df = df.loc[~key.duplicated()].reset_index(drop=True);         log.append(("buang duplikat isi (500 karakter awal)", len(df)))
    assert df["doc_id"].is_unique
    return df, log


def build_docs_tokens(df, pre):
    return {r["doc_id"]: pre(doc_text(r)) for _, r in df.iterrows()}


def save_preprocessed(df, docs_tokens, pre, path=None):
    path = config.PREPROCESSED_PATH if path is None else path
    keep = [c for c in ["doc_id", "title", "url", "content", "crawl_date"] if c in df.columns]
    out = df[keep].copy()
    out["text_cleaned"] = [pre.clean(doc_text(r)) for _, r in df.iterrows()]
    out["tokens"] = [" ".join(docs_tokens[d]) for d in df["doc_id"]]
    out["n_tokens"] = [len(docs_tokens[d]) for d in df["doc_id"]]
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    # Demo modul: python -m src.preprocessing
    df, log = load_and_clean()
    pre = default_preprocessor()
    r = df.iloc[0]
    s = pre.steps(doc_text(r)[:300])
    print("Dokumen:", r["doc_id"], "|", r["title"])
    for k in ["raw", "cleaned", "tokens", "no_stopword", "stemmed"]:
        print(f"\n{k.upper():<12}:", s[k] if k in ("raw", "cleaned") else s[k][:20])
