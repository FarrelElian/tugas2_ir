"""Konfigurasi terpusat Tugas 2. Ubah nilai di sini, bukan di modul lain."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- Lokasi berkas ----
DATA_PATH         = ROOT / "data" / "dataset_crawling_ir.csv"       # hasil Tugas 1 (WAJIB ada)
PREPROCESSED_PATH = ROOT / "data" / "dataset_preprocessed.csv"      # dibuat otomatis
INDEX_PATH        = ROOT / "index" / "inverted_index.json"          # dibuat otomatis
QUERIES_PATH      = ROOT / "evaluation" / "queries.csv"
QRELS_V2_PATH     = ROOT / "evaluation" / "qrels_labeled_v2.csv"    # label manual (diisi kelompok)
QRELS_V2_TEMPLATE = ROOT / "evaluation" / "qrels_template_v2.csv"   # dibuat otomatis
QRELS_FINAL_PATH  = ROOT / "evaluation" / "qrels_final.csv"
HASIL_DIR         = ROOT / "hasil"

# ---- Preprocessing ----
USE_STEMMING     = True
REMOVE_STOPWORDS = True
MIN_TOKEN_LEN    = 2
TITLE_REPEAT     = 1                # judul diulang n kali pada teks dokumen
KEEP_WORDS       = {"naik"}         # kata stopword Sastrawi yang TETAP dipertahankan (sinyal arah harga)

# ---- BM25 (nilai awal sesuai ketentuan tugas) ----
K1, B = 1.2, 0.75

# ---- Retrieval dan evaluasi ----
TOP_K      = 10
POOL_DEPTH = 10
EVAL_KS    = (5, 10)

# Query pendek dan spesifik untuk Eksperimen 3
SHORT_QUERIES = ["bulog", "satgas", "sidak", "impor", "telur", "bawang merah"]

# Query uji bawaan (dipakai jika evaluation/queries.csv tidak ada)
# (id, teks query, query Boolean, kata 'must' untuk mode demo label otomatis)
DEFAULT_QUERIES = [
    ("Q1",  "harga beras naik",            "harga AND beras AND naik",               "beras"),
    ("Q2",  "harga cabai turun",           "cabai AND (turun OR anjlok)",            "cabai"),
    ("Q3",  "operasi pasar minyak goreng", "operasi AND pasar AND minyak AND goreng", "minyak;goreng"),
    ("Q4",  "stok beras Bulog",            "bulog AND stok AND beras",               "bulog;beras"),
    ("Q5",  "inflasi bahan pangan",        "inflasi AND pangan",                     "inflasi"),
    ("Q6",  "harga telur ayam",            "telur AND (ayam OR harga)",              "telur"),
    ("Q7",  "bawang merah naik",           "bawang AND merah AND naik",              "bawang"),
    ("Q8",  "harga eceran tertinggi HET",  "harga AND eceran AND tertinggi",         "eceran"),
    ("Q9",  "impor beras",                 "impor AND beras",                        "impor;beras"),
    ("Q10", "satgas pangan sidak pasar",   "satgas AND pangan AND (sidak OR pasar)", "satgas"),
]
