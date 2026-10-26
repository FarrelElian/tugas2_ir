# Tugas 2 — Retrieval Data dan Mesin Pencari Sederhana

> **Hasil eksekusi sudah tersimpan** di `hasil/run_log.txt` (seluruh output terminal), beserta CSV dan grafik
> di `hasil/`. langsung dapat membaca hasil tanpa menjalankan ulang; untuk mereproduksi, jalankan `python main.py`.
> Jika dataset atau label diubah, jalankan ulang agar log ikut diperbarui.

## Struktur folder
```
tugas2_ir/
├── data/         dataset_crawling_ir.csv (hasil Tugas 1), dataset_preprocessed.csv (dibuat otomatis)
├── index/        inverted_index.json, inverted_index.csv
├── src/
│   ├── config.py             semua parameter dan lokasi berkas
│   ├── preprocessing.py      pembersihan data, case folding, tokenisasi, stopword, stemming
│   ├── indexing.py           inverted index (term -> {doc_id: tf})
│   ├── boolean_retrieval.py  AND / OR / NOT / tanda kurung
│   ├── tfidf_retrieval.py    TF-IDF, vektor, cosine similarity
│   ├── bm25.py               BM25 (k1 = 1,2; b = 0,75)
│   ├── search_engine.py      alur query -> Top-k, cuplikan, CLI terminal
│   ├── qrels.py              query uji, pooling, label relevansi
│   ├── evaluation.py         P@k, R@k, F1@k, MAP, MRR, nDCG, bootstrap
│   ├── experiments.py        Eksperimen 1-3, analisis BM25, sensitivitas, ablasi
│   └── sanity_tests.py       uji kebenaran implementasi
├── evaluation/   queries.csv, qrels_template_v2.csv, qrels_labeled_v2.csv, qrels_final.csv
├── hasil/        run_log.txt, hasil_ranking_topk.csv, metrik_*.csv, grafik *.png
├── main.py       menjalankan seluruh pipeline dan MENCETAK setiap hasil
├── app.py        antarmuka web (Streamlit)
└── requirements.txt
```

## Instalasi
Python 3.9 atau lebih baru.
```
pip install -r requirements.txt
```

## Cara menjalankan
| Tujuan | Perintah |
|---|---|
| Seluruh pipeline + evaluasi + log | `python main.py` |
| Antarmuka web | `streamlit run app.py` |
| Mesin pencari di terminal | `python -m src.search_engine` (tambah `--demo` untuk contoh tanpa input) |
| Demo satu modul | `python -m src.preprocessing`, `python -m src.indexing`, `python -m src.boolean_retrieval`, `python -m src.tfidf_retrieval`, `python -m src.bm25`, `python -m src.sanity_tests` |

## Label relevansi (wajib untuk evaluasi yang valid)
Evaluasi membutuhkan penilaian relevansi buatan manusia.
1. Jalankan `python main.py`. Program membuat `evaluation/qrels_template_v2.csv`.
   Label lama (berkas `*qrels*.csv` di folder `evaluation/` atau root) dipakai ulang otomatis.
2. Isi kolom `relevance` pada baris berstatus `BARU`: `2` = sangat relevan, `1` = relevan sebagian, `0` = tidak relevan.
3. Simpan sebagai `evaluation/qrels_labeled_v2.csv`, lalu jalankan ulang `python main.py`.
4. Pastikan log menampilkan `QRELS_MODE = MANUAL`. Selama `DEMO-OTOMATIS` atau `MANUAL-BELUM-LENGKAP`, hasil belum final.

Jangan melabeli langsung di `qrels_template_v2.csv` (berkas itu dibuat ulang setiap kali `main.py` dijalankan).

## Contoh query
| Query | Metode |
|---|---|
| `harga beras naik` | bm25 / tfidf |
| `operasi pasar minyak goreng` | bm25 / tfidf |
| `bulog AND stok AND beras` | boolean |
| `cabai AND (turun OR anjlok)` | boolean |
| `kucing terbang` | contoh "dokumen tidak ditemukan" |

## Alur data (query sampai Top-k)
1. Query pengguna diproses dengan pipeline yang sama seperti dokumen (case folding, cleaning, stopword, stemming).
2. Postings tiap term query diambil dari inverted index sehingga hanya dokumen kandidat yang dihitung.
3. Skor dihitung: cosine pada bobot TF-IDF, atau BM25, atau himpunan Boolean.
4. Dokumen diurutkan menurut skor menurun.
5. Top-k ditampilkan: nomor peringkat, judul, URL, skor, dan cuplikan kalimat yang paling cocok dengan query.

## Keterbatasan
- `naik` dikecualikan dari stopword (ada di daftar Sastrawi) karena merupakan sinyal arah harga.
- Angka dibuang saat preprocessing, sehingga query berbasis angka tidak didukung.
- Dengan ±10 query, selisih TF-IDF vs BM25 perlu dibaca bersama interval bootstrap pada log bagian 12.
- Recall dihitung terhadap dokumen relevan yang ada di pool; dokumen tak berlabel dianggap tidak relevan.
