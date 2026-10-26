"""Fungsi bantu: log ganda (terminal + berkas), pencetakan tabel, ranking."""
import pandas as pd


class Tee:
    """Menulis ke beberapa aliran sekaligus, agar semua output terminal juga tersimpan di hasil/run_log.txt."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def show(df, index=False, decimals=4, max_colwidth=70):
    """Cetak DataFrame sebagai teks (pengganti display() pada notebook)."""
    if df is None or len(df) == 0:
        print("(kosong)")
        return
    with pd.option_context("display.max_colwidth", max_colwidth, "display.width", 250,
                           "display.max_columns", 50):
        print(df.round(decimals).to_string(index=index))


def rank_docs(scores):
    """Urutkan dokumen menurut skor menurun; seri diurutkan menurut doc_id agar deterministik."""
    return [d for d, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def spearman(x, y):
    """Korelasi peringkat Spearman (tanpa scipy)."""
    sx, sy = pd.Series(list(x)).rank(), pd.Series(list(y)).rank()
    return float(sx.corr(sy)) if len(sx) > 2 else float("nan")
