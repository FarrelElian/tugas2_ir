"""Antarmuka web mesin pencari sederhana (Streamlit).

    pip install streamlit
    streamlit run app.py

Alur: query -> preprocessing (sama dengan dokumen) -> inverted index -> skor (cosine TF-IDF / BM25 / Boolean)
      -> ranking -> Top-k (judul, URL, skor, cuplikan).
"""
import pandas as pd
import streamlit as st

from src import config
from src.bm25 import BM25Retriever
from src.search_engine import SearchEngine

st.set_page_config(page_title="Mesin Pencari Berita Pangan", page_icon="🔎", layout="wide")

METHOD_LABELS = {
    "BM25": "bm25",
    "TF-IDF + Cosine": "tfidf",
    "Boolean (AND / OR / NOT)": "boolean",
    "Bandingkan TF-IDF vs BM25": "both",
}
EXAMPLES = ["harga beras naik", "operasi pasar minyak goreng", "stok beras Bulog", "harga cabai turun",
            "kucing terbang"]


@st.cache_resource(show_spinner="Membangun index ...")
def load_engine():
    return SearchEngine()


def render_results(engine, res, k, explain):
    st.caption(f"Metode: **{res['method'].upper()}** | token query: `{res['tokens']}` | "
               f"dokumen cocok: **{len(res['ranked'])}**")
    if res["message"]:
        st.warning(res["message"])
        return
    for r in engine.result_rows(res, k):
        score = "cocok (Boolean tidak berskor)" if r["score"] is None else f"{r['score']:.4f}"
        st.markdown(f"**{r['rank']}. [{r['title']}]({r['url']})**")
        st.caption(f"Skor: {score}  |  {r['doc_id']}  |  {r['url']}")
        st.write(r["snippet"])
        if explain and res["method"] != "boolean":
            with st.expander("Penguraian skor per term"):
                tbl, dl, ln = engine.explain_doc(res["query"], r["doc_id"])
                st.caption(f"panjang dokumen = {dl} (avgdl = {engine.index.avgdl:.1f}) | faktor panjang = {ln:.3f}")
                st.dataframe(tbl.round(4), hide_index=True)
        st.divider()


def main():
    st.title("🔎 Mesin Pencari Sederhana - Berita Harga Pangan dan Inflasi")

    try:
        engine = load_engine()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.caption(f"Korpus: {engine.index.N} dokumen | vocabulary: {len(engine.index.postings)} term | "
               f"stemming Sastrawi | k1={config.K1}, b={config.B}")

    with st.sidebar:
        st.header("Pengaturan")
        method_label = st.selectbox("Metode retrieval", list(METHOD_LABELS))
        k = st.slider("Top-k", 1, 20, 5)
        explain = st.checkbox("Tampilkan penguraian skor per term", value=False)
        k1, b = config.K1, config.B
        if METHOD_LABELS[method_label] in ("bm25", "both"):
            k1 = st.slider("k1 (saturasi TF)", 0.0, 3.0, config.K1, 0.1)
            b = st.slider("b (normalisasi panjang)", 0.0, 1.0, config.B, 0.05)
        st.markdown("**Contoh query**")
        for ex in EXAMPLES:
            if st.button(ex, key=f"ex_{ex}"):
                st.session_state["query"] = ex
        st.caption("Sintaks Boolean: `beras AND (naik OR turun) AND NOT cabai`")

    query = st.text_input("Ketik query", key="query", placeholder="contoh: harga beras naik")
    go = st.button("Cari", type="primary") or bool(query)

    if go:
        engine.bm25 = BM25Retriever(engine.index, k1, b)          
        method = METHOD_LABELS[method_label]
        if method == "both":
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("TF-IDF + Cosine")
                render_results(engine, engine.search(query, "tfidf", k), k, explain)
            with c2:
                st.subheader("BM25")
                render_results(engine, engine.search(query, "bm25", k), k, explain)
        else:
            render_results(engine, engine.search(query, method, k), k, explain)


main()
