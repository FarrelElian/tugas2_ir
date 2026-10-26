"""Boolean retrieval (pelengkap).

Operator AND, OR, NOT (huruf kapital) dan tanda kurung. Prioritas: NOT > AND > OR.
Dua kata berurutan tanpa operator dianggap AND. Kata query diproses dengan preprocessing yang sama;
kata yang dibuang sebagai stopword diabaikan. Boolean TIDAK menghasilkan ranking.
"""
import re


class BooleanRetriever:
    def __init__(self, index, analyzer):
        self.index = index
        self.analyzer = analyzer
        self.universe = set(index.doc_ids)

    def _word_docs(self, word):
        toks = self.analyzer(word)
        if not toks:
            return None                       # dibuang stopword -> diabaikan
        return set.intersection(*[set(self.index.get(t)) for t in toks])

    @staticmethod
    def _and(a, b):
        if a is None: return b
        if b is None: return a
        return a & b

    @staticmethod
    def _or(a, b):
        if a is None: return b
        if b is None: return a
        return a | b

    def evaluate(self, query):
        """Kembalikan daftar doc_id (urutan index) yang memenuhi ekspresi Boolean."""
        toks = re.findall(r"\(|\)|[^\s()]+", query)
        pos = 0

        def peek():
            return toks[pos] if pos < len(toks) else None

        def parse_or():                       # or_expr  := and_expr (OR and_expr)*
            nonlocal pos
            left = parse_and()
            while peek() == "OR":
                pos += 1
                left = self._or(left, parse_and())
            return left

        def parse_and():                      # and_expr := not_expr ((AND)? not_expr)*
            nonlocal pos
            left = parse_not()
            while peek() is not None and peek() not in (")", "OR"):
                if peek() == "AND":
                    pos += 1
                left = self._and(left, parse_not())
            return left

        def parse_not():                      # not_expr := NOT not_expr | '(' or_expr ')' | KATA
            nonlocal pos
            tok = peek()
            if tok is None or tok in (")", "AND", "OR"):
                raise ValueError(f"Query Boolean tidak valid di dekat token ke-{pos}: {tok!r}")
            if tok == "NOT":
                pos += 1
                inner = parse_not()
                return None if inner is None else self.universe - inner
            if tok == "(":
                pos += 1
                inner = parse_or()
                if peek() != ")":
                    raise ValueError("Tanda kurung tutup ')' tidak ditemukan")
                pos += 1
                return inner
            pos += 1
            return self._word_docs(tok)

        if not toks:
            return []
        result = parse_or()
        if pos != len(toks):
            raise ValueError(f"Token tersisa tidak terparse: {toks[pos:]}")
        if result is None:
            return []
        return [d for d in self.index.doc_ids if d in result]


if __name__ == "__main__":
    # Demo modul: python -m src.boolean_retrieval
    from .preprocessing import load_and_clean, default_preprocessor, build_docs_tokens
    from .indexing import InvertedIndex
    df, _ = load_and_clean()
    pre = default_preprocessor()
    bl = BooleanRetriever(InvertedIndex(build_docs_tokens(df, pre)), pre)
    for q in ["beras AND harga", "cabai AND (naik OR turun)", "beras AND NOT cabai"]:
        r = bl.evaluate(q)
        print(f"{q:<30} -> {len(r):>3} dokumen | {r[:6]}")
