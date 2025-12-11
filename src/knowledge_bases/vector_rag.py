import re
from typing import List

import nltk
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer

from src.utils.helpers import read_file

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize


def tok_len(tokenizer: PreTrainedTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


def split_by_heading_level(text: str, level: int) -> List[str]:
    pattern = re.compile(rf"(?m)^({'#' * level})\s+.*$")
    matches = list(pattern.finditer(text))
    if not matches:
        return [text] if text != "" else []

    parts: List[str] = []

    if matches[0].start() > 0:
        preface = text[: matches[0].start()]
        if preface != "":
            parts.append(preface)

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end]
        if chunk != "":
            parts.append(chunk)

    return parts


def _pack_by_newlines(text: str, tokenizer: PreTrainedTokenizer, max_token_len: int) -> List[str]:
    tokens = re.findall(r"[^\n]+|\n+", text)

    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf != "":
            chunks.append(buf)
            buf = ""

    for tk in tokens:
        if tok_len(tokenizer, buf + tk) <= max_token_len:
            buf += tk
            continue

        flush()

        if tok_len(tokenizer, tk) <= max_token_len:
            buf = tk
            continue

        if tk.startswith("\n"):
            nl_run = tk
            while nl_run:
                piece = nl_run[:1]
                if tok_len(tokenizer, piece) <= max_token_len:
                    chunks.append(piece)
                    nl_run = nl_run[1:]
                else:
                    chunks.append("")
                    nl_run = nl_run[1:]
            buf = ""
        else:
            pieces = re.findall(r"\S+|\s+", tk)
            seg = ""
            for p in pieces:
                if tok_len(tokenizer, seg + p) <= max_token_len:
                    seg += p
                else:
                    if seg != "":
                        chunks.append(seg)
                    if tok_len(tokenizer, p) <= max_token_len:
                        seg = p
                    else:
                        tmp = ""
                        for ch in p:
                            if tok_len(tokenizer, tmp + ch) <= max_token_len:
                                tmp += ch
                            else:
                                if tmp != "":
                                    chunks.append(tmp)
                                tmp = ch
                        seg = tmp
            if seg != "":
                buf = seg
            else:
                buf = ""

    flush()
    return [c for c in chunks if c != ""]


def _pack_sentences_with_newline_fallback(text: str, tokenizer: PreTrainedTokenizer, max_token_len: int) -> List[str]:
    sentences = [s for s in sent_tokenize(text) if s.strip()]
    chunks: List[str] = []
    current = ""

    for s in sentences:
        s_ok = tok_len(tokenizer, s) <= max_token_len

        if current == "":
            if s_ok:
                current = s
            else:
                chunks.extend(_pack_by_newlines(s, tokenizer, max_token_len))
        else:
            candidate = current + s
            if s_ok and tok_len(tokenizer, candidate) <= max_token_len:
                current = candidate
            else:
                chunks.append(current)
                if s_ok:
                    current = s
                else:
                    chunks.extend(_pack_by_newlines(s, tokenizer, max_token_len))
                    current = ""

    if current:
        chunks.append(current)
    return chunks


def recurse(text: str, tokenizer: PreTrainedTokenizer, level: int, max_token_len: int) -> List[str]:
    if text == "":
        return []

    if tok_len(tokenizer, text) <= max_token_len:
        return [text]

    if level <= 6:
        chunks: List[str] = []
        for section in split_by_heading_level(text=text, level=level):
            if tok_len(tokenizer, section) > max_token_len:
                chunks.extend(recurse(section, tokenizer, level=level + 1, max_token_len=max_token_len))
            else:
                chunks.append(section)
        return chunks

    return _pack_by_newlines(text, tokenizer, max_token_len)


def merge_small_chunks(chunks: List[str], tokenizer: PreTrainedTokenizer, max_token_len: int) -> List[str]:
    if not chunks:
        return []

    merged: List[str] = []
    current = chunks[0]

    for nxt in chunks[1:]:
        candidate = current + nxt  # exact boundary (no added whitespace)
        if tok_len(tokenizer, candidate) <= max_token_len:
            current = candidate
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


def chunk_text_recursive(
    text: str, tokenizer: PreTrainedTokenizer, max_chunk_tokens: int, max_merge_tokens: int
) -> List[str]:
    chunks = recurse(text, tokenizer, level=1, max_token_len=max_chunk_tokens)
    return merge_small_chunks(chunks, tokenizer, max_token_len=max_merge_tokens)


def chunk_text_tokens(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_chunk_tokens: int,
    tokens_overlap: int,
) -> List[str]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = encoded["input_ids"]

    chunks: List[str] = []
    stride = max_chunk_tokens - tokens_overlap
    start = 0

    while start < len(input_ids):
        end = start + max_chunk_tokens
        chunk_ids = input_ids[start:end]
        if not chunk_ids:
            break

        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(input_ids):
            break

        start += stride

    return chunks


def embed_chunks(model: SentenceTransformer, chunks: list[str]) -> np.ndarray:
    return model.encode(
        sentences=chunks, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    )


def _render(template: str, *args) -> sql.Composed:
    return sql.SQL(template).format(*args)


def prepare_vector_rag(
    dsn: str,
    table: str = "rag_chunks",
    dim: int = 768,
    index: str = "rag_chunks_embedding_idx",
    lists: int = 100,
    extensions_path: str = "sql/postgres/extensions.sql",
    create_table_path: str = "sql/postgres/create_table.sql",
    create_index_path: str = "sql/postgres/create_index.sql",
) -> None:
    ext_sql = read_file(extensions_path)
    tbl_tpl = read_file(create_table_path)
    # idx_tpl = read_file(create_index_path)

    with psycopg.connect(dsn) as con:
        with con.cursor() as cur:
            cur.execute(ext_sql)
            cur.execute(_render(tbl_tpl, sql.Identifier(table), sql.Literal(dim)))
            # cur.execute(_render(idx_tpl, sql.Identifier(index), sql.Identifier(table), sql.Literal(lists)))
        con.commit()

        register_vector(con)


def insert_chunks(
    dsn: str,
    chunks: list[str],
    embeddings: np.ndarray,
    source: str = "openstax",
    truncate_table: bool = False,
    table: str = "rag_chunks",
    truncate_sql_path: str = "sql/postgres/truncate_table.sql",
    insert_sql_path: str = "sql/postgres/insert_table.sql",
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")

    trunc_tpl = read_file(truncate_sql_path)
    insert_tpl = read_file(insert_sql_path)

    trunc_stmt = _render(trunc_tpl, sql.Identifier(table))
    insert_stmt = _render(insert_tpl, sql.Identifier(table))

    rows = [(source, i, chunks[i], embeddings[i].tolist()) for i in range(len(chunks))]

    with psycopg.connect(dsn) as con:
        register_vector(con)
        with con.cursor() as cur:
            if truncate_table:
                cur.execute(trunc_stmt)

            cur.executemany(insert_stmt, rows)

        con.commit()


class PgVectorRetriever:
    def __init__(
        self,
        dsn: str,
        table: str = "rag_chunks",
        model: str = "sentence-transformers/all-mpnet-base-v2",
        memory: bool = True,
    ):
        self.dsn = dsn
        self.table = table
        self.model = SentenceTransformer(model)
        self.memory = memory
        self.retrieved_chunk_ids = set()

    def clear_memory(self):
        self.retrieved_chunk_ids = set()

    def __call__(self, query: str, top_k: int = 3) -> list[dict[str, object]]:
        qvec = self.model.encode(query, normalize_embeddings=True)
        with psycopg.connect(self.dsn) as con:
            register_vector(con)
            with con.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT source, chunk_id, text, (1 - (embedding <=> %s)) AS score
                    FROM {self.table}
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (qvec, qvec, top_k),
                )
                rows = cur.fetchall()

        rows_parsed = [{"source": s, "chunk_id": cid, "text": t, "score": float(sc)} for (s, cid, t, sc) in rows]

        if self.memory:
            rows_parsed = [r for r in rows_parsed if r["chunk_id"] not in self.retrieved_chunk_ids]
            self.retrieved_chunk_ids.update({record["chunk_id"] for record in rows_parsed})

        return rows_parsed
