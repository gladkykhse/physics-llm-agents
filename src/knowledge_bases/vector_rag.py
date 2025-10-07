from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer
import numpy as np
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector
from src.utils.helpers import read_file


def chunk_text(tokenizer: PreTrainedTokenizer, text: str, chunk_size: int, drop_last: bool = False) -> list[str]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = enc["offset_mapping"]

    chunks = []
    n = len(offsets)
    for i in range(0, n, chunk_size):
        if drop_last and i + chunk_size > n:
            break
        start_char = offsets[i][0]
        end_char = offsets[min(i + chunk_size, n) - 1][1]
        chunks.append(text[start_char:end_char])
    return chunks


def embed_chunks(model: SentenceTransformer, chunks: list[str]) -> np.ndarray:
    return model.encode(sentences=chunks,
                        batch_size=32,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        normalize_embeddings=True)



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
    idx_tpl = read_file(create_index_path)

    with psycopg.connect(dsn) as con:
        with con.cursor() as cur:
            cur.execute(ext_sql)
            cur.execute(_render(tbl_tpl, sql.Identifier(table), sql.Literal(dim)))
            cur.execute(_render(idx_tpl, sql.Identifier(index), sql.Identifier(table), sql.Literal(lists)))
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
