import argparse
import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.knowledge_bases.vector_rag import (
    chunk_text_recursive,
    chunk_text_tokens,
    embed_chunks,
    insert_chunks,
    prepare_vector_rag,
)
from src.utils.helpers import load_yaml, read_file

load_dotenv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run insertion to vector RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="recursive",
        choices=["recursive", "overlap"],
        help="Chunking strategy",
    )
    args = parser.parse_args()

    cfg = load_yaml("config/vector_rag.yaml")

    model = SentenceTransformer(cfg["embedding_model"])
    tokenizer = model.tokenizer

    dsn = (
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@"
        f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    )

    prepare_vector_rag(
        dsn=dsn,
        table=cfg["table"],
        dim=model.get_sentence_embedding_dimension(),
        index=cfg["index"],
        lists=cfg["index_lists"],
    )

    text = "\n".join([read_file(file=text_file) for text_file in cfg["rag_text_files"]])

    if args.method == "recursive":
        chunks = chunk_text_recursive(
            tokenizer=tokenizer,
            text=text,
            max_chunk_tokens=cfg["max_chunk_tokens"],
            max_merge_tokens=cfg["max_merge_tokens"],
        )
    elif args.method == "overlap":
        chunks = chunk_text_tokens(
            tokenizer=tokenizer,
            text=text,
            max_chunk_tokens=cfg["max_chunk_tokens"],
            tokens_overlap=cfg["tokens_overlap"],
        )

    embeddings = embed_chunks(
        model=model,
        chunks=chunks,
    )

    insert_chunks(
        dsn=dsn,
        chunks=chunks,
        embeddings=embeddings,
        truncate_table=True,
        table=cfg["table"],
    )
