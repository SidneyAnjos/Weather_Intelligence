"""
notebooks/ingest_weather_embeddings.py

Batch job: reads unembedded rows from weather_documents, chunks long
narrative_text, embeds each chunk with sentence-transformers, and writes
the vectors into weather_embeddings.

Mirrors notebooks/ingest_ticker_news_embeddings.py's job, but deliberately
does NOT use spark.write.jdbc -- plain psycopg2 + execute_values, per the
homework spec (Spark JDBC writes are unreliable against this Lakebase
instance).

Run:
    python notebooks/ingest_weather_embeddings.py
    python notebooks/ingest_weather_embeddings.py --batch-size 64 --dry-run
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timezone
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

from lakebase import get_connection, init_weather_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, matches
                                                          # the existing news
                                                          # pipeline so both
                                                          # tables stay
                                                          # queryable with the
                                                          # same <=> operator.

# Most NWS narrative text (alert description+instruction, or a single
# forecast period) is well under CHUNK_SIZE. Chunking mainly kicks in for
# long combined alert text (description + instruction can run long for
# multi-hazard alerts). Values match the existing news pipeline's
# convention for consistency.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Sliding-window character chunker. Simple and dependency-free; good
    enough for narrative text that's usually a handful of sentences.
    Splits on whitespace boundaries where possible to avoid cutting words.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # back off to the nearest preceding whitespace so we don't
            # split mid-word
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # guarantee forward progress

    return chunks


def fetch_unembedded_documents(conn, batch_size: int) -> List[dict]:
    """
    Documents in weather_documents that have no rows yet in
    weather_embeddings. Ordered by synced_at so re-runs process new data
    first if a previous run was interrupted mid-batch.
    """
    query = """
        SELECT d.id, d.narrative_text
        FROM weather_documents d
        LEFT JOIN weather_embeddings e ON e.document_id = d.id
        WHERE e.id IS NULL
        ORDER BY d.synced_at ASC
        LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, (batch_size,))
        return cur.fetchall()


def write_embeddings(conn, rows: List[Tuple], dry_run: bool = False) -> int:
    """
    rows: list of (document_id, chunk_index, chunk_text, embedding_list, model_name)
    Batched upsert via execute_values; embedding is passed as a Python
    list and cast to ::vector in SQL, which psycopg2 + the pgvector
    adapter handle without needing Spark's stringtype=unspecified trick.
    """
    if dry_run:
        logger.info("[dry-run] would write %d embedding rows", len(rows))
        return len(rows)

    insert_sql = """
        INSERT INTO weather_embeddings
            (document_id, chunk_index, chunk_text, embedding, model_name, created_at)
        VALUES %s
        ON CONFLICT (document_id, chunk_index) DO UPDATE
            SET chunk_text = EXCLUDED.chunk_text,
                embedding = EXCLUDED.embedding,
                model_name = EXCLUDED.model_name,
                created_at = EXCLUDED.created_at;
    """
    now = datetime.now(timezone.utc)
    values = [
        (doc_id, idx, text, embedding, model_name, now)
        for (doc_id, idx, text, embedding, model_name) in rows
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            insert_sql,
            values,
            template="(%s, %s, %s, %s::vector, %s, %s)",
            page_size=200,
        )
    return len(values)


def run(batch_size: int = 100, dry_run: bool = False, max_batches: int = 1000):
    logger.info("Loading embedding model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    init_weather_schema()

    total_docs = 0
    total_chunks = 0

    with get_connection() as conn:
        for batch_num in range(max_batches):
            docs = fetch_unembedded_documents(conn, batch_size)
            if not docs:
                break

            logger.info("Batch %d: embedding %d documents", batch_num, len(docs))

            # Build a flat list of (doc_id, chunk_index, chunk_text) across
            # the whole batch, then embed all chunks in one model.encode()
            # call for throughput.
            flat_chunks = []
            for doc in docs:
                pieces = chunk_text(doc["narrative_text"])
                for i, piece in enumerate(pieces):
                    flat_chunks.append((doc["id"], i, piece))

            if not flat_chunks:
                logger.info("Batch %d: no chunkable text, skipping", batch_num)
                continue

            texts = [c[2] for c in flat_chunks]
            embeddings = model.encode(texts, show_progress_bar=False,
                                       normalize_embeddings=True)

            rows = [
                (doc_id, idx, text, embedding.tolist(), MODEL_NAME)
                for (doc_id, idx, text), embedding in zip(flat_chunks, embeddings)
            ]

            written = write_embeddings(conn, rows, dry_run=dry_run)
            total_docs += len(docs)
            total_chunks += written

            logger.info("Batch %d: wrote %d chunk embeddings for %d documents",
                        batch_num, written, len(docs))

    logger.info("Done. %d documents processed, %d chunk embeddings written "
                "(dry_run=%s)", total_docs, total_chunks, dry_run)
    return {"documents_processed": total_docs, "chunks_written": total_chunks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed weather_documents into weather_embeddings")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Documents to fetch/embed per DB round trip")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute embeddings but don't write to the DB")
    args = parser.parse_args()

    run(batch_size=args.batch_size, dry_run=args.dry_run)
