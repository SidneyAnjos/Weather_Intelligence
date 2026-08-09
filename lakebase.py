"""
lakebase.py

Connection helper + schema management for Lakebase (Databricks-managed
Postgres + pgvector), per the homework spec.

  - a single get_connection() context manager wrapping psycopg2
  - RealDictCursor so query results come back as dicts (JSON-friendly)
  - explicit DDL functions the app/scripts call once at startup
  - upsert_documents() shared by both the Flask /weather/sync endpoint
    and the Databricks scheduled job, so the upsert SQL lives in one place
"""

import os
import json
import logging
from contextlib import contextmanager
from typing import List

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection config -- Lakebase Postgres
# ---------------------------------------------------------------------------
# Same env vars a Lakebase-backed Flask app typically expects. Adjust names
# here if your reference app's lakebase.py uses different ones -- keep it
# consistent with whatever /news/sync already reads from.
LAKEBASE_HOST = os.environ.get("LAKEBASE_HOST", "localhost")
LAKEBASE_PORT = os.environ.get("LAKEBASE_PORT", "5432")
LAKEBASE_DB = os.environ.get("LAKEBASE_DB", "databricks_postgres")
LAKEBASE_USER = os.environ.get("LAKEBASE_USER", "postgres")
LAKEBASE_PASSWORD = os.environ.get("LAKEBASE_PASSWORD", "")
LAKEBASE_SSLMODE = os.environ.get("LAKEBASE_SSLMODE", "require")


@contextmanager
def get_connection():
    """
    Yields a psycopg2 connection configured with RealDictCursor as the
    default cursor factory, and commits on clean exit / rolls back on
    exception. Mirrors the existing news pipeline's connection helper so
    weather code can be dropped into the same app without a new pattern.
    """
    conn = psycopg2.connect(
        host=LAKEBASE_HOST,
        port=LAKEBASE_PORT,
        dbname=LAKEBASE_DB,
        user=LAKEBASE_USER,
        password=LAKEBASE_PASSWORD,
        sslmode=LAKEBASE_SSLMODE,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DDL -- weather_documents / weather_embeddings
# ---------------------------------------------------------------------------
# Mirrors the ticker_news_documents / ticker_news_embeddings pattern:
#   documents table holds the raw normalized record + provenance payload
#   embeddings table holds one row per chunk, FK'd back to the document
EMBEDDING_DIM = 384  # sentence-transformers/all-MiniLM-L6-v2

DDL_WEATHER_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS weather_documents (
    id              TEXT PRIMARY KEY,
    location        TEXT NOT NULL,
    source_type     TEXT NOT NULL CHECK (source_type IN ('alert', 'forecast')),
    headline        TEXT,
    narrative_text  TEXT NOT NULL,
    issued_at       TIMESTAMPTZ,
    payload         JSONB,
    synced_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

DDL_WEATHER_DOCUMENTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_weather_documents_location
    ON weather_documents (location);
"""

DDL_PGVECTOR_EXTENSION = """
CREATE EXTENSION IF NOT EXISTS vector;
"""

DDL_WEATHER_EMBEDDINGS = f"""
CREATE TABLE IF NOT EXISTS weather_embeddings (
    id              BIGSERIAL PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    embedding       vector({EMBEDDING_DIM}) NOT NULL,
    model_name      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);
"""

# HNSW is the modern pgvector default for cosine similarity search;
# ivfflat is the fallback for older pgvector builds (< 0.5.0). We try
# HNSW first and fall back automatically -- see init_weather_schema().
DDL_WEATHER_EMBEDDINGS_HNSW_INDEX = """
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_hnsw
    ON weather_embeddings
    USING hnsw (embedding vector_cosine_ops);
"""

DDL_WEATHER_EMBEDDINGS_IVFFLAT_INDEX = """
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_ivfflat
    ON weather_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""

DDL_WEATHER_EMBEDDINGS_DOC_FK_INDEX = """
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id
    ON weather_embeddings (document_id);
"""


def init_weather_schema():
    """
    Idempotently creates the weather_documents / weather_embeddings tables
    and their indexes. Safe to call on every app startup.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL_PGVECTOR_EXTENSION)
            cur.execute(DDL_WEATHER_DOCUMENTS)
            cur.execute(DDL_WEATHER_DOCUMENTS_INDEX)
            cur.execute(DDL_WEATHER_EMBEDDINGS)
            cur.execute(DDL_WEATHER_EMBEDDINGS_DOC_FK_INDEX)

            try:
                cur.execute(DDL_WEATHER_EMBEDDINGS_HNSW_INDEX)
                logger.info("weather_embeddings: created HNSW index")
            except psycopg2.Error as e:
                logger.warning("HNSW index unavailable (%s); falling back to ivfflat", e)
                conn.rollback()
                with conn.cursor() as cur2:
                    cur2.execute(DDL_WEATHER_EMBEDDINGS_IVFFLAT_INDEX)

    logger.info("Weather schema ready (weather_documents, weather_embeddings)")


# ---------------------------------------------------------------------------
# Shared upsert -- used by both the Flask /weather/sync endpoint and the
# Databricks scheduled job, so the SQL lives in exactly one place.
# ---------------------------------------------------------------------------

UPSERT_WEATHER_DOCUMENTS_SQL = """
    INSERT INTO weather_documents
        (id, location, source_type, headline, narrative_text, issued_at, payload, synced_at)
    VALUES %s
    ON CONFLICT (id) DO UPDATE
        SET headline = EXCLUDED.headline,
            narrative_text = EXCLUDED.narrative_text,
            issued_at = EXCLUDED.issued_at,
            payload = EXCLUDED.payload,
            synced_at = EXCLUDED.synced_at;
"""


def upsert_documents(documents: List[dict]) -> int:
    """
    Upserts a list of normalized document dicts (as produced by
    weather_client.sync_locations()) into weather_documents. Returns the
    count written. No-op (returns 0) if `documents` is empty.
    """
    if not documents:
        return 0

    from psycopg2.extras import execute_values

    rows = [
        (
            d["id"], d["location"], d["source_type"], d["headline"],
            d["narrative_text"], d["issued_at"], json.dumps(d["payload"]),
            d["synced_at"],
        )
        for d in documents
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur, UPSERT_WEATHER_DOCUMENTS_SQL, rows,
                template="(%s, %s, %s, %s, %s, %s, %s::jsonb, %s)",
                page_size=200,
            )

    return len(documents)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_weather_schema()
