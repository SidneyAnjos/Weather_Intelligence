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
from typing import List, Optional, Tuple

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection config -- Lakebase Postgres
# ---------------------------------------------------------------------------
# Two connection modes, chosen at runtime in get_connection():
#
#   1. OAuth (default in this workspace) -- resolves the Lakebase endpoint's
#      connect host and mints a short-lived scoped credential through the
#      Databricks REST API, using the current identity's token. Works the
#      same from a Databricks App (the support_app pattern), a Databricks
#      serverless Job (this workspace only allows serverless compute), or a
#      local machine with databricks-cli auth. Requires LAKEBASE_ENDPOINT.
#
#   2. Static password -- set LAKEBASE_PASSWORD (and optionally LAKEBASE_HOST/
#      USER) for local development against a plain Postgres or a course
#      Lakebase that uses static credentials.
#
# psycopg2 gotcha: on Databricks serverless compute use the SOURCE `psycopg2`
# wheel, NOT `psycopg2-binary` -- the binary wheel's bundled libpq aborts the
# kernel on import. Local dev on Windows can keep psycopg2-binary (no system
# libpq there); see requirements.txt / databricks.yml for the split.
LAKEBASE_ENDPOINT = os.environ.get(
    "LAKEBASE_ENDPOINT",
    # The workspace Lakebase instance this homework targets (same one the
    # support_app uses). Override LAKEBASE_ENDPOINT for a different instance.
    "projects/support-app/branches/production/endpoints/primary",
)
LAKEBASE_HOST = os.environ.get("LAKEBASE_HOST", "localhost")
LAKEBASE_PORT = int(os.environ.get("LAKEBASE_PORT", "5432"))
LAKEBASE_DB = os.environ.get("LAKEBASE_DB", "databricks_postgres")
LAKEBASE_USER = os.environ.get("LAKEBASE_USER", "")
LAKEBASE_PASSWORD = os.environ.get("LAKEBASE_PASSWORD", "")
LAKEBASE_SSLMODE = os.environ.get("LAKEBASE_SSLMODE", "require")


def _workspace_client():
    """Lazily build a databricks-sdk WorkspaceClient (OAuth path only)."""
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def _endpoint_host(client) -> str:
    """Resolve the Lakebase endpoint's connect host via the postgres REST API."""
    ep = client.api_client.do("GET", f"/api/2.0/postgres/{LAKEBASE_ENDPOINT}")
    return ep["status"]["hosts"]["host"]


def _mint_db_password(client) -> str:
    """Mint a short-lived scoped credential to use as the DB password."""
    cred = client.api_client.do(
        "POST",
        "/api/2.0/postgres/credentials",
        headers={"X-Databricks-Workspace-Id": str(client.get_workspace_id())},
        body={"endpoint": LAKEBASE_ENDPOINT, "ttl": "600s"},
    )
    return cred["token"]


def _resolve_user(client) -> str:
    if LAKEBASE_USER:
        return LAKEBASE_USER
    return client.current_user.me().user_name


def _resolve_connection_params() -> Tuple[str, int, str, str, str]:
    """
    Returns (host, port, db, user, password) for psycopg2.connect().

    Prefers a static password when set (local dev / course instance); otherwise
    uses the OAuth path against the workspace Lakebase.
    """
    if LAKEBASE_PASSWORD:
        return (LAKEBASE_HOST, LAKEBASE_PORT, LAKEBASE_DB,
                LAKEBASE_USER or "postgres", LAKEBASE_PASSWORD)

    if LAKEBASE_ENDPOINT:
        client = _workspace_client()
        host = _endpoint_host(client)
        user = _resolve_user(client)
        password = _mint_db_password(client)
        logger.info(
            "Lakebase connection resolved via OAuth endpoint %s (host=%s, user=%s)",
            LAKEBASE_ENDPOINT, host, user,
        )
        return (host, LAKEBASE_PORT, LAKEBASE_DB, user, password)

    # Nothing configured: keep the historical localhost defaults so the module
    # stays importable and init_weather_schema() can run against a local PG.
    return (LAKEBASE_HOST, LAKEBASE_PORT, LAKEBASE_DB,
            LAKEBASE_USER or "postgres", LAKEBASE_PASSWORD)


@contextmanager
def get_connection():
    """
    Yields a psycopg2 connection configured with RealDictCursor as the
    default cursor factory, and commits on clean exit / rolls back on
    exception. Mirrors the existing news pipeline's connection helper so
    weather code can be dropped into the same app without a new pattern.
    """
    host, port, db, user, password = _resolve_connection_params()
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=db,
        user=user,
        password=password,
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
