"""
app.py

Flask REST API. This file assumes it's merged into (or replaces) the
existing reference app's app.py -- it adds two routes:

    POST /weather/sync    -- harvest + normalize + upsert weather docs
    POST /weather/search  -- embed a query, cosine-similarity search

If your existing app.py already has /news/sync and /news/search, drop
these two route functions in alongside them and share the single
`embedding_model` loaded at module level (one model in memory, used by
both news and weather search) rather than loading it twice.
"""

import logging
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

from lakebase import get_connection, init_weather_schema, upsert_documents
import weather_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Loaded once at module/app level -- NOT per-request. Must be the same
# model used by the ingestion script (notebooks/ingest_weather_embeddings.py)
# so query vectors and stored vectors live in the same embedding space.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
logger.info("Loading embedding model %s ...", MODEL_NAME)
embedding_model = SentenceTransformer(MODEL_NAME)
logger.info("Embedding model ready.")

# Ensure tables exist on startup (idempotent).
init_weather_schema()


# ---------------------------------------------------------------------------
# POST /weather/sync
# ---------------------------------------------------------------------------
@app.route("/weather/sync", methods=["POST"])
def weather_sync():
    """
    Body: {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}

    Harvests active alerts + forecast periods for each location via
    weather_client, normalizes them, and upserts into weather_documents.
    Returns the count of documents synced.
    """
    body = request.get_json(silent=True) or {}
    locations = body.get("locations")
    limit = body.get("limit", 50)

    if not locations or not isinstance(locations, list):
        return jsonify({"error": "Body must include a non-empty 'locations' list"}), 400

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return jsonify({"error": "'limit' must be an integer"}), 400
    limit = max(1, min(limit, 500))  # sane bounds

    if not weather_client.API_KEY:
        return jsonify({
            "error": "OPENWEATHER_API_KEY is not set on the server. Get a free "
                     "key at https://openweathermap.org/api and set it before "
                     "calling /weather/sync.",
        }), 500

    documents = weather_client.sync_locations(locations, limit=limit)

    if not documents:
        return jsonify({
            "synced": 0,
            "message": "No documents harvested -- check that each location is "
                       "geocodable by OpenWeatherMap (try 'City, Country' or "
                       "pass 'lat,lon' directly) and that the One Call 3.0 "
                       "subscription on your API key is active.",
        }), 200

    upsert_documents(documents)

    return jsonify({
        "synced": len(documents),
        "locations_requested": locations,
        "by_source_type": {
            "alert": sum(1 for d in documents if d["source_type"] == "alert"),
            "forecast": sum(1 for d in documents if d["source_type"] == "forecast"),
        },
    }), 200


# ---------------------------------------------------------------------------
# POST /weather/search
# ---------------------------------------------------------------------------
@app.route("/weather/search", methods=["POST"])
def weather_search():
    """
    Body: {"query": "risk of flooding near rivers", "top_k": 5}

    Embeds the query with the same model used for ingestion, runs a
    cosine-similarity search over weather_embeddings via pgvector's <=>
    operator, and returns the top_k matches.
    """
    body = request.get_json(silent=True) or {}
    query = body.get("query")
    top_k = body.get("top_k", 5)

    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Body must include a non-empty 'query' string"}), 400

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return jsonify({"error": "'top_k' must be an integer"}), 400
    top_k = max(1, min(top_k, 20))  # clamp per spec

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM weather_embeddings;")
            count = cur.fetchone()["n"]

    if count == 0:
        return jsonify({
            "results": [],
            "message": "weather_embeddings is empty -- run POST /weather/sync "
                       "then notebooks/ingest_weather_embeddings.py first.",
        }), 200

    query_embedding = embedding_model.encode(
        query, normalize_embeddings=True
    ).tolist()

    search_sql = """
        SELECT d.id AS document_id, d.location, d.headline, d.narrative_text,
               e.chunk_text,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM weather_embeddings e
        JOIN weather_documents d ON d.id = e.document_id
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(search_sql, (query_embedding, query_embedding, top_k))
            rows = cur.fetchall()

    results = [
        {
            "location": r["location"],
            "headline": r["headline"],
            "chunk_text": r["chunk_text"],
            "similarity": round(float(r["similarity"]), 4),
        }
        for r in rows
    ]

    return jsonify({"query": query, "top_k": top_k, "results": results}), 200


# ---------------------------------------------------------------------------
# Optional stretch: GET /weather/search with an LLM summary of top results
# ---------------------------------------------------------------------------
@app.route("/weather/search", methods=["GET"])
def weather_search_get():
    """
    GET variant: /weather/search?query=...&top_k=5
    Delegates to the same logic as the POST handler by re-shaping the
    query args into a JSON body, then reusing weather_search().
    """
    query = request.args.get("query")
    top_k = request.args.get("top_k", 5)
    if not query:
        return jsonify({"error": "Missing 'query' query parameter"}), 400

    # Re-dispatch through the POST handler's logic without duplicating it
    with app.test_request_context(
        "/weather/search", method="POST",
        json={"query": query, "top_k": top_k},
    ):
        return weather_search()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
