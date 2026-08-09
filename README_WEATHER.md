# Weather Intelligence — Unstructured Data → Lakebase Vector Search → REST API

Harvests free-text weather content, embeds it, and exposes a semantic
search endpoint over it, mirroring the existing `ticker_news_documents` /
`ticker_news_embeddings` RAG pipeline in this app.

## Architecture

- **Databricks** runs the harvest + embed pipeline as a **scheduled Job**
  (`databricks.yml` + `notebooks/databricks_job_weather_pipeline.py`) —
  no manual triggering needed once deployed, it re-syncs on its own
  cadence (default: every 30 minutes). This is the stretch goal from the
  spec ("Add a scheduled Databricks Job... that re-syncs alerts every N
  minutes"), done as a first-class deliverable rather than an afterthought.
- **Lakebase Postgres + pgvector** is the database both the scheduled
  job and the Flask API read/write, per the spec.
- **Flask (`app.py`)** serves `/weather/sync` (manual/ad-hoc trigger, on
  top of the scheduled job) and `/weather/search`, both against the same
  Lakebase instance.

```
┌─────────────────────┐        ┌──────────────────────┐
│  Databricks Job      │  psycopg2  │   Lakebase Postgres    │
│  (scheduled, every    │───write──▶│   + pgvector           │
│  30 min via cron)     │           │   weather_documents /   │
│  sync + embed         │           │   weather_embeddings    │
└─────────────────────┘        └───────────▲──────────────┘
                                              │ psycopg2
                                    ┌─────────┴─────────┐
                                    │   Flask app.py      │
                                    │   /weather/search    │
                                    │   /weather/sync       │
                                    └─────────────────────┘
```

## Deliverables

| Original spec ask | This repo |
|---|---|
| `weather_client.py` — API client | `weather_client.py` (OpenWeatherMap) |
| Updated `app.py` with sync/search | `app.py` |
| `lakebase.py` (or new module) with DDL | `lakebase.py` |
| psycopg2-based embedding ingestion script | `notebooks/ingest_weather_embeddings.py` |
| README | `README_WEATHER.md` (this file) |
| *(new)* Scheduled re-sync | `notebooks/databricks_job_weather_pipeline.py` + `databricks.yml` (Databricks Job, cron-scheduled) |

## Data source: OpenWeatherMap One Call API 3.0

Chosen over the National Weather Service API and the CPC discussion
products because:

- **Global coverage** — not limited to the US, so `locations` can be any
  geocodable place name worldwide, not just US cities.
- **Genuinely unstructured narrative text** in two flavors that give the
  retrieval demo some contrast: `alerts` (government-issued alert
  `description` text, often several sentences of hazard-specific
  guidance) and `forecast` (the daily `summary` field, which
  OpenWeatherMap generates as an actual prose sentence — e.g. "There
  will be clear sky until morning, then partly cloudy" — rather than a
  short tag).
- **One call, two data types** — a single `/onecall` request returns
  both current alerts and the multi-day forecast, so `sync_locations()`
  only needs one HTTP round trip per location (plus one geocoding call
  the first time a place name is resolved).

Trade-offs versus NWS:
- **Requires an API key.** Free tier is generous, but this does mean the
  app now has a secret to provision (`OPENWEATHER_API_KEY`) and a key
  activation delay (new keys can take up to ~2 hours to start working).
- **Alerts depend on the location having an active government alerts
  feed OpenWeatherMap aggregates** — coverage/quality varies more by
  country than NWS's US-only but uniformly structured alerts.
- **Free-text geocoding is a live API call**, not a static gazetteer, so
  `resolve_location()` makes a network round trip for any location that
  isn't already given as `"lat,lon"`. Unresolvable place names are
  skipped with a logged warning rather than failing the whole sync.

## Schema

**`weather_documents`** — one row per alert or forecast period.

| column | notes |
|---|---|
| `id` | `alert:<hash of location+event+sender+start>` or `forecast:<hash of location+lat+lon+day timestamp>` — stable across re-syncs so upserts dedupe correctly |
| `location` | resolved place label from OpenWeatherMap's Geocoding API (`"City, State, Country"`), falling back to a raw `"lat,lon"` string if that's what the caller passed in |
| `source_type` | `alert` or `forecast` (checked via `CHECK` constraint) |
| `headline` | alert `event` name, or `"<weekday>: <weather.main>"` (e.g. `"Friday: Rain"`) |
| `narrative_text` | the actual free text to embed — alert `description` for alerts, daily `summary` (falling back to the short `weather[].description`) for forecasts |
| `issued_at` | alert `start` (unix ts → ISO), or forecast day `dt` (unix ts → ISO) |
| `payload` | raw JSON feature/period, kept for provenance/debugging |
| `synced_at` | last sync timestamp |

**`weather_embeddings`** — one row per chunk.

| column | notes |
|---|---|
| `document_id` | FK → `weather_documents.id`, `ON DELETE CASCADE` |
| `chunk_index` | 0-based position within the document |
| `chunk_text` | the actual chunk that was embedded |
| `embedding` | `vector(384)` |
| `model_name` | recorded per-row so the table can hold multiple model generations if you ever re-embed with something else |
| unique constraint | `(document_id, chunk_index)` — makes re-running ingestion an upsert, not a duplicate insert |

## Chunking

`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` (chars), matching the existing news
pipeline's convention. In practice, most daily forecast `summary` text
(one or two sentences) never hits this threshold and comes back as a
single chunk. The place chunking actually activates is longer
multi-hazard alert `description` text. The chunker is a plain
sliding-window splitter that backs off to the nearest whitespace
boundary so it doesn't cut mid-word; no extra dependency needed for
text this short.

## Embedding model

`sentence-transformers/all-MiniLM-L6-v2`, 384-dim — same model as the
existing news pipeline, specifically so `weather_embeddings` and
`ticker_news_embeddings` stay compatible/comparable if you ever want to
query across both with the same `<=>` operator and top_k logic.

## Running the pipeline end-to-end

**Local / manual run:**

```bash
# 1. One-time: install deps
pip install -r requirements.txt --break-system-packages

# 2. Get a free API key at https://openweathermap.org/api (One Call 3.0
#    requires signing up for that specific subscription, separately from
#    the general API key -- it's still free up to 1,000 calls/day).
#    New keys can take up to ~2 hours to activate.
export OPENWEATHER_API_KEY=...
export OPENWEATHER_UNITS=imperial   # or metric / standard

# 3. Set Lakebase connection env vars (same ones lakebase.py already reads)
export LAKEBASE_HOST=<your-lakebase-instance>.database.cloud.databricks.com
export LAKEBASE_PORT=5432
export LAKEBASE_DB=databricks_postgres
export LAKEBASE_USER=...
export LAKEBASE_PASSWORD=...
export LAKEBASE_SSLMODE=require

# 4. Start the app (creates weather_documents / weather_embeddings on
#    startup if they don't exist yet -- see lakebase.init_weather_schema())
python app.py

# 5. Harvest + normalize + upsert documents (manual/ad-hoc; the
#    scheduled Databricks Job does this automatically once deployed)
#    Locations can be free-text place names (geocoded automatically) or
#    "lat,lon" directly.
curl -X POST http://localhost:5000/weather/sync \
  -H "Content-Type: application/json" \
  -d '{"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}'

# 6. Embed the documents just synced
python notebooks/ingest_weather_embeddings.py

# 7. Search
curl -X POST http://localhost:5000/weather/search \
  -H "Content-Type: application/json" \
  -d '{"query": "flash flood risk this weekend", "top_k": 5}'

# GET variant (also supported):
curl "http://localhost:5000/weather/search?query=flash+flood+risk&top_k=5"
```

**Scheduled Databricks Job (harvest + embed, hands-off):**

```bash
# 1. One-time: create a Databricks secret scope and store your secrets
#    (never put real API keys or passwords directly in databricks.yml)
databricks secrets create-scope weather-pipeline
databricks secrets put-secret weather-pipeline openweather-api-key
databricks secrets put-secret weather-pipeline lakebase-db-password

# 2. Edit databricks.yml:
#    - set targets.dev.workspace.host / targets.prod.workspace.host
#      to your actual workspace URL
#    - set LAKEBASE_HOST and LAKEBASE_USER in
#      job_clusters[0].new_cluster.spark_env_vars to your actual
#      Lakebase instance details
#    - adjust the cron schedule / node_type_id for your cloud if needed

# 3. Validate, then deploy the bundle
databricks bundle validate
databricks bundle deploy -t dev

# 4. Trigger one run immediately to confirm it works, without waiting
#    for the schedule
databricks bundle run weather_sync_and_embed_job -t dev

# From here, it runs automatically every 30 minutes (or whatever cron
# you set) with no further action needed. Check runs and logs under
# Workflows in the Databricks UI, or `databricks bundle run --help`.
```

Re-running `/weather/sync`, the ingestion script, or the scheduled job
is always safe — all three upsert on the document/chunk key rather than
duplicating rows.

## Testing notes

This was developed in a sandboxed environment without egress to
`api.openweathermap.org`, `huggingface.co`, or a real Databricks
workspace, so validation here split into layers:

1. **`weather_client.py` parsing/normalization logic** — tested for real
   against mocked OpenWeatherMap Geocoding + One Call 3.0 responses
   (`tests/test_weather_client.py`, 12 passing tests covering
   lat/lon shortcuts, geocoding, missing-API-key handling, alert/forecast
   normalization including the `summary`→short-description fallback, and
   the full `sync_locations()` flow with a `limit`).
2. **DB schema, chunking, ingestion write path, `lakebase.upsert_documents()`,
   and the Flask `/weather/search` + `/weather/sync` endpoints** — run
   for real against a local Postgres 16 + pgvector instance standing in
   for Lakebase (same wire protocol, same DDL), using documents built
   through the actual `normalize_alert`/`normalize_forecast_day`
   functions. HNSW index confirmed created, 384-dim vectors confirmed
   stored, re-running ingestion confirmed idempotent, `/weather/sync`
   confirmed to return a clear 500 when `OPENWEATHER_API_KEY` is unset.
3. **`notebooks/databricks_job_weather_pipeline.py` — the actual script
   Databricks schedules** — run standalone end-to-end (harvest →
   `upsert_documents()` → embed) against the same local Postgres
   instance, confirming the combined job entry point works, not just
   its individual pieces.
4. **`databricks.yml`** — validated as well-formed YAML with the
   expected job name, cron schedule, task, and cluster env vars, but
   *not* deployed to a real workspace (no Databricks CLI credentials in
   this environment). Run `databricks bundle validate` yourself once you
   have a workspace to catch any schema issues specific to your account.

Across all of this, a stub embedder stood in only for the
`sentence-transformers` model download step. All SQL — the
`execute_values` batched upsert, the `::vector` cast, the `<=>` cosine
search, the `weather_documents` ⋈ `weather_embeddings` join — ran
against real Postgres.

Before submitting: run the Databricks bundle deploy/run commands above
against your actual workspace and Lakebase instance, with a real
`OPENWEATHER_API_KEY`, to confirm the live harvest, the scheduled
trigger, and real model embeddings all behave the same way.

## Known limitations / what I'd improve with more time

- **Every free-text location triggers a live geocoding call** — fine at
  homework scale, but for a larger location list it's worth caching
  resolved lat/lon (e.g. a small `location_cache` table) instead of
  re-geocoding the same city on every sync.
- **`source_type` filtering isn't exposed on `/weather/search`** —
  useful if you want "only alerts" vs "only forecasts" results;
  straightforward to add as an optional `source_type` filter in the
  `WHERE` clause.
- **No RAG summary endpoint yet** — the GET variant currently just
  re-dispatches to the same ranked results; the stretch goal of an
  LLM-generated natural-language summary over the top-k chunks isn't
  implemented.
- **HNSW index build isn't benchmarked** against a no-index baseline —
  worth doing once there's a realistic-sized `weather_embeddings` table
  (a handful of rows, as tested here, won't show a meaningful latency
  difference).
- **The scheduled job cluster spins up fresh each run** (`job_clusters`
  in `databricks.yml`), which means paying the ~38s `sentence-transformers`
  cold-import cost (torch + transformers) every single run. For a
  30-minute cadence that's a real, avoidable cost — worth switching to
  an existing/shared cluster (`existing_cluster_id`) or a serverless job
  cluster if your workspace supports it, once you've confirmed the job
  works on a dedicated cluster first.
- **No dead-letter handling for individual location failures** — if one
  location's geocoding or One Call fetch fails mid-run, it's logged and
  skipped, but there's no retry queue or alerting beyond the job-level
  `on_failure` email notification.
