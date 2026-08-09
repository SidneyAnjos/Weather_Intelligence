# Weather Intelligence — Unstructured Data → Lakebase Vector Search → REST API

Harvests free-text weather content, embeds it, and exposes a semantic
search endpoint over it, mirroring the existing `ticker_news_documents` /
`ticker_news_embeddings` RAG pipeline in this app.

## Architecture

- **Databricks** runs the harvest + embed pipeline as a **scheduled
  SERVERLESS Job** (`databricks.yml` + `notebooks/databricks_job_weather_pipeline.py`)
  — no manual triggering needed once deployed, it re-syncs on its own
  cadence (default: every 30 minutes). This is the stretch goal from the
  spec ("Add a scheduled Databricks Job... that re-syncs alerts every N
  minutes"), done as a first-class deliverable rather than an afterthought.
  The workspace only allows serverless compute, so the job has no
  `job_cluster`; its runtime dependencies live in the serverless
  `environments` spec (`base_environment: workspace-base-environments/databricks_ml`
  + pip deps), referenced from the task via `environment_key`.
- **Lakebase Postgres + pgvector** is the database both the scheduled
  job and the Flask API read/write, per the spec.
- **Flask (`app.py`)** serves `/weather/sync` (manual/ad-hoc trigger, on
  top of the scheduled job) and `/weather/search`, both against the same
  Lakebase instance.

```
┌─────────────────────┐        ┌──────────────────────┐
│  Databricks Job      │  psycopg2  │   Lakebase Postgres    │
│  (serverless, every  │───write──▶│   + pgvector           │
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
| *(new)* Scheduled re-sync | `notebooks/databricks_job_weather_pipeline.py` + `databricks.yml` (serverless Databricks Job, cron-scheduled) |

## Data source: NWS (National Weather Service) + OWM geocoding

Harvested data comes from the **National Weather Service** (`api.weather.gov`):

- **Forecast** — `detailedForecast` on each forecast period is genuine
  prose ("Showers and thunderstorms likely after noon. Partly sunny. High
  near 81..."), which is exactly the unstructured free-text the search
  is built on. 14 periods per city, refreshed regularly.
- **Alerts** — active government alerts (`description`, with
  `instruction` appended when present): multi-sentence hazard guidance.

Place-name locations ("Chicago") are resolved to lat/lon via the
**OpenWeatherMap Geocoding API** — a separate, free endpoint that does
NOT require the One Call plan or a card-verified subscription. NWS data
itself needs **no API key, no subscription, no credit card**.

Why NWS instead of OpenWeatherMap's prose endpoints? OpenWeatherMap
One Call 3.0 — the only endpoint that returns prose (`summary` +
government alert text) — now demands a card-verified subscription even
on its free tier, which wasn't an option here. NWS is also what the
assignment recommends.

Trade-offs of NWS:
- **US-only coverage.** NWS only covers US territories, so `locations`
  must be US places (the default list is 8 major US cities).
- **Geocoding format quirk:** OpenWeatherMap's geocoder returns **0 hits
  for `"City, ST"` two-letter state abbreviations** (`"Chicago, IL"` — the
  `IL` is parsed as a country code, Israel). Use plain city names
  (`"Chicago"`) or `"City, FullState"` (`"Chicago, Illinois"`). This is
  baked into the default location list and documented in `app.py`.
- **Alerts are only present when an area actually has an active
  alert** — on a quiet day you may harvest only forecasts. That's fine:
  forecasts alone carry the pipeline.
- **NWS requires a descriptive `User-Agent`** header; the bare default
  client string gets HTTP 403. `weather_client.py` sets one.

## Schema

**`weather_documents`** — one row per alert or forecast period.

| column | notes |
|---|---|
| `id` | `alert:<hash of location+event+sender+effective>` or `forecast:<hash of location+lat+lon+startTime>` — stable across re-syncs so upserts dedupe correctly |
| `location` | resolved place label from OpenWeatherMap's Geocoding API (`"City, State, Country"`), falling back to a raw `"lat,lon"` string if that's what the caller passed in |
| `source_type` | `alert` or `forecast` (checked via `CHECK` constraint) |
| `headline` | alert `event` name, or the forecast period `name` (e.g. `"Today"`, `"Tonight"`, `"Saturday"`) |
| `narrative_text` | the actual free text to embed — alert `description` (+ `instruction`) for alerts, `detailedForecast` (falling back to `shortForecast`) for forecasts |
| `issued_at` | NWS `effective` (alerts) or period `startTime` (forecasts), ISO 8601 with tz offset |
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

The HNSW index (`idx_weather_embeddings_hnsw`, `vector_cosine_ops`) is
created automatically by `init_weather_schema()`; `init_weather_schema()`
falls back to `ivfflat` on pgvector builds < 0.5.0 that lack HNSW.

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

## Lakebase connection: OAuth, not a static password

`lakebase.py` connects to Lakebase (Databricks-managed Postgres +
pgvector) in one of two modes, chosen at runtime in `get_connection()`:

1. **OAuth (default in this workspace).** Resolves the Lakebase
   endpoint's host and mints a short-lived scoped credential through the
   Databricks REST API, using the current identity's token:
   - endpoint: `LAKEBASE_ENDPOINT` (default
     `projects/support-app/branches/production/endpoints/primary`)
   - host: `GET /api/2.0/postgres/{endpoint}` → `status.hosts.host`
   - password: `POST /api/2.0/postgres/credentials` → `token`
   - user: `LAKEBASE_USER` if set, else the current Databricks user
   - psycopg2 connects with `sslmode=require` (no stored password —
     the credential is minted fresh each connection, 10-min TTL)
2. **Static password.** Set `LAKEBASE_PASSWORD` (and optionally
   `LAKEBASE_HOST`/`LAKEBASE_USER`) for local development against a
   plain Postgres or a course Lakebase that uses static credentials.

The same `WorkspaceClient()` path works locally (via `databricks` CLI
profile auth), in a Databricks App, and in a serverless job. Note the
in-job SDK has no `postgres` service, so the code uses the raw
`w.api_client.do("GET"/"POST", "/api/2.0/postgres/...")` REST calls
(with the `X-Databricks-Workspace-Id` header on the credential call).

### The API key was secretly base64-encoded

The `openweather-api-key` secret stored in the `weather-pipeline` scope
was a **base64-wrapped** 32-char OpenWeatherMap key (44 chars). Sent
as-is, OpenWeatherMap rejects it with HTTP 401. `weather_client.py`
now resolves the key through `_decode_key_if_base64()`: if the env value
is 44 chars and cleanly base64-decodes to a 32-char hex key, it uses the
decoded form; otherwise it leaves it alone. This one fix is what turned
the pipeline's 401s into working geocoding calls.

### psycopg2 gotcha on serverless

Use the SOURCE `psycopg2` wheel in the serverless job environment, NOT
`psycopg2-binary` — the binary wheel's bundled libpq aborts the
serverless kernel on import (SIGABRT). Local Windows dev can keep
`psycopg2-binary` (there's no system libpq to bind against); the split
is documented in `requirements.txt` / `databricks.yml`.

## Running the pipeline end-to-end

**Local / manual run:**

```bash
# 1. One-time: install deps
pip install -r requirements.txt --break-system-packages

# 2. Optional: set an OpenWeatherMap Geocoding key so place names like
#    "Chicago" can be resolved to lat/lon. NWS data itself needs no key.
#    If you pass "lat,lon" locations you can skip this entirely. A 32-hex
#    key; a base64-wrapped key is auto-decoded.
export OPENWEATHER_API_KEY=...

# 3. Lakebase connection: either set a static password for a course/plain PG ...
export LAKEBASE_HOST=<...>
export LAKEBASE_PASSWORD=...
# ... OR leave them unset to use the OAuth path (resolves the endpoint
#    via LAKEBASE_ENDPOINT + the current databricks CLI identity).
#    Optional: export LAKEBASE_ENDPOINT to point at a different instance.

# 4. Start the app (creates weather_documents / weather_embeddings on
#    startup if they don't exist yet -- see lakebase.init_weather_schema())
python app.py

# 5. Harvest + normalize + upsert documents (manual/ad-hoc; the
#    scheduled Databricks Job does this automatically once deployed)
#    Use plain city names or "City, FullState" -- "City, ST" returns
#    0 geocoding hits. Or pass "lat,lon" directly.
curl -X POST http://localhost:5000/weather/sync \
  -H "Content-Type: application/json" \
  -d '{"locations": ["Chicago", "Austin", "Miami"], "limit": 50}'

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
# 1. One-time: create a Databricks secret scope and store the API key
#    (never put the key directly in databricks.yml)
databricks secrets create-scope weather-pipeline
databricks secrets put-secret weather-pipeline openweather-api-key

# 2. Validate, then deploy the bundle (uses your -p <profile> identity)
databricks bundle validate -p <profile>
databricks bundle deploy -t dev -p <profile>

# 3. Trigger one run immediately to confirm it works, without waiting
#    for the schedule
databricks bundle run weather_sync_and_embed_job -t dev -p <profile>
```

A few things the serverless bundle handles that a cluster job didn't:
- **No `spark_env_vars` / `{{secrets/...}}` references.** Serverless
  `spark_python_task`s don't support them, so the job script itself
  fetches the key from the Databricks secrets API via the in-job
  `WorkspaceClient()` (`_ensure_openweather_key()`).
- **No `__file__`.** The serverless runner `exec()`s the script through
  an IPython kernel, so `__file__` is undefined. The script resolves its
  bundle root from the exec'd code filename (`sys._getframe().f_code.co_filename`)
  with `__file__`/CWD fallbacks, so sibling modules (`lakebase.py`,
  `weather_client.py`, the ingest script) import cleanly.

From here, it runs automatically every 30 minutes (or whatever cron
you set) with no further action needed. Check runs and logs under
Workflows in the Databricks UI, or `databricks bundle run --help`.

Re-running `/weather/sync`, the ingestion script, or the scheduled job
is always safe — all three upsert on the document/chunk key rather than
duplicating rows.

## Testing notes

Validated for real against the live stack (not just mocks):

1. **`weather_client.py`** — parsing/normalization tested against mocked
   NWS + OWM geocoding responses (`tests/test_weather_client.py`, 15
   tests, all green). Live validation confirmed the base64-key fix (raw
   44-char secret 401s, auto-decoded 32-hex key geocodes with HTTP 200)
   and that all 8 default locations resolve deterministically.
2. **`lakebase.py` OAuth connection** — connects to the real Lakebase as
   the current user (`pgvector 0.8.0`, host resolved from the endpoint).
   `init_weather_schema()` creates `weather_documents` /
   `weather_embeddings` with the HNSW index on the real instance.
3. **Serverless scheduled job** — deployed via the bundle and run for
   real: the serverless environment builds (`databricks_ml` base + source
   `psycopg2`/`sentence-transformers` deps), the in-job SDK fetches the
   geocoding key from secrets, the OAuth Lakebase connection resolves,
   the schema/HNSW index is created, and the `all-MiniLM-L6-v2` model
   downloads from Hugging Face. A run synced **100 documents** across 7
   US cities and wrote **109 chunk embeddings** to `weather_embeddings`.
4. **pgvector search on real data** — a nearest-neighbor query returns
   semantically sensible results (the "Today/Chicago" embedding matches
   itself at distance 0.0, then the similar "Today" forecasts for New
   York and Miami at ~0.11).

## Known limitations / what I'd improve with more time

- **NWS is US-only** — `locations` must be US places. For global
  coverage you'd need a paid data source (OpenWeatherMap One Call 3.0
  requires a card-verified subscription even on the free tier).
- **Geocoding format trap** — `"City, ST"` silently returns 0 hits.
  Callers must use plain city names or full state names; there's no
  graceful fallback yet.
- **Every free-text location triggers a live geocoding call** — fine at
  homework scale, but for a larger location list it's worth caching
  resolved lat/lon (e.g. a small `location_cache` table).
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
  (a handful of rows won't show a meaningful latency difference).
- **No dead-letter handling for individual location failures** — if one
  location's geocoding or One Call fetch fails mid-run, it's logged and
  skipped, but there's no retry queue or alerting beyond the job-level
  `on_failure` email notification.
