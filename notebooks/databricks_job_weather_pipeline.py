"""
notebooks/databricks_job_weather_pipeline.py

Entry point for the scheduled Databricks Job. Runs the full
harvest -> upsert -> embed cycle in one process:

    1. weather_client.sync_locations(...)   -- hit OpenWeatherMap
    2. db.upsert_documents(...)             -- write weather_documents
    3. ingest_weather_embeddings.run(...)   -- embed + write weather_embeddings

This is deliberately a plain Python script (no pyspark, no
spark.write.jdbc) so it runs the same way locally, in a notebook cell,
or as a Databricks Job "Python script" task -- see databricks.yml for
the job/schedule definition.

Locations and other parameters come from either CLI args (for local
runs) or Databricks Job parameters, read via widgets when running
inside a notebook context -- see `_get_param()` below.

Usage (local):
    python notebooks/databricks_job_weather_pipeline.py \
        --locations "Chicago, IL" "Austin, TX" "Miami, FL" \
        --limit 50 --batch-size 100
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _repo_root() -> str:
    """
    Resolve the bundle source dir (the one containing lakebase.py /
    weather_client.py) so the sibling modules import cleanly.

    Three anchors, tried in order:
      1. __file__ -- set when run directly (python notebooks/<script>.py).
      2. sys._getframe().f_code.co_filename -- Databricks serverless does
         NOT define __file__; it exec()s the file through an IPython
         kernel with the real uploaded path as the code filename, so the
         frame's co_filename still points at
         .../.bundle/<name>/<target>/files/notebooks/<script>.py.
      3. CWD ancestry -- covers kernels that chdir into the script dir.

    Returns "" if nothing resolves (imports then fail with a clear
    traceback instead of a cryptic one).
    """
    candidates = []
    here = globals().get("__file__")
    if here:
        candidates.append(os.path.dirname(os.path.dirname(os.path.abspath(here))))
    try:
        exec_file = sys._getframe().f_code.co_filename
        if exec_file and exec_file not in ("<stdin>", "<string>", ""):
            candidates.append(os.path.dirname(os.path.dirname(os.path.abspath(exec_file))))
            candidates.append(os.path.dirname(os.path.abspath(exec_file)))
    except Exception:
        pass
    d = os.getcwd()
    for _ in range(6):  # cwd + up to 5 ancestors
        candidates.append(d)
        d = os.path.dirname(d)
    for root in candidates:
        root = os.path.abspath(root)
        if os.path.exists(os.path.join(root, "lakebase.py")):
            return root
    return ""


REPO_ROOT = _repo_root()
logger.info("PATH_BOOTSTRAP repo_root=%r cwd=%r argv0=%r",
            REPO_ROOT, os.getcwd(), sys.argv[0])
if REPO_ROOT:
    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, os.path.join(REPO_ROOT, "notebooks"))

# Plain city names (or "City, FullState") -- OpenWeatherMap's geocoder
# returns 0 hits for "City, ST" two-letter abbreviations ("IL" is parsed
# as the country code for Israel).
DEFAULT_LOCATIONS = [
    "Chicago", "Austin", "Miami", "Phoenix",
    "Seattle", "Denver", "New York", "Atlanta",
]

# Where the OpenWeatherMap key lives when running as a scheduled job.
# Serverless tasks have no spark_env_vars, so the job fetches the secret
# itself via the workspace API (see _ensure_openweather_key()).
WEATHER_SECRET_SCOPE = os.environ.get("WEATHER_SECRET_SCOPE", "weather-pipeline")
WEATHER_SECRET_KEY = os.environ.get("WEATHER_SECRET_KEY", "openweather-api-key")


def _ensure_openweather_key() -> None:
    """
    Ensure OPENWEATHER_API_KEY is in the environment before weather_client
    is imported (it reads the key at import time).

    Locally you just export OPENWEATHER_API_KEY. In the scheduled job there
    is no spark_env_vars, so we pull the key from the Databricks secret
    scope via the workspace API -- the same in-job WorkspaceClient() that
    resolves the Lakebase OAuth credential, verified to work on serverless.
    """
    if os.environ.get("OPENWEATHER_API_KEY"):
        return
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        path = ("/api/2.0/secrets/get?scope=%s&key=%s"
                % (WEATHER_SECRET_SCOPE, WEATHER_SECRET_KEY))
        secret = w.api_client.do("GET", path)
        os.environ["OPENWEATHER_API_KEY"] = secret.get("value") or ""
        if os.environ["OPENWEATHER_API_KEY"]:
            logger.info("Loaded OPENWEATHER_API_KEY from secrets/%s/%s",
                        WEATHER_SECRET_SCOPE, WEATHER_SECRET_KEY)
    except Exception as e:
        logger.warning("Could not load OPENWEATHER_API_KEY from Databricks "
                       "secrets (%s/%s): %s", WEATHER_SECRET_SCOPE, WEATHER_SECRET_KEY, e)


def _get_param(name: str, default: str) -> str:
    """
    Reads a parameter from a Databricks notebook widget if running inside
    Databricks (dbutils is injected into notebook globals there), falling
    back to an environment variable, then the given default. Lets the
    same script work as a plain CLI tool locally and as a scheduled
    Databricks Job task with Job parameters wired to widgets.
    """
    try:
        # dbutils is only defined when this runs inside a Databricks
        # notebook/job context -- not importable, so we check globals().
        dbutils = globals().get("dbutils")
        if dbutils is not None:
            return dbutils.widgets.get(name)
    except Exception:
        pass
    return os.environ.get(name.upper(), default)


def main(locations, limit, batch_size, dry_run):
    # Set the API key BEFORE weather_client is imported (it reads the env
    # var at module import time).
    _ensure_openweather_key()

    import weather_client
    from lakebase import init_weather_schema, upsert_documents
    try:
        import notebooks.ingest_weather_embeddings as embed_job
    except ImportError:
        # Same file, but when the notebooks dir isn't importable as a
        # package (exec'd on serverless), import it as a plain sibling.
        import ingest_weather_embeddings as embed_job

    logger.info("=== Weather pipeline job starting ===")
    logger.info("Locations: %s", locations)

    init_weather_schema()

    logger.info("Step 1/2: harvesting from OpenWeatherMap ...")
    documents = weather_client.sync_locations(locations, limit=limit)
    logger.info("Harvested %d documents (%d alerts, %d forecasts)",
                len(documents),
                sum(1 for d in documents if d["source_type"] == "alert"),
                sum(1 for d in documents if d["source_type"] == "forecast"))

    written = upsert_documents(documents) if not dry_run else len(documents)
    logger.info("Upserted %d documents into weather_documents (dry_run=%s)", written, dry_run)

    logger.info("Step 2/2: embedding unembedded documents ...")
    embed_result = embed_job.run(batch_size=batch_size, dry_run=dry_run)

    logger.info("=== Weather pipeline job complete ===")
    logger.info("Summary: %d documents synced, %d chunks embedded",
                written, embed_result["chunks_written"])
    return {"documents_synced": written, **embed_result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest + embed weather data (Databricks Job entry point)")
    parser.add_argument("--locations", nargs="+", default=None,
                        help="Space-separated 'City, ST' or 'lat,lon' locations")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max documents to harvest across all locations")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Documents per embedding batch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # The bundle passes locations as one comma-separated string, but a local
    # CLI run passes several "--locations City, ST" args -- normalize both.
    raw_locations = args.locations or _get_param("locations", ",".join(DEFAULT_LOCATIONS))
    if isinstance(raw_locations, list):
        raw_locations = ",".join(raw_locations)
    locations = [loc.strip() for loc in raw_locations.split(",") if loc.strip()]
    limit = args.limit or int(_get_param("limit", "100"))
    batch_size = args.batch_size or int(_get_param("batch_size", "100"))

    result = main(locations, limit, batch_size, args.dry_run)
    print(result)
