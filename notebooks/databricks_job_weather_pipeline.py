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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LOCATIONS = [
    "Chicago, IL", "Austin, TX", "Miami, FL", "Phoenix, AZ",
    "Seattle, WA", "Denver, CO", "New York, NY", "Atlanta, GA",
]


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
    import weather_client
    from lakebase import init_weather_schema, upsert_documents
    import notebooks.ingest_weather_embeddings as embed_job

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

    locations = args.locations or _get_param("locations", ",".join(DEFAULT_LOCATIONS)).split(",")
    locations = [loc.strip() for loc in locations if loc.strip()]
    limit = args.limit or int(_get_param("limit", "100"))
    batch_size = args.batch_size or int(_get_param("batch_size", "100"))

    result = main(locations, limit, batch_size, args.dry_run)
    print(result)
