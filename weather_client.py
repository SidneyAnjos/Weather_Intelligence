"""
weather_client.py

Client for the OpenWeatherMap One Call API 3.0 (openweathermap.org).
Requires a free API key (set OPENWEATHER_API_KEY). Mirrors the shape of
massive_client.py: resolve locations, fetch raw records, normalize into
the shared document schema used by weather_documents.

Two narrative text sources map cleanly onto source_type:
  - "alert"    -> the `description` field of each active government alert
  - "forecast" -> the `summary` field of each daily forecast entry, which
                  OpenWeatherMap generates as an actual prose sentence
                  (e.g. "There will be clear sky until morning, then
                  partly cloudy"), unlike the short "clear sky" tag in
                  `weather[].description`.

Public entry point: sync_locations(locations, limit) -> List[dict]
"""

import base64
import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

GEO_URL = "https://api.openweathermap.org/geo/1.0/direct"
ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"
REQUEST_TIMEOUT = 15  # seconds

UNITS = os.environ.get("OPENWEATHER_UNITS", "imperial")  # imperial | metric | standard


def _decode_key_if_base64(raw: str) -> str:
    """
    OpenWeatherMap keys are 32 lowercase hex chars. The key stored in the
    weather-pipeline Databricks secret scope is base64-wrapped (44 chars),
    which the API rejects with 401 if sent as-is. If the raw value cleanly
    base64-decodes to a 32-char hex key, use the decoded form; otherwise
    return it unchanged so a legitimately-plain key is never mangled.
    """
    raw = (raw or "").strip()
    if len(raw) != 44:  # base64 of a 32-byte value is exactly 44 chars
        return raw
    try:
        decoded = base64.b64decode(raw, validate=True).decode("ascii")
    except Exception:
        return raw
    if len(decoded) == 32 and all(c in "0123456789abcdef" for c in decoded):
        return decoded
    return raw


API_KEY = _decode_key_if_base64(os.environ.get("OPENWEATHER_API_KEY", ""))


class OpenWeatherClientError(Exception):
    pass


def _require_api_key():
    if not API_KEY:
        raise OpenWeatherClientError(
            "OPENWEATHER_API_KEY is not set. Get a free key at "
            "https://openweathermap.org/api and export it before running sync."
        )


def _get(url: str, params: dict) -> dict:
    _require_api_key()
    params = {**params, "appid": API_KEY}
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 401:
        raise OpenWeatherClientError(
            "401 from OpenWeatherMap -- check OPENWEATHER_API_KEY is valid "
            "and (for new keys) has finished activating (~10 min-2 hr)."
        )
    if resp.status_code == 404:
        raise OpenWeatherClientError(f"404 from OpenWeatherMap: {url}")
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Location resolution
# ---------------------------------------------------------------------------

def resolve_location(location: str) -> Optional[Dict]:
    """
    Accepts either "lat,lon" or a free-text place name ("City, ST" /
    "City, Country"). Numeric "lat,lon" is resolved locally; anything
    else is geocoded via OpenWeatherMap's Geocoding API.
    """
    location = location.strip()

    if "," in location:
        parts = [p.strip() for p in location.split(",")]
        if len(parts) == 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
                return {"lat": lat, "lon": lon, "label": location}
            except ValueError:
                pass  # not numeric -- fall through to geocoding

    try:
        data = _get(GEO_URL, {"q": location, "limit": 1})
    except (OpenWeatherClientError, requests.RequestException) as e:
        logger.warning("Geocoding failed for %r: %s", location, e)
        return None

    if not data:
        logger.warning("Geocoding returned no results for %r", location)
        return None

    hit = data[0]
    label_parts = [hit.get("name")]
    if hit.get("state"):
        label_parts.append(hit["state"])
    if hit.get("country"):
        label_parts.append(hit["country"])

    return {
        "lat": hit["lat"],
        "lon": hit["lon"],
        "label": ", ".join(p for p in label_parts if p),
    }


def fetch_onecall(lat: float, lon: float) -> dict:
    """
    GET /data/3.0/onecall -> current + daily forecast (with narrative
    `summary`) + active government alerts. minutely/hourly are excluded
    since we don't need them for this pipeline.
    """
    return _get(ONECALL_URL, {
        "lat": lat,
        "lon": lon,
        "units": UNITS,
        "exclude": "minutely,hourly",
    })


# ---------------------------------------------------------------------------
# Normalization -> shared document schema
# ---------------------------------------------------------------------------

def _stable_hash(*parts: str) -> str:
    raw = "|".join(str(p) if p is not None else "" for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def normalize_alert(alert: dict, location_label: str) -> Optional[dict]:
    narrative = (alert.get("description") or "").strip()
    if not narrative:
        return None

    event = alert.get("event", "Weather Alert")
    start = alert.get("start")  # unix timestamp
    alert_id = _stable_hash(location_label, event, alert.get("sender_name"), start)

    issued_at = (
        datetime.fromtimestamp(start, tz=timezone.utc).isoformat()
        if start is not None else None
    )

    return {
        "id": f"alert:{alert_id}",
        "location": location_label,
        "source_type": "alert",
        "headline": event,
        "narrative_text": narrative,
        "issued_at": issued_at,
        "payload": alert,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


def normalize_forecast_day(day: dict, location_label: str, lat: float, lon: float) -> Optional[dict]:
    """
    OpenWeatherMap's daily `summary` is genuine prose -- e.g. "Expect a
    day of partly cloudy with rain" -- which is exactly the free-text we
    want to embed. Falls back to the short weather[].description if a
    plan/response doesn't include `summary`.
    """
    narrative = (day.get("summary") or "").strip()
    if not narrative and day.get("weather"):
        narrative = (day["weather"][0].get("description") or "").strip()
    if not narrative:
        return None

    dt = day.get("dt")  # unix timestamp, noon of the forecast day
    dedup_key = _stable_hash(location_label, f"{lat:.4f}", f"{lon:.4f}", dt)

    issued_at = (
        datetime.fromtimestamp(dt, tz=timezone.utc).isoformat()
        if dt is not None else None
    )

    headline = None
    if day.get("weather"):
        headline = day["weather"][0].get("main")
    if issued_at:
        day_label = datetime.fromtimestamp(dt, tz=timezone.utc).strftime("%A")
        headline = f"{day_label}: {headline}" if headline else day_label

    return {
        "id": f"forecast:{dedup_key}",
        "location": location_label,
        "source_type": "forecast",
        "headline": headline,
        "narrative_text": narrative,
        "issued_at": issued_at,
        "payload": day,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


def sync_locations(locations: List[str], limit: int = 50) -> List[dict]:
    """
    Given a list of "City, ST"/"City, Country" or "lat,lon" strings,
    resolves each, fetches alerts + daily forecast via One Call, and
    normalizes everything into document dicts (up to `limit` total).

    This is the function the /weather/sync endpoint calls.
    """
    documents: List[dict] = []

    for raw_location in locations:
        resolved = resolve_location(raw_location)
        if not resolved:
            continue
        lat, lon, location_label = resolved["lat"], resolved["lon"], resolved["label"]

        try:
            data = fetch_onecall(lat, lon)
        except (OpenWeatherClientError, requests.RequestException) as e:
            logger.warning("One Call fetch failed for %r: %s", raw_location, e)
            continue

        for alert in data.get("alerts", []):
            doc = normalize_alert(alert, location_label)
            if doc:
                documents.append(doc)

        for day in data.get("daily", []):
            doc = normalize_forecast_day(day, location_label, lat, lon)
            if doc:
                documents.append(doc)

        if len(documents) >= limit:
            break

    return documents[:limit]
