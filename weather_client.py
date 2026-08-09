"""
weather_client.py

Harvests unstructured weather narrative text from the National Weather
Service (NWS) API -- free, no API key, no subscription, no credit card.
Free-text location names ("Chicago", "City, FullState") are resolved to
lat/lon via the OpenWeatherMap Geocoding API, which is a separate free
endpoint that does NOT require the paid/One-Call plan (or card
verification).

Why NWS instead of OpenWeatherMap One Call 3.0? One Call 3.0 -- the
endpoint that returns OpenWeatherMap's prose forecast `summary` and
government alert text -- now requires a (card-verified) subscription even
on the free tier. NWS returns the same two flavors of real narrative for
free:

  - "alert"    -> active government alert `description` (plus `instruction`
                  when present): multi-sentence hazard guidance
  - "forecast" -> each forecast period's `detailedForecast`, which is real
                  prose (e.g. "Showers and thunderstorms likely after noon.
                  Partly sunny. High near 81..."), unlike the short tags in
                  the free OWM endpoints.

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
NWS_BASE = "https://api.weather.gov"
# NWS requires a descriptive User-Agent; a bare client gets 403s.
NWS_USER_AGENT = "weather-intelligence-homework (md.258@outlook.com)"
REQUEST_TIMEOUT = 15  # seconds

API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")


class WeatherClientError(Exception):
    pass


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


def needs_geocoding(locations: List[str]) -> bool:
    """
    True if any location is a free-text place name rather than a numeric
    "lat,lon" pair. Only text locations need OPENWEATHER_API_KEY.
    """
    for location in locations:
        location = (location or "").strip()
        if "," in location:
            parts = [p.strip() for p in location.split(",")]
            if len(parts) == 2:
                try:
                    float(parts[0]), float(parts[1])
                    continue  # numeric lat,lon -- no geocoding needed
                except ValueError:
                    pass
        return True
    return False


def _require_api_key():
    if not API_KEY:
        raise WeatherClientError(
            "OPENWEATHER_API_KEY is not set. It's only needed to geocode "
            "place-name locations to lat/lon (NWS itself needs no key). "
            "Either set the key or pass 'lat,lon' locations directly."
        )


def _get(url: str, params: dict) -> dict:
    """OpenWeatherMap Geocoding GET (only used for place-name resolution)."""
    _require_api_key()
    params = {**params, "appid": API_KEY}
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 401:
        raise WeatherClientError(
            "401 from OpenWeatherMap -- check OPENWEATHER_API_KEY is valid "
            "and (for new keys) has finished activating (~10 min-2 hr)."
        )
    if resp.status_code == 404:
        raise WeatherClientError(f"404 from OpenWeatherMap: {url}")
    resp.raise_for_status()
    return resp.json()


def _nws_get(url: str) -> dict:
    """NWS GET -- the data source itself needs no auth, just a User-Agent."""
    resp = requests.get(
        url,
        headers={"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"},
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code == 403:
        raise WeatherClientError(
            "403 from NWS -- the API rejects requests without a descriptive "
            "User-Agent (see NWS_USER_AGENT)."
        )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Location resolution
# ---------------------------------------------------------------------------

def resolve_location(location: str) -> Optional[Dict]:
    """
    Accepts either "lat,lon" or a free-text place name ("City", "City, ST" /
    "City, Country"). Numeric "lat,lon" is resolved locally; anything else
    is geocoded via OpenWeatherMap's (free, key-only) Geocoding API.
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
    except (WeatherClientError, requests.RequestException) as e:
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


# ---------------------------------------------------------------------------
# NWS harvest
# ---------------------------------------------------------------------------

def fetch_nws_forecast(lat: float, lon: float) -> List[dict]:
    """
    Resolve the lat/lon to its NWS gridpoint, then fetch the multi-day
    forecast. Returns the list of period dicts -- each carries
    `detailedForecast`, the prose narrative we embed.
    """
    lat4, lon4 = round(lat, 4), round(lon, 4)
    points = _nws_get(f"{NWS_BASE}/points/{lat4},{lon4}")
    forecast_url = points["properties"]["forecast"]
    data = _nws_get(forecast_url)
    return data.get("properties", {}).get("periods", [])


def fetch_nws_alerts(lat: float, lon: float) -> List[dict]:
    """
    Active government alerts within range of the lat/lon point. Returns the
    list of alert `properties` dicts (event, description, instruction, ...).
    May be empty -- that just means no active alerts for the area.
    """
    lat4, lon4 = round(lat, 4), round(lon, 4)
    data = _nws_get(f"{NWS_BASE}/alerts/active?point={lat4},{lon4}")
    return [f["properties"] for f in data.get("features", [])]


# ---------------------------------------------------------------------------
# Normalization -> shared document schema
# ---------------------------------------------------------------------------

def _stable_hash(*parts: str) -> str:
    raw = "|".join(str(p) if p is not None else "" for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def normalize_alert(alert: dict, location_label: str) -> Optional[dict]:
    """
    NWS alert `properties` -> document dict. The narrative is the alert
    `description` (multi-sentence hazard text), with `instruction`
    appended when present. Id is stable across re-syncs so upserts dedupe.
    """
    event = (alert.get("event") or "Weather Alert").strip()
    description = (alert.get("description") or "").strip()
    instruction = (alert.get("instruction") or "").strip()

    narrative = description
    if instruction and instruction != description:
        narrative = f"{description}\n\n{instruction}".strip()
    if not narrative:
        return None

    alert_id = _stable_hash(
        location_label, event, alert.get("senderName"), alert.get("effective")
    )

    return {
        "id": f"alert:{alert_id}",
        "location": location_label,
        "source_type": "alert",
        "headline": event,
        "narrative_text": narrative,
        "issued_at": alert.get("effective"),
        "payload": alert,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


def normalize_forecast_period(period: dict, location_label: str,
                              lat: float, lon: float) -> Optional[dict]:
    """
    NWS forecast period -> document dict. The narrative is the
    `detailedForecast` prose ("Showers and thunderstorms likely after
    noon..."), falling back to the short `shortForecast` tag. Id is stable
    across re-syncs (location + period startTime).
    """
    narrative = (period.get("detailedForecast") or "").strip()
    if not narrative:
        narrative = (period.get("shortForecast") or "").strip()
    if not narrative:
        return None

    start = period.get("startTime")
    dedup_key = _stable_hash(location_label, f"{lat:.4f}", f"{lon:.4f}", start)

    return {
        "id": f"forecast:{dedup_key}",
        "location": location_label,
        "source_type": "forecast",
        "headline": period.get("name"),
        "narrative_text": narrative,
        "issued_at": start,
        "payload": period,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


def sync_locations(locations: List[str], limit: int = 50) -> List[dict]:
    """
    Given a list of "City" / "City, ST" / "City, Country" or "lat,lon"
    strings, resolves each, fetches NWS forecast periods + active alerts,
    and normalizes everything into document dicts (up to `limit` total).

    This is the function the /weather/sync endpoint and the scheduled job
    call.
    """
    documents: List[dict] = []

    for raw_location in locations:
        resolved = resolve_location(raw_location)
        if not resolved:
            continue
        lat, lon, location_label = resolved["lat"], resolved["lon"], resolved["label"]

        try:
            periods = fetch_nws_forecast(lat, lon)
            alerts = fetch_nws_alerts(lat, lon)
        except (WeatherClientError, requests.RequestException) as e:
            logger.warning("NWS fetch failed for %r: %s", raw_location, e)
            continue

        for alert in alerts:
            doc = normalize_alert(alert, location_label)
            if doc:
                documents.append(doc)

        for period in periods:
            doc = normalize_forecast_period(period, location_label, lat, lon)
            if doc:
                documents.append(doc)

        if len(documents) >= limit:
            break

    return documents[:limit]
