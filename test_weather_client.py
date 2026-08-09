"""
Tests for weather_client.py using mocked HTTP responses.

The sandbox this was developed in can't reach api.openweathermap.org
directly (egress is allowlisted to package registries only), so these
tests mock `requests.get` with realistic One Call 3.0 / Geocoding API
response shapes to validate the parsing/normalization logic. Run this
for real against the live API (with a real OPENWEATHER_API_KEY) before
submitting -- see README_WEATHER.md.
"""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weather_client as wc


GEOCODE_RESPONSE = [
    {"name": "Chicago", "lat": 41.8781, "lon": -87.6298, "country": "US", "state": "Illinois"}
]

ONECALL_RESPONSE = {
    "lat": 41.8781,
    "lon": -87.6298,
    "timezone": "America/Chicago",
    "alerts": [
        {
            "sender_name": "NWS Chicago",
            "event": "Flash Flood Warning",
            "start": 1754568000,
            "end": 1754575200,
            "description": "The National Weather Service has issued a Flash Flood "
                            "Warning. Turn around, don't drown. Move to higher ground.",
        }
    ],
    "daily": [
        {
            "dt": 1754568000,
            "summary": "Expect a day of partly cloudy with rain",
            "temp": {"day": 78.0, "min": 65.0, "max": 80.0},
            "weather": [{"main": "Rain", "description": "light rain"}],
        },
        {
            "dt": 1754654400,
            "summary": "There will be clear sky until morning, then partly cloudy",
            "temp": {"day": 81.0, "min": 66.0, "max": 83.0},
            "weather": [{"main": "Clear", "description": "clear sky"}],
        },
    ],
}


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = lambda: None
    resp.json.return_value = json_data
    return resp


def test_resolve_location_lat_lon_shortcut():
    coords = wc.resolve_location("41.88, -87.63")
    assert coords == {"lat": 41.88, "lon": -87.63, "label": "41.88, -87.63"}


@patch("weather_client.requests.get")
def test_resolve_location_geocodes_place_name(mock_get):
    mock_get.return_value = _mock_response(GEOCODE_RESPONSE)
    with patch.object(wc, "API_KEY", "fake-key"):
        resolved = wc.resolve_location("Chicago, IL")
    assert resolved["lat"] == 41.8781
    assert resolved["lon"] == -87.6298
    assert resolved["label"] == "Chicago, Illinois, US"


@patch("weather_client.requests.get")
def test_resolve_location_no_geocode_hit_returns_none(mock_get):
    mock_get.return_value = _mock_response([])
    with patch.object(wc, "API_KEY", "fake-key"):
        assert wc.resolve_location("Nowhereville") is None


def test_missing_api_key_raises():
    with patch.object(wc, "API_KEY", ""):
        try:
            wc._get(wc.GEO_URL, {"q": "Chicago"})
            assert False, "expected OpenWeatherClientError"
        except wc.OpenWeatherClientError as e:
            assert "OPENWEATHER_API_KEY" in str(e)


def test_normalize_alert_uses_description():
    doc = wc.normalize_alert(ONECALL_RESPONSE["alerts"][0], "Chicago, Illinois, US")
    assert doc["source_type"] == "alert"
    assert doc["headline"] == "Flash Flood Warning"
    assert "Turn around" in doc["narrative_text"]
    assert doc["id"].startswith("alert:")
    assert doc["issued_at"] is not None


def test_normalize_alert_empty_description_returns_none():
    assert wc.normalize_alert({"event": "Test", "description": ""}, "X") is None


def test_normalize_forecast_day_uses_summary():
    day = ONECALL_RESPONSE["daily"][1]
    doc = wc.normalize_forecast_day(day, "Chicago, Illinois, US", 41.8781, -87.6298)
    assert doc["source_type"] == "forecast"
    assert "clear sky until morning" in doc["narrative_text"]
    assert doc["headline"].endswith("Clear")
    assert doc["id"].startswith("forecast:")


def test_normalize_forecast_day_falls_back_to_short_description():
    day = {
        "dt": 1754568000,
        "summary": "",  # some plans/responses may omit it
        "weather": [{"main": "Rain", "description": "light rain"}],
    }
    doc = wc.normalize_forecast_day(day, "Chicago, Illinois, US", 41.8781, -87.6298)
    assert doc["narrative_text"] == "light rain"


def test_normalize_forecast_day_no_text_returns_none():
    assert wc.normalize_forecast_day({"dt": 123, "summary": "", "weather": []}, "X", 0, 0) is None


@patch("weather_client.requests.get")
def test_sync_locations_end_to_end(mock_get):
    def side_effect(url, params=None, timeout=None):
        if url == wc.GEO_URL:
            return _mock_response(GEOCODE_RESPONSE)
        elif url == wc.ONECALL_URL:
            return _mock_response(ONECALL_RESPONSE)
        raise AssertionError(f"unexpected URL {url}")

    mock_get.side_effect = side_effect

    with patch.object(wc, "API_KEY", "fake-key"):
        docs = wc.sync_locations(["Chicago, IL"], limit=50)

    # 1 alert + 2 daily forecasts = 3 documents
    assert len(docs) == 3
    source_types = {d["source_type"] for d in docs}
    assert source_types == {"alert", "forecast"}
    assert all(d["location"] == "Chicago, Illinois, US" for d in docs)


@patch("weather_client.requests.get")
def test_sync_locations_respects_limit(mock_get):
    def side_effect(url, params=None, timeout=None):
        if url == wc.GEO_URL:
            return _mock_response(GEOCODE_RESPONSE)
        elif url == wc.ONECALL_URL:
            return _mock_response(ONECALL_RESPONSE)
        raise AssertionError(f"unexpected URL {url}")

    mock_get.side_effect = side_effect

    with patch.object(wc, "API_KEY", "fake-key"):
        docs = wc.sync_locations(["Chicago, IL"], limit=2)
    assert len(docs) == 2


@patch("weather_client.requests.get")
def test_sync_locations_skips_unresolvable_location(mock_get):
    mock_get.return_value = _mock_response([])  # geocoding finds nothing
    with patch.object(wc, "API_KEY", "fake-key"):
        docs = wc.sync_locations(["Nowhereville"], limit=50)
    assert docs == []


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
