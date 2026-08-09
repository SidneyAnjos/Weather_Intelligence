"""
Tests for weather_client.py using mocked HTTP responses.

weather_client harvests NWS prose (forecast `detailedForecast` + alert
`description`/`instruction`) and geocodes place names via the
OpenWeatherMap Geocoding API. These tests mock `requests.get` with
realistic NWS / Geocoding response shapes to validate the
parsing/normalization logic without touching the network. Run the live
smoke check (see README_WEATHER.md) against api.weather.gov before
submitting.
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weather_client as wc


GEOCODE_RESPONSE = [
    {"name": "Chicago", "lat": 41.8781, "lon": -87.6298, "country": "US", "state": "Illinois"}
]

POINTS_RESPONSE = {
    "properties": {"forecast": "https://api.weather.gov/gridpoints/LOT/76,73/forecast"}
}

FORECAST_RESPONSE = {
    "properties": {
        "periods": [
            {
                "name": "Today",
                "startTime": "2026-08-09T12:00:00-05:00",
                "detailedForecast": "Showers and thunderstorms likely after noon. "
                                    "Partly sunny. High near 81.",
                "shortForecast": "Showers And Thunderstorms Likely",
            },
            {
                "name": "Tonight",
                "startTime": "2026-08-09T18:00:00-05:00",
                "detailedForecast": "Showers and thunderstorms likely. Mostly cloudy, "
                                    "with a low around 72.",
                "shortForecast": "Showers And Thunderstorms Likely",
            },
        ]
    }
}

ALERTS_RESPONSE = {
    "features": [
        {"properties": {
            "event": "Flash Flood Warning",
            "senderName": "NWS Chicago",
            "effective": "2026-08-09T12:00:00-05:00",
            "description": "The National Weather Service has issued a Flash Flood "
                           "Warning. Turn around, don't drown.",
            "instruction": "Move to higher ground immediately.",
        }}
    ]
}


def _mock_response(json_data, status_code=200):
    resp = type("Resp", (), {})()
    resp.status_code = status_code
    resp.raise_for_status = lambda: None
    resp.json = lambda: json_data
    return resp


def _mock_get(side_effect):
    return patch("weather_client.requests.get", side_effect=side_effect)


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


def test_missing_api_key_raises_for_geocoding():
    with patch.object(wc, "API_KEY", ""):
        try:
            wc._get(wc.GEO_URL, {"q": "Chicago"})
            assert False, "expected WeatherClientError"
        except wc.WeatherClientError as e:
            assert "OPENWEATHER_API_KEY" in str(e)


def test_needs_geocoding_true_for_place_name():
    assert wc.needs_geocoding(["Chicago", "41.5, -100.2"]) is True


def test_needs_geocoding_false_for_all_lat_lon():
    assert wc.needs_geocoding(["41.88, -87.63", "30.27, -97.74"]) is False


def test_normalize_alert_uses_description():
    alert = ALERTS_RESPONSE["features"][0]["properties"]
    doc = wc.normalize_alert(alert, "Chicago, Illinois, US")
    assert doc["source_type"] == "alert"
    assert doc["headline"] == "Flash Flood Warning"
    assert "Turn around" in doc["narrative_text"]
    assert doc["id"].startswith("alert:")
    assert doc["issued_at"] == "2026-08-09T12:00:00-05:00"


def test_normalize_alert_appends_instruction():
    alert = ALERTS_RESPONSE["features"][0]["properties"]
    doc = wc.normalize_alert(alert, "Chicago, Illinois, US")
    assert "Move to higher ground immediately" in doc["narrative_text"]


def test_normalize_alert_empty_description_returns_none():
    assert wc.normalize_alert({"event": "Test", "description": ""}, "X") is None


def test_normalize_forecast_period_uses_detailed_forecast():
    period = FORECAST_RESPONSE["properties"]["periods"][0]
    doc = wc.normalize_forecast_period(period, "Chicago, Illinois, US", 41.8781, -87.6298)
    assert doc["source_type"] == "forecast"
    assert "Showers and thunderstorms likely after noon" in doc["narrative_text"]
    assert doc["headline"] == "Today"
    assert doc["id"].startswith("forecast:")


def test_normalize_forecast_period_falls_back_to_short():
    period = {"name": "Today", "startTime": "2026-08-09T12:00:00-05:00",
              "detailedForecast": "", "shortForecast": "Showers Likely"}
    doc = wc.normalize_forecast_period(period, "Chicago, Illinois, US", 41.8781, -87.6298)
    assert doc["narrative_text"] == "Showers Likely"


def test_normalize_forecast_period_no_text_returns_none():
    assert wc.normalize_forecast_period(
        {"name": "Today", "startTime": "x", "detailedForecast": "", "shortForecast": ""},
        "X", 0, 0) is None


@_mock_get
def test_sync_locations_end_to_end(mock_get):
    def side_effect(url, **kwargs):
        if url == wc.GEO_URL:
            return _mock_response(GEOCODE_RESPONSE)
        if url == "https://api.weather.gov/points/41.8781,-87.6298":
            return _mock_response(POINTS_RESPONSE)
        if url == "https://api.weather.gov/gridpoints/LOT/76,73/forecast":
            return _mock_response(FORECAST_RESPONSE)
        if url == "https://api.weather.gov/alerts/active?point=41.8781,-87.6298":
            return _mock_response(ALERTS_RESPONSE)
        raise AssertionError(f"unexpected URL {url}")

    mock_get.side_effect = side_effect

    with patch.object(wc, "API_KEY", "fake-key"):
        docs = wc.sync_locations(["Chicago, IL"], limit=50)

    # 1 alert + 2 forecast periods = 3 documents
    assert len(docs) == 3
    source_types = {d["source_type"] for d in docs}
    assert source_types == {"alert", "forecast"}
    assert all(d["location"] == "Chicago, Illinois, US" for d in docs)


@_mock_get
def test_sync_locations_respects_limit(mock_get):
    def side_effect(url, **kwargs):
        if url == wc.GEO_URL:
            return _mock_response(GEOCODE_RESPONSE)
        if url == "https://api.weather.gov/points/41.8781,-87.6298":
            return _mock_response(POINTS_RESPONSE)
        if url == "https://api.weather.gov/gridpoints/LOT/76,73/forecast":
            return _mock_response(FORECAST_RESPONSE)
        if url == "https://api.weather.gov/alerts/active?point=41.8781,-87.6298":
            return _mock_response(ALERTS_RESPONSE)
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
