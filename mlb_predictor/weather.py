"""Game-time wind lookup via Open-Meteo (https://open-meteo.com), a free
forecast API that requires no API key.

Honest limitation: we don't have a reliable free source for each park's
home-plate-to-outfield orientation, so we can't correctly resolve "wind
blowing out to left" vs. "blowing in from center" per park. Rather than
fabricate a directional call, wind is used only as a magnitude signal: high
wind speed AMPLIFIES the park's existing static run factor (further from 100
in whichever direction it already leans), on the idea that windy conditions
generally increase scoring variance and tend to compound a park's existing
character (small hitter-friendly parks get more so, pitcher's parks with
swirling wind get harder to square up). This is a coarse proxy, not a precise
physical model.

Forecast data is only available for the next several days -- lookups for past
dates (backtesting/tuning) will return None, which callers should treat as
"no wind adjustment, use the static park factor as-is."
"""

from datetime import datetime, timezone

import requests

from . import cache

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TTL = 60 * 60  # forecasts shift run to run; refresh hourly

HIGH_WIND_MPH = 15.0   # above this, treat wind as a significant factor
MAX_AMPLIFICATION = 0.05  # cap on how much wind can further scale the park factor's deviation from neutral


@cache.cached(ttl_seconds=WEATHER_TTL)
def get_wind_speed_mph(lat: float, lon: float, game_datetime_utc: str):
    """Return forecast wind speed (mph) at the given coordinates for the hour
    closest to game_datetime_utc (an ISO8601 UTC string like
    '2026-08-03T22:40:00Z'). Returns None if unavailable (past date, API
    down, etc.) -- callers should fall back to the static park factor.
    """
    try:
        game_dt = datetime.fromisoformat(game_datetime_utc.replace("Z", "+00:00"))
    except ValueError:
        return None

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m",
        "wind_speed_unit": "mph",
        "forecast_days": 7,
        "timezone": "GMT",
    }
    try:
        r = requests.get(FORECAST_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException:
        return None

    times = data.get("hourly", {}).get("time", [])
    speeds = data.get("hourly", {}).get("wind_speed_10m", [])
    if not times or not speeds:
        return None

    target = game_dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    target_str = target.strftime("%Y-%m-%dT%H:%M")
    if target_str not in times:
        return None

    idx = times.index(target_str)
    return speeds[idx]


def wind_amplification_factor(wind_mph):
    """Return a multiplier (0.0 to MAX_AMPLIFICATION) representing how much
    high wind should amplify a park's existing run factor deviation from
    neutral. 0.0 below the significance threshold.
    """
    if wind_mph is None or wind_mph < HIGH_WIND_MPH:
        return 0.0
    excess = wind_mph - HIGH_WIND_MPH
    scaled = min(MAX_AMPLIFICATION, (excess / 20.0) * MAX_AMPLIFICATION + 0.02)
    return scaled
