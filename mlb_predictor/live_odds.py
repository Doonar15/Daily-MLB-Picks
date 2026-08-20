"""Live MLB moneyline odds via The Odds API (https://the-odds-api.com), used
to compute real expected value for today's picks instead of relying only on
a static backtested confidence threshold.

This hits the LIVE odds endpoint (1 credit/request on the free tier), not
the historical one -- confirmed directly against this project's key that
historical is paid-only (401 HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN) but
live odds work fine on the free plan. At 1 credit/call this is cheap enough
(~500/month free budget) to just fetch fresh every predictor run.

Deliberately NOT disk-cached: cache.cached() keys and stores its payload
including the raw call args verbatim in the cache file (see cache.py's
set()), so decorating a function that takes api_key as an argument would
write the key to disk in plaintext. The API key is only ever read from the
ODDS_API_KEY environment variable, never written to any file.
"""

import os
import statistics
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# MLB's scheduling day boundary is Eastern time, not UTC -- a West coast
# night game's commence_time often rolls into the next UTC calendar date
# (e.g. an 8:41pm Pacific first pitch is already "tomorrow" in UTC) while
# still being "today" in api.get_schedule()'s date convention. Converting
# through Eastern before taking .date() keeps this module's date buckets
# aligned with the rest of the project's.
_MLB_TZ = ZoneInfo("America/New_York")


def _game_date_et(commence_time_utc: str):
    dt = datetime.fromisoformat(commence_time_utc.replace("Z", "+00:00")).astimezone(_MLB_TZ)
    return dt.date().isoformat()


def fetch_live_odds(api_key: str = None, target_date: str = None, timeout: int = 15):
    """Return {(home_team, away_team): (home_ml, away_ml)} for MLB games on
    target_date (default: today, in the same US-Eastern "MLB day" convention
    api.get_schedule() uses), using the median price across every bookmaker
    The Odds API returns as a robust consensus line (mirrors how the
    shanemcd.org files describe picking "a typical payoff" across sites
    rather than any one book). Team name strings match api.get_schedule()'s
    convention exactly (confirmed against a real response, including
    "Athletics").

    The live endpoint returns whatever games are upcoming/in-progress right
    now, which can span more than one date (e.g. the Mets play the Padres
    two days running). Filtering to target_date avoids the two games
    colliding on the (home_team, away_team) key and one silently
    overwriting the other -- without this, an early-evening query for
    today's game could return tomorrow's price instead. If the SAME
    matchup still appears twice after date-filtering (a true doubleheader),
    both are dropped rather than guessing which one a caller's game
    actually is -- callers fall back to a non-market-dependent rule for
    those specific games instead of risking a wrong-game price.

    Already-started games are excluded entirely, not just date-filtered:
    once first pitch happens, this same h2h market starts returning IN-PLAY
    prices instead of a pre-game line (confirmed directly -- a game 30
    minutes in with the home team already up 3-0 priced at -2000, not a
    real pre-game number). Every caller here treats the result as a
    pre-game price, so a mid-game price would silently corrupt whatever
    it's used for -- the model's probability blend, or the RECOMMENDED-pick
    decision -- with information the model itself was never given.

    Returns None if no API key is configured (api_key not passed and
    ODDS_API_KEY isn't set) or the request fails for any reason (network
    error, bad key, rate limit, malformed response) -- callers should treat
    None as "live odds unavailable this run" and fall back to a
    non-market-dependent rule, not as "zero games have odds today." An
    empty dict {} means the call succeeded but nothing matched target_date.
    """
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        return None

    if target_date is None:
        target_date = datetime.now(timezone.utc).astimezone(_MLB_TZ).date().isoformat()

    try:
        r = requests.get(
            f"{ODDS_API_BASE}/sports/baseball_mlb/odds/",
            params={"apiKey": key, "regions": "us", "markets": "h2h", "oddsFormat": "american"},
            timeout=timeout,
        )
        r.raise_for_status()
        games = r.json()
    except (requests.RequestException, ValueError):
        return None

    now = datetime.now(timezone.utc)
    odds = {}
    seen_twice = set()
    for g in games:
        home_team, away_team = g.get("home_team"), g.get("away_team")
        commence_time = g.get("commence_time")
        if not home_team or not away_team or not commence_time:
            continue
        try:
            if _game_date_et(commence_time) != target_date:
                continue
            # The live endpoint's docstring warns it returns "upcoming/
            # in-progress" games -- once a game has actually started, this
            # same h2h market key starts returning IN-PLAY prices instead
            # of a pre-game line (confirmed directly: a game 30 minutes in
            # with the home team already up 3-0 came back priced at -2000,
            # not a real pre-game number). Every caller of this function
            # treats its output as a pre-game price -- for model blending,
            # for the RECOMMENDED-pick decision -- so a started game must be
            # excluded here, not filtered ad hoc by every caller.
            if datetime.fromisoformat(commence_time.replace("Z", "+00:00")) <= now:
                continue
        except ValueError:
            continue

        key_tuple = (home_team, away_team)
        if key_tuple in odds:
            # True doubleheader -- can't tell which leg is which without a
            # game_pk match the API doesn't give us. Drop it rather than
            # silently pricing one game with the other's odds.
            seen_twice.add(key_tuple)
            continue

        home_prices, away_prices = [], []
        for book in g.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home_team:
                        home_prices.append(outcome["price"])
                    elif outcome.get("name") == away_team:
                        away_prices.append(outcome["price"])
        if not home_prices or not away_prices:
            continue
        odds[key_tuple] = (statistics.median(home_prices), statistics.median(away_prices))

    for key_tuple in seen_twice:
        odds.pop(key_tuple, None)

    return odds
