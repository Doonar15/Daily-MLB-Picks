"""All calls to the free, public MLB Stats API (https://statsapi.mlb.com).
No API key required. Every fetch function is disk-cached (see cache.py) so
repeated runs within the TTL window don't re-hit the network.
"""

from datetime import date, timedelta

import requests

from . import cache

BASE = "https://statsapi.mlb.com/api/v1"
BASE_V11 = "https://statsapi.mlb.com/api/v1.1"

STATS_TTL = 4 * 60 * 60       # season/date-range stats: refresh every 4 hours
SCHEDULE_TTL = 60 * 60        # schedule/probables: refresh hourly (pitchers can change)
LINEUP_TTL = 15 * 60          # confirmed lineups: refresh every 15 min (posted ~game time)
HISTORICAL_TTL = 30 * 24 * 60 * 60  # past/completed data never changes: cache a month


def _get(url, params=None, timeout=15):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


@cache.cached(ttl_seconds=HISTORICAL_TTL)
def get_season_start_date(season: int):
    """Return the regular season's opening day (YYYY-MM-DD) for a given
    year, used as the start of byDateRange lookups for point-in-time stats.
    Falls back to a fixed late-March date if the API call fails.
    """
    try:
        data = _get(f"{BASE}/seasons", {"sportId": 1, "season": season})
        return data["seasons"][0]["regularSeasonStartDate"]
    except (KeyError, IndexError, requests.RequestException):
        return f"{season}-03-20"


def get_schedule(game_date: str, force_refresh: bool = False):
    """Return list of games (with probable pitchers) for a given date (YYYY-MM-DD).

    force_refresh bypasses the cache entirely -- used by grading, where an
    hour-old "still in progress" snapshot would silently block a genuinely
    final game from being graded until the cache naturally expired.
    """
    cache_key = f"schedule|{game_date}"
    is_past = game_date < date.today().isoformat()
    ttl = HISTORICAL_TTL if is_past else SCHEDULE_TTL

    if not force_refresh:
        cached_val = cache.get(cache_key, ttl)
        if cached_val is not None:
            return cached_val

    url = f"{BASE}/schedule"
    params = {
        "sportId": 1,
        "date": game_date,
        "hydrate": "probablePitcher,team",
    }
    data = _get(url, params)
    games = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            try:
                home = g["teams"]["home"]
                away = g["teams"]["away"]
                games.append({
                    "game_pk": g["gamePk"],
                    "status": g["status"]["detailedState"],
                    "coded_status": g["status"].get("codedGameState"),  # "F" = final; the authoritative code, not the display string
                    "game_datetime": g.get("gameDate"),
                    "home_team": home["team"]["name"],
                    "home_team_id": home["team"]["id"],
                    "away_team": away["team"]["name"],
                    "away_team_id": away["team"]["id"],
                    "home_pitcher": home.get("probablePitcher", {}).get("fullName"),
                    "home_pitcher_id": home.get("probablePitcher", {}).get("id"),
                    "away_pitcher": away.get("probablePitcher", {}).get("fullName"),
                    "away_pitcher_id": away.get("probablePitcher", {}).get("id"),
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                })
            except KeyError:
                continue

    cache.set(cache_key, games)
    return games


@cache.cached(ttl_seconds=STATS_TTL)
def get_team_run_stats(team_id: int, season: int):
    """Return (runs_scored, runs_allowed, games_played, team_era) for the
    team's full season through today.
    """
    url = f"{BASE}/teams/{team_id}/stats"
    hitting = _get(url, {"stats": "season", "group": "hitting", "season": season})
    runs_scored, games_played = None, None
    try:
        stat = hitting["stats"][0]["splits"][0]["stat"]
        runs_scored = int(stat["runs"])
        games_played = int(stat["gamesPlayed"])
    except (KeyError, IndexError):
        pass

    pitching = _get(url, {"stats": "season", "group": "pitching", "season": season})
    runs_allowed, team_era = None, None
    try:
        stat = pitching["stats"][0]["splits"][0]["stat"]
        runs_allowed = int(stat["runs"])
        team_era = float(stat["era"])
    except (KeyError, IndexError):
        pass

    return runs_scored, runs_allowed, games_played, team_era


@cache.cached(ttl_seconds=STATS_TTL)
def get_team_run_stats_as_of(team_id: int, season: int, as_of_date: str):
    """Same as get_team_run_stats, but scoped to games from the season's
    opening day through the day before as_of_date -- a true point-in-time
    read instead of the team's full current-season totals. This is what
    makes backtesting/tuning on a past date reflect what was actually known
    at that time, rather than "seeing the future" via today's cumulative
    stats. For as_of_date == today, this is equivalent to the season totals.
    """
    start = get_season_start_date(season)
    end = (date.fromisoformat(as_of_date) - timedelta(days=1)).isoformat()
    if end < start:
        return None, None, None, None  # asking about a date before the season started

    url = f"{BASE}/teams/{team_id}/stats"
    params_common = {"stats": "byDateRange", "startDate": start, "endDate": end}

    runs_scored, games_played = None, None
    try:
        data = _get(url, {**params_common, "group": "hitting"})
        stat = data["stats"][0]["splits"][0]["stat"]
        runs_scored = int(stat["runs"])
        games_played = int(stat["gamesPlayed"])
    except (KeyError, IndexError, requests.RequestException):
        pass

    runs_allowed, team_era = None, None
    try:
        data = _get(url, {**params_common, "group": "pitching"})
        stat = data["stats"][0]["splits"][0]["stat"]
        runs_allowed = int(stat["runs"])
        team_era = float(stat["era"])
    except (KeyError, IndexError, requests.RequestException):
        pass

    return runs_scored, runs_allowed, games_played, team_era


@cache.cached(ttl_seconds=STATS_TTL)
def get_pitcher_era_as_of(pitcher_id: int, season: int, as_of_date: str):
    """Same as get_pitcher_era, but scoped to starts before as_of_date
    instead of the pitcher's full current-season ERA.
    """
    if pitcher_id is None:
        return None
    start = get_season_start_date(season)
    end = (date.fromisoformat(as_of_date) - timedelta(days=1)).isoformat()
    if end < start:
        return None

    try:
        data = _get(
            f"{BASE}/people/{pitcher_id}/stats",
            {"stats": "byDateRange", "group": "pitching", "startDate": start, "endDate": end},
        )
        return float(data["stats"][0]["splits"][0]["stat"]["era"])
    except (KeyError, IndexError, ValueError, requests.RequestException):
        return None


@cache.cached(ttl_seconds=STATS_TTL)
def get_team_recent_form(team_id: int, season: int, as_of_date: str, games: int = 10):
    """Return (wins, losses) over the team's last N completed games strictly
    before as_of_date this season. Date-scoped so that backtesting a past
    game only sees results the team had actually played by that point,
    rather than the whole season including games "in the future" relative
    to the game being predicted.
    """
    as_of = date.fromisoformat(as_of_date)
    url = f"{BASE}/schedule"
    params = {
        "sportId": 1,
        "teamId": team_id,
        "season": season,
        "gameType": "R",
    }
    data = _get(url, params)
    results = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("codedGameState") != "F":
                continue
            try:
                game_date = date.fromisoformat(day["date"])
                if game_date >= as_of:
                    continue
                home = g["teams"]["home"]
                away = g["teams"]["away"]
                is_home = home["team"]["id"] == team_id
                team_side = home if is_home else away
                opp_side = away if is_home else home
                if team_side.get("score") is None or opp_side.get("score") is None:
                    continue
                won = team_side.get("isWinner", team_side["score"] > opp_side["score"])
                results.append((g["gameDate"], won))
            except (KeyError, ValueError):
                continue

    results.sort(key=lambda x: x[0])
    last_n = results[-games:]
    wins = sum(1 for _, won in last_n if won)
    losses = len(last_n) - wins
    return wins, losses


@cache.cached(ttl_seconds=STATS_TTL)
def get_pitcher_era(pitcher_id: int, season: int):
    """Return a probable starter's season ERA, or None if unavailable (e.g. TBD)."""
    if pitcher_id is None:
        return None
    url = f"{BASE}/people/{pitcher_id}/stats"
    params = {"stats": "season", "group": "pitching", "season": season}
    try:
        data = _get(url, params)
        return float(data["stats"][0]["splits"][0]["stat"]["era"])
    except (KeyError, IndexError, requests.RequestException):
        return None


@cache.cached(ttl_seconds=STATS_TTL)
def get_pitcher_game_log(pitcher_id: int, season: int):
    """Return this pitcher's starts so far this season, oldest first: a list
    of {"date", "strikeouts", "batters_faced"} dicts. Relief appearances
    (gamesStarted == 0) are excluded -- this is meant for projecting a
    probable starter's next start, not general workload.
    """
    if pitcher_id is None:
        return []
    url = f"{BASE}/people/{pitcher_id}/stats"
    params = {"stats": "gameLog", "group": "pitching", "season": season}
    try:
        data = _get(url, params)
        splits = data["stats"][0]["splits"]
    except (KeyError, IndexError, requests.RequestException):
        return []

    log = []
    for s in splits:
        stat = s.get("stat", {})
        if not stat.get("gamesStarted"):
            continue
        try:
            log.append({
                "date": s["date"],
                "strikeouts": int(stat["strikeOuts"]),
                "batters_faced": int(stat["battersFaced"]),
            })
        except (KeyError, ValueError):
            continue
    log.sort(key=lambda r: r["date"])
    return log


@cache.cached(ttl_seconds=STATS_TTL)
def get_bullpen_era_l30(team_id: int, as_of_date: str):
    """Return the team's bullpen (relief-only) ERA over the 30 days ending
    as_of_date, computed by aggregating earned runs / innings pitched across
    all active-roster pitchers' relief appearances (gamesStarted == 0).
    Returns None if data is unavailable.
    """
    end = date.fromisoformat(as_of_date)
    start = end - timedelta(days=30)

    try:
        roster = _get(f"{BASE}/teams/{team_id}/roster", {"rosterType": "active"})
    except requests.RequestException:
        return None

    pitcher_ids = [
        p["person"]["id"] for p in roster.get("roster", [])
        if p.get("position", {}).get("abbreviation") == "P"
    ]
    if not pitcher_ids:
        return None

    total_er, total_outs = 0.0, 0.0
    for pid in pitcher_ids:
        try:
            data = _get(
                f"{BASE}/people/{pid}/stats",
                {
                    "stats": "byDateRange",
                    "group": "pitching",
                    "startDate": start.isoformat(),
                    "endDate": end.isoformat(),
                },
            )
            stat = data["stats"][0]["splits"][0]["stat"]
        except (KeyError, IndexError, requests.RequestException):
            continue

        if stat.get("gamesStarted", 0) != 0:
            continue  # only count pitchers used exclusively in relief in this window

        try:
            er = float(stat["earnedRuns"])
            ip = float(stat["inningsPitched"])
        except (KeyError, ValueError):
            continue

        whole, _, frac = str(ip).partition(".")
        outs = int(whole) * 3 + int(frac or 0)
        total_er += er
        total_outs += outs

    if total_outs == 0:
        return None
    return round((total_er * 27) / total_outs, 2)


@cache.cached(ttl_seconds=LINEUP_TTL)
def get_confirmed_lineup(game_pk: int, home: bool):
    """Return list of player IDs in the confirmed batting order for a game,
    or None if lineups aren't posted yet.
    """
    try:
        data = _get(f"{BASE_V11}/game/{game_pk}/feed/live")
    except requests.RequestException:
        return None
    side = "home" if home else "away"
    order = data.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {}).get("battingOrder")
    return order or None


def get_live_game_state(game_pk: int):
    """Return this game's current live state (inning, outs, score), fetched
    fresh from the live feed on every call -- deliberately NOT disk-cached,
    since even the shortest existing TTL (LINEUP_TTL, 15 min) would show a
    stale score for an entire game to --scoreboard's 30-second polling loop.

    Never raises: a network failure or an unexpected payload shape returns a
    dict with "error" set and every other field None/False, so a caller can
    treat this game as "not final, unavailable this cycle" and keep going
    instead of crashing the whole display.

    "is_final" is codedGameState == "F" -- the same check already used
    elsewhere in this codebase (get_schedule, history.grade_predictions).
    NOT abstractGameState: a postponed game can report
    abstractGameState == "Final" while codedGameState == "D" and never
    actually reach a real final score.
    """
    try:
        data = _get(f"{BASE_V11}/game/{game_pk}/feed/live")
    except requests.RequestException as e:
        return {
            "game_pk": game_pk, "detailed_state": None, "is_final": False,
            "inning": None, "inning_state": None, "outs": None,
            "home_score": None, "away_score": None, "error": str(e),
        }

    try:
        status = data.get("gameData", {}).get("status", {})
        linescore = data.get("liveData", {}).get("linescore", {})
        home = linescore.get("teams", {}).get("home", {})
        away = linescore.get("teams", {}).get("away", {})
        return {
            "game_pk": game_pk,
            "detailed_state": status.get("detailedState"),
            "is_final": status.get("codedGameState") == "F",
            "inning": linescore.get("currentInning"),
            "inning_state": linescore.get("inningState"),
            "outs": linescore.get("outs"),
            "home_score": home.get("runs"),
            "away_score": away.get("runs"),
            "error": None,
        }
    except (KeyError, AttributeError) as e:
        return {
            "game_pk": game_pk, "detailed_state": None, "is_final": False,
            "inning": None, "inning_state": None, "outs": None,
            "home_score": None, "away_score": None, "error": str(e),
        }


@cache.cached(ttl_seconds=STATS_TTL)
def get_player_season_ops(player_id: int, season: int):
    """Return a hitter's season OPS, or None if unavailable."""
    try:
        data = _get(
            f"{BASE}/people/{player_id}/stats",
            {"stats": "season", "group": "hitting", "season": season},
        )
        return float(data["stats"][0]["splits"][0]["stat"]["ops"])
    except (KeyError, IndexError, ValueError, requests.RequestException):
        return None


@cache.cached(ttl_seconds=STATS_TTL)
def get_team_season_roster_ops(team_id: int, season: int):
    """Return the average season OPS across the team's active hitters, used
    as a baseline to compare a confirmed lineup's strength against, and as
    the season-OPS denominator for the handedness adjustment.

    Uses hydrate=person(stats(...)) to pull every roster player's season
    hitting line in a single API call instead of one call per player -- with
    --handedness on by default, this function runs for both teams in every
    game, so the old one-call-per-hitter version was the dominant cost in
    any multi-game run (a full backtest day easily meant 300+ extra calls).

    Known limitation: rosterType=active always returns the CURRENT roster
    regardless of `season`, so for a historical backtest date this is technically
    today's roster's stats-as-of-`season`, not the roster as it stood on that
    date. Roster turnover mid-season means this is an approximation for
    backtesting on old dates -- fine for recent dates, weaker the further back
    you go. Fixing this fully would need a date-scoped roster endpoint, which
    the free API doesn't offer.
    """
    try:
        roster = _get(
            f"{BASE}/teams/{team_id}/roster",
            {
                "rosterType": "active",
                "hydrate": f"person(stats(type=season,group=hitting,season={season}))",
            },
        )
    except requests.RequestException:
        return None

    ops_values = []
    for p in roster.get("roster", []):
        if p.get("position", {}).get("abbreviation") == "P":
            continue
        try:
            splits = p["person"]["stats"][0]["splits"]
            ops_values.append(float(splits[0]["stat"]["ops"]))
        except (KeyError, IndexError, ValueError):
            continue

    if not ops_values:
        return None
    return sum(ops_values) / len(ops_values)


@cache.cached(ttl_seconds=STATS_TTL)
def get_team_ops_vs_hand(team_id: int, hand: str, season: int):
    """Return the team's season OPS against a given throwing hand ('L' or
    'R'), via the sitCodes situational split. Returns None if unavailable.
    The free API's situational splits expose OPS, not wOBA.
    """
    sit_code = "vl" if hand == "L" else "vr"
    try:
        data = _get(
            f"{BASE}/teams/{team_id}/stats",
            {"stats": "season", "group": "hitting", "season": season, "sitCodes": sit_code},
        )
        return float(data["stats"][0]["splits"][0]["stat"]["ops"])
    except (KeyError, IndexError, ValueError, requests.RequestException):
        return None


@cache.cached(ttl_seconds=HISTORICAL_TTL)
def get_pitcher_hand(pitcher_id: int):
    """Return a pitcher's throwing hand, 'L' or 'R', or None if unknown.
    A pitcher's hand doesn't change season to season, so this is cached long.
    """
    if pitcher_id is None:
        return None
    try:
        data = _get(f"{BASE}/people/{pitcher_id}")
        return data["people"][0]["pitchHand"]["code"]
    except (KeyError, IndexError, requests.RequestException):
        return None


@cache.cached(ttl_seconds=STATS_TTL)
def get_pitcher_days_rest(pitcher_id: int, as_of_date: str, season: int):
    """Return days between the pitcher's most recent start before
    as_of_date and as_of_date itself, or None if no prior start is found
    this season (season debut, or reliever with no starts).
    """
    if pitcher_id is None:
        return None
    try:
        data = _get(
            f"{BASE}/people/{pitcher_id}/stats",
            {"stats": "gameLog", "group": "pitching", "season": season},
        )
        splits = data["stats"][0]["splits"]
    except (KeyError, IndexError, requests.RequestException):
        return None

    as_of = date.fromisoformat(as_of_date)
    start_dates = [
        date.fromisoformat(s["date"]) for s in splits
        if s.get("stat", {}).get("gamesStarted") and date.fromisoformat(s["date"]) < as_of
    ]
    if not start_dates:
        return None
    return (as_of - max(start_dates)).days


HEAVY_WORKLOAD_PITCH_THRESHOLD = 20  # an appearance at/above this many pitches counts as "heavy" for availability purposes
AVAILABILITY_LOOKBACK_DAYS = 2       # how recently a heavy appearance still counts against today's availability
TOP_RELIEVER_COUNT = 2               # how many of a team's highest-leverage relievers we track for availability


@cache.cached(ttl_seconds=STATS_TTL)
def get_top_reliever_ids(team_id: int, as_of_date: str, season: int):
    """Return up to TOP_RELIEVER_COUNT pitcher IDs for the team's
    highest-leverage relievers this season (ranked by saves + holds,
    point-in-time as of the day before as_of_date), or [] if unavailable.
    Falls back to nothing rather than guessing if no reliever has recorded
    a save or hold yet (e.g. very early season).
    """
    start = get_season_start_date(season)
    end = (date.fromisoformat(as_of_date) - timedelta(days=1)).isoformat()
    if end < start:
        return []

    try:
        roster = _get(f"{BASE}/teams/{team_id}/roster", {"rosterType": "active"})
    except requests.RequestException:
        return []
    pitcher_ids = [
        p["person"]["id"] for p in roster.get("roster", [])
        if p.get("position", {}).get("abbreviation") == "P"
    ]

    leverage = []
    for pid in pitcher_ids:
        try:
            data = _get(
                f"{BASE}/people/{pid}/stats",
                {"stats": "byDateRange", "group": "pitching", "startDate": start, "endDate": end},
            )
            stat = data["stats"][0]["splits"][0]["stat"]
        except (KeyError, IndexError, requests.RequestException):
            continue
        score = stat.get("saves", 0) + stat.get("holds", 0)
        if score > 0:
            leverage.append((score, pid))

    leverage.sort(reverse=True)
    return [pid for _, pid in leverage[:TOP_RELIEVER_COUNT]]


@cache.cached(ttl_seconds=STATS_TTL)
def get_recent_heavy_appearance(pitcher_id: int, as_of_date: str, season: int):
    """Return True if this pitcher threw a heavy-workload appearance
    (>= HEAVY_WORKLOAD_PITCH_THRESHOLD pitches) within AVAILABILITY_LOOKBACK_DAYS
    of as_of_date, a proxy for "probably unavailable or fatigued today."
    Returns False if there's no such appearance or data is unavailable.
    """
    if pitcher_id is None:
        return False
    try:
        data = _get(
            f"{BASE}/people/{pitcher_id}/stats",
            {"stats": "gameLog", "group": "pitching", "season": season},
        )
        splits = data["stats"][0]["splits"]
    except (KeyError, IndexError, requests.RequestException):
        return False

    as_of = date.fromisoformat(as_of_date)
    cutoff = as_of - timedelta(days=AVAILABILITY_LOOKBACK_DAYS)
    for s in splits:
        try:
            appearance_date = date.fromisoformat(s["date"])
        except ValueError:
            continue
        if not (cutoff <= appearance_date < as_of):
            continue
        pitches = s.get("stat", {}).get("numberOfPitches", 0)
        if pitches >= HEAVY_WORKLOAD_PITCH_THRESHOLD:
            return True
    return False


def get_bullpen_availability_penalty_inputs(team_id: int, as_of_date: str, season: int):
    """Return (top_reliever_count, fatigued_count) for a team as of
    as_of_date -- how many of their top relievers we're tracking, and how
    many of those recently threw a heavy-workload appearance. Both 0 if no
    top relievers are identified yet (e.g. early season).
    """
    top_ids = get_top_reliever_ids(team_id, as_of_date, season)
    if not top_ids:
        return 0, 0
    fatigued = sum(1 for pid in top_ids if get_recent_heavy_appearance(pid, as_of_date, season))
    return len(top_ids), fatigued


@cache.cached(ttl_seconds=SCHEDULE_TTL)
def get_previous_day_game_location(team_id: int, as_of_date: str):
    """Return the home_team_id of the park this team played in the day
    before as_of_date, or None if they didn't play (off day) or the lookup
    fails. Used to detect travel: if that park's team_id isn't this team's
    own, they were on the road; the caller compares parks to judge distance
    /timezone change.
    """
    prev_day = (date.fromisoformat(as_of_date) - timedelta(days=1)).isoformat()
    try:
        games = get_schedule(prev_day)
    except requests.RequestException:
        return None
    for g in games:
        if g["home_team_id"] == team_id or g["away_team_id"] == team_id:
            return g["home_team_id"]
    return None


@cache.cached(ttl_seconds=STATS_TTL)
def get_home_road_splits(team_id: int, season: int, as_of_date: str):
    """Return ((home_wins, home_losses), (road_wins, road_losses)) for this
    team's completed games strictly before as_of_date this season.
    Point-in-time by construction (only past results are ever counted).
    """
    as_of = date.fromisoformat(as_of_date)
    try:
        data = _get(f"{BASE}/schedule", {"sportId": 1, "teamId": team_id, "season": season, "gameType": "R"})
    except requests.RequestException:
        return (0, 0), (0, 0)

    home_w = home_l = road_w = road_l = 0
    for day in data.get("dates", []):
        try:
            game_date = date.fromisoformat(day["date"])
        except ValueError:
            continue
        if game_date >= as_of:
            continue
        for g in day.get("games", []):
            if g.get("status", {}).get("codedGameState") != "F":
                continue
            try:
                home = g["teams"]["home"]
                away = g["teams"]["away"]
                is_home = home["team"]["id"] == team_id
                team_side = home if is_home else away
                opp_side = away if is_home else home
                if team_side.get("score") is None or opp_side.get("score") is None:
                    continue
                won = team_side.get("isWinner", team_side["score"] > opp_side["score"])
            except KeyError:
                continue
            if is_home:
                home_w, home_l = home_w + won, home_l + (not won)
            else:
                road_w, road_l = road_w + won, road_l + (not won)

    return (home_w, home_l), (road_w, road_l)


@cache.cached(ttl_seconds=STATS_TTL)
def get_head_to_head_record(team_id: int, opponent_id: int, season: int, as_of_date: str):
    """Return (wins, losses) for team_id against opponent_id this season,
    counting only completed games strictly before as_of_date -- point-in-time
    by construction since we only ever look at past results.
    """
    as_of = date.fromisoformat(as_of_date)
    try:
        data = _get(
            f"{BASE}/schedule",
            {"sportId": 1, "season": season, "teamId": team_id, "opponentId": opponent_id, "gameType": "R"},
        )
    except requests.RequestException:
        return 0, 0

    wins = losses = 0
    for day in data.get("dates", []):
        try:
            game_date = date.fromisoformat(day["date"])
        except ValueError:
            continue
        if game_date >= as_of:
            continue
        for g in day.get("games", []):
            if g.get("status", {}).get("codedGameState") != "F":
                continue
            try:
                home = g["teams"]["home"]
                away = g["teams"]["away"]
                is_home = home["team"]["id"] == team_id
                team_side = home if is_home else away
                opp_side = away if is_home else home
                if team_side.get("score") is None or opp_side.get("score") is None:
                    continue
                won = team_side.get("isWinner", team_side["score"] > opp_side["score"])
            except KeyError:
                continue
            if won:
                wins += 1
            else:
                losses += 1

    return wins, losses
