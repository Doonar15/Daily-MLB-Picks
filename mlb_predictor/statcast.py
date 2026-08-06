"""Statcast-derived team quality signals (xwOBA batting, xwOBA-against
pitching), sourced from Baseball Savant via pybaseball.

Raw Statcast data is pitch-level (roughly 2,000+ rows per day across all
games) -- far more granular than we need. Rather than re-fetching and
re-aggregating that on every prediction, we pull each calendar day ONCE,
aggregate it down to one row per team per day (a compact rate stat: xwOBA,
barrel%, plate appearances), and store only that locally. A watermark date
tracks how much has been pulled, same pattern as calibration.py, so a daily
top-up only ever fetches the day(s) since the last run.

xwOBA formula (matches how Baseball Savant computes it): for each
plate-appearance-ending pitch, use estimated_woba_using_speedangle when the
PA ended on a ball in play (this is the "expected" part -- it removes
luck/defense from batted-ball outcomes), and the actual woba_value for PAs
that didn't involve a batted ball (walks, strikeouts, HBP -- there's no
"expected" version of a strikeout, so the real outcome is used as-is).
Team xwOBA-against (a pitching-quality proxy) is the same computation from
the opposing team's plate appearances.

Point-in-time by construction: because we aggregate from raw per-day pulls
ourselves rather than using pybaseball's season-only leaderboard functions
(which don't support a date range), any as-of-date query only ever includes
days before that date -- unlike the MLB Stats API's handedness/lineup
splits, which are season-only regardless of backtest date.
"""

import json
import os
import warnings
from datetime import date, timedelta
from pathlib import Path

from . import api

os.environ.setdefault("TQDM_DISABLE", "1")  # suppress pybaseball's progress bars in normal (non-interactive) use

STATCAST_DIR = Path(__file__).parent / ".history"
STATCAST_FILE = STATCAST_DIR / "statcast_team_days.jsonl"
WATERMARK_FILE = STATCAST_DIR / "statcast_watermark.json"

# Same reasoning as calibration.py's window, but Statcast pulls are pitch-level
# (much heavier per day than the game-level backtest data), so this needs its
# own real-world timing measurement before trusting anything larger.
DEFAULT_BACKFILL_START_DAYS_AGO = 120

MIN_PA_FOR_SIGNAL = 20  # below this, a team-day's xwOBA sample is too small to trust much


def _team_abbrev_to_id_map():
    """Build {abbreviation: team_id} from the MLB Stats API's team list, so
    we don't hardcode a mapping that could silently drift if an abbreviation
    ever changes. Statcast's home_team/away_team columns use the same
    abbreviation strings (including the 'AZ' vs the full 'ARI' quirk).
    """
    import requests
    r = requests.get(f"{api.BASE}/teams", params={"sportId": 1, "activeStatus": "Y"}, timeout=15)
    r.raise_for_status()
    return {t["abbreviation"]: t["id"] for t in r.json().get("teams", [])}


def _team_nickname_to_id_map():
    """Build {nickname: team_id} (e.g. 'Nationals' -> 120), matching the
    MLB Stats API's teamName field. Baseball Savant's OAA leaderboard
    identifies teams by nickname only (display_team_name), not the full
    "City Name" or an abbreviation.
    """
    import requests
    r = requests.get(f"{api.BASE}/teams", params={"sportId": 1, "activeStatus": "Y"}, timeout=15)
    r.raise_for_status()
    return {t["teamName"]: t["id"] for t in r.json().get("teams", [])}


@api.cache.cached(ttl_seconds=api.STATS_TTL)
def get_team_oaa(season: int):
    """Return {team_id: outs_above_average} for every team's current-season
    fielding, aggregated from player-level OAA. NOT point-in-time -- Baseball
    Savant's OAA leaderboard only exposes a season-to-date/season-end
    snapshot, with no raw per-play data we can aggregate ourselves the way
    xwOBA is built. This is the same limitation as the MLB Stats API's
    handedness/lineup splits: live predictions get a meaningful "as of today"
    number, but a backtest on a past date will see today's full-season OAA
    rather than what was true then. Returns {} if the pull fails.
    """
    import pybaseball as pb

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pb.statcast_outs_above_average(season, pos="all", min_att="q")
    except Exception:
        return {}
    if df is None or df.empty:
        return {}

    nickname_to_id = _team_nickname_to_id_map()
    team_oaa = {}
    for _, row in df.iterrows():
        team_id = nickname_to_id.get(row.get("display_team_name"))
        if team_id is None:
            continue  # '---' placeholder for traded/multi-team players, or an unmapped name
        team_oaa[team_id] = team_oaa.get(team_id, 0) + row["outs_above_average"]

    return team_oaa


def get_team_oaa_value(team_id: int, season: int):
    """Convenience wrapper: this team's OAA for the season, or None if
    unavailable. Fetches (and caches) the full-league table once per call
    site's cache TTL window rather than per-team, since the underlying pull
    returns everyone at once anyway.

    get_team_oaa's dict is disk-cached via cache.py, which round-trips
    through JSON -- JSON object keys are always strings, so a fresh
    (uncached) call returns int team_id keys but a cached call returns str
    keys. Look up both forms rather than relying on a specific type.
    """
    oaa = get_team_oaa(season)
    return oaa.get(team_id, oaa.get(str(team_id)))


_read_cache = None  # process-local cache of the parsed file; invalidated whenever we append


def _read_all():
    global _read_cache
    if _read_cache is not None:
        return _read_cache
    if not STATCAST_FILE.exists():
        _read_cache = []
        return _read_cache
    records = []
    for line in STATCAST_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    _read_cache = records
    return _read_cache


def _append_all(new_records):
    global _read_cache
    if not new_records:
        return
    STATCAST_DIR.mkdir(parents=True, exist_ok=True)
    with STATCAST_FILE.open("a") as f:
        for r in new_records:
            f.write(json.dumps(r) + "\n")
    _read_cache = None  # invalidate so the next read picks up the new records


def get_watermark():
    if not WATERMARK_FILE.exists():
        return None
    try:
        return json.loads(WATERMARK_FILE.read_text()).get("last_covered_date")
    except json.JSONDecodeError:
        return None


def _set_watermark(covered_through_date: str):
    STATCAST_DIR.mkdir(parents=True, exist_ok=True)
    WATERMARK_FILE.write_text(json.dumps({"last_covered_date": covered_through_date}))


def default_backfill_start():
    return (date.today() - timedelta(days=DEFAULT_BACKFILL_START_DAYS_AGO)).isoformat()


def _aggregate_day(raw_df, abbrev_to_id, day_str):
    """Reduce one day's raw pitch-level Statcast rows to one team-day record
    per team, for both batting (offense) and pitching-against (the team's
    pitchers, i.e. what opposing batters did against them) sides.
    """
    pa_rows = raw_df[raw_df["woba_denom"].notna() & (raw_df["woba_denom"] > 0)].copy()
    if pa_rows.empty:
        return []

    # Savant populates estimated_woba_using_speedangle on every PA-ending row,
    # not just batted balls (e.g. a walk gets a small league-average-based
    # expected value distinct from its actual wOBA value) -- use it directly
    # when present, falling back to the actual value only on the rare row
    # where it's missing.
    pa_rows["xwoba_component"] = pa_rows["estimated_woba_using_speedangle"].fillna(pa_rows["woba_value"])
    pa_rows["is_bip"] = pa_rows["type"] == "X"  # actual ball in play, for the barrel-rate denominator
    pa_rows["is_barrel"] = pa_rows.get("launch_speed_angle", 0) == 6  # Savant's barrel classification code

    records = []
    for game_pk, game_rows in pa_rows.groupby("game_pk"):
        home_abbrev = game_rows["home_team"].iloc[0]
        away_abbrev = game_rows["away_team"].iloc[0]
        home_id = abbrev_to_id.get(home_abbrev)
        away_id = abbrev_to_id.get(away_abbrev)
        if home_id is None or away_id is None:
            continue

        home_bat = game_rows[game_rows["inning_topbot"] == "Bot"]  # home team batting
        away_bat = game_rows[game_rows["inning_topbot"] == "Top"]  # away team batting

        def _team_stat(rows):
            denom = rows["woba_denom"].sum()
            if denom == 0:
                return None
            return {
                "xwoba_sum": float(rows["xwoba_component"].sum()),
                "pa": int(denom),
                "barrels": int(rows["is_barrel"].sum()),
                "bip": int(rows["is_bip"].sum()),
            }

        home_batting = _team_stat(home_bat)
        away_batting = _team_stat(away_bat)

        # A team's pitching-against xwOBA is what the OPPOSING batters did --
        # home team's pitchers faced away_bat, away team's pitchers faced home_bat.
        if home_batting:
            records.append({"date": day_str, "team_id": home_id, "side": "batting", **home_batting})
            records.append({"date": day_str, "team_id": away_id, "side": "pitching", **home_batting})
        if away_batting:
            records.append({"date": day_str, "team_id": away_id, "side": "batting", **away_batting})
            records.append({"date": day_str, "team_id": home_id, "side": "pitching", **away_batting})

    return records


def pull_day(day_str: str):
    """Pull, aggregate, and store one day's Statcast data. Returns the
    number of team-day records added (0 on an off-day or if the pull fails).
    """
    import pybaseball as pb

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_df = pb.statcast(start_dt=day_str, end_dt=day_str, verbose=False)
    except Exception:
        return 0

    if raw_df is None or raw_df.empty:
        return 0

    abbrev_to_id = _team_abbrev_to_id_map()
    records = _aggregate_day(raw_df, abbrev_to_id, day_str)
    _append_all(records)
    return len(records)


def backfill(start_date: str = None, end_date: str = None):
    """One-time (or re-run-to-extend) pull covering start_date through
    end_date (default: yesterday), one day at a time. Sequential, not
    parallel: pybaseball's underlying scrape doesn't document safe
    concurrency the way the MLB Stats API calls do, and each day is already
    a substantial pull on its own.
    """
    start_date = start_date or default_backfill_start()
    end_date = end_date or (date.today() - timedelta(days=1)).isoformat()
    if start_date > end_date:
        return 0

    total = 0
    day = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    while day <= end:
        total += pull_day(day.isoformat())
        day += timedelta(days=1)

    existing_watermark = get_watermark()
    if existing_watermark is None or end_date > existing_watermark:
        _set_watermark(end_date)

    return total


def top_up():
    """Fetch only the gap between the watermark and yesterday. Returns the
    number of newly-added team-day records. No-op if no backfill has
    happened yet or nothing new to cover.
    """
    watermark = get_watermark()
    if watermark is None:
        return 0

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    gap_start = (date.fromisoformat(watermark) + timedelta(days=1)).isoformat()
    if gap_start > yesterday:
        return 0

    return backfill(start_date=gap_start, end_date=yesterday)


def get_team_xwoba_as_of(team_id: int, as_of_date: str, side: str = "batting"):
    """Season-to-date, point-in-time xwOBA for a team's batting (side=
    'batting') or xwOBA-against for their pitching (side='pitching'),
    aggregated from all locally-stored team-days strictly before as_of_date
    within the same season. Returns None if there's not enough sample yet.
    """
    as_of = date.fromisoformat(as_of_date)
    season_start = date(as_of.year, 1, 1)  # cheap season boundary; off-season days are never in the store anyway

    total_xwoba, total_pa, total_barrels, total_bip = 0.0, 0, 0, 0
    for r in _read_all():
        if r["team_id"] != team_id or r["side"] != side:
            continue
        r_date = date.fromisoformat(r["date"])
        if not (season_start <= r_date < as_of):
            continue
        total_xwoba += r["xwoba_sum"]
        total_pa += r["pa"]
        total_barrels += r["barrels"]
        total_bip += r["bip"]

    if total_pa < MIN_PA_FOR_SIGNAL:
        return None

    return {
        "xwoba": total_xwoba / total_pa,
        "pa": total_pa,
        "barrel_pct": (total_barrels / total_bip) if total_bip else None,
    }
