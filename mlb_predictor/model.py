"""Prediction model: combines season Pythagorean win expectation (adjusted
for park factor and blended with a Statcast xwOBA-based win expectation),
recent form, starting pitcher quality, bullpen quality, confirmed-lineup
strength, handedness matchup, starter rest, and home field advantage into a
single home-team win probability.

All tunable weights live in the Weights dataclass so they can be swapped out
for auto-tuned values (see backtest.tune_weights) without touching this
module's logic.
"""

from dataclasses import dataclass

from . import api, parks, statcast, weather


@dataclass(frozen=True)
class Weights:
    home_field_edge: float = 0.04     # ~54/46 baseline home advantage in MLB, applied additively
    recent_form_weight: float = 0.20  # weight given to last-10-games win% vs. season Pythagorean
    pitcher_weight: float = 0.15
    pitcher_cap: float = 0.08
    bullpen_weight: float = 0.08      # smaller than starter weight: bullpen matters less than the starter
    bullpen_cap: float = 0.05
    lineup_weight: float = 0.06
    lineup_cap: float = 0.04
    handedness_weight: float = 0.10
    handedness_cap: float = 0.05
    rest_weight: float = 0.03
    rest_cap: float = 0.03
    statcast_blend_weight: float = 0.35  # how much the xwOBA-based win% counts vs. the raw-runs Pythagorean win%
    defense_weight: float = 0.5  # scales the fixed runs-per-out conversion in defense_adjusted_runs_allowed
    bullpen_availability_weight: float = 0.04
    bullpen_availability_cap: float = 0.04
    travel_weight: float = 0.02
    travel_cap: float = 0.03
    h2h_weight: float = 0.15
    h2h_cap: float = 0.04
    home_road_split_weight: float = 0.20
    home_road_split_cap: float = 0.04


DEFAULT_WEIGHTS = Weights()
PROB_FLOOR, PROB_CEIL = 0.05, 0.95
FULL_REST_DAYS = 4  # typical 5-man rotation gives a starter 4 days between starts; fewer is "short rest"


def pythagorean_win_pct(runs_scored, runs_allowed, exponent=1.83):
    """MLB-calibrated Pythagorean expectation (Bill James, exponent ~1.83)."""
    if not runs_scored or not runs_allowed:
        return 0.5
    rs_exp = runs_scored ** exponent
    ra_exp = runs_allowed ** exponent
    return rs_exp / (rs_exp + ra_exp)


def xwoba_win_pct(batting_xwoba, pitching_xwoba_against, exponent=1.83):
    """A Pythagorean-style win expectation computed from Statcast xwOBA
    instead of actual runs scored/allowed. xwOBA (a team's expected weighted
    on-base average when batting) and xwOBA-against (what opposing batters
    have done against this team's pitching) both scale roughly linearly with
    scoring rate, so plugging them into the same Pythagorean exponent formula
    as a proxy for "runs" gives a comparable win-expectation signal that's
    been stripped of the luck/defense noise present in actual runs scored --
    without needing an external, year-specific wOBA-to-runs conversion
    constant, since we only need the RELATIVE comparison between two teams.
    Returns None if either input is missing (not enough Statcast sample yet).
    """
    if batting_xwoba is None or pitching_xwoba_against is None:
        return None
    if batting_xwoba <= 0 or pitching_xwoba_against <= 0:
        return None
    off_exp = batting_xwoba ** exponent
    def_exp = pitching_xwoba_against ** exponent
    return off_exp / (off_exp + def_exp)


def blend_recent_form(season_pyth, wins, losses, weight):
    """Blend season-long Pythagorean win% with recent (last-N-games) win% so
    hot/cold streaks nudge the estimate without overreacting to small samples.
    """
    games = wins + losses
    if games == 0:
        return season_pyth
    recent_pct = wins / games
    return (1 - weight) * season_pyth + weight * recent_pct


def pitcher_adjustment(starter_era, team_era, weight, cap):
    """Nudge win probability based on how much better/worse the probable
    starter's ERA is than the team's overall rotation ERA.
    """
    if starter_era is None or team_era is None or team_era == 0:
        return 0.0
    era_diff = team_era - starter_era  # positive = starter better than team average
    adj = (era_diff / team_era) * weight
    return max(-cap, min(cap, adj))


def bullpen_adjustment(bullpen_era, team_era, weight, cap):
    """Same idea as pitcher_adjustment but for bullpen ERA (L30) vs. team
    season ERA, weighted lower since the bullpen only covers part of a game.
    """
    if bullpen_era is None or team_era is None or team_era == 0:
        return 0.0
    era_diff = team_era - bullpen_era
    adj = (era_diff / team_era) * weight
    return max(-cap, min(cap, adj))


def lineup_adjustment(lineup_ops, roster_avg_ops, weight, cap):
    """Nudge win probability based on whether the confirmed starting lineup's
    average OPS is above/below the team's full-roster season average --
    a proxy for "are the best hitters actually playing today."
    Returns 0.0 if lineup isn't posted yet or data is unavailable.
    """
    if lineup_ops is None or not roster_avg_ops:
        return 0.0
    ops_diff = lineup_ops - roster_avg_ops
    adj = (ops_diff / roster_avg_ops) * weight
    return max(-cap, min(cap, adj))


def park_adjusted_runs(runs_scored, runs_allowed, games_played, park_run_factor):
    """Scale a team's season runs scored/allowed to reflect playing this
    specific game in park_run_factor's park (100 = neutral) instead of their
    season-average mix of parks. Both teams share the same park for a given
    game, since the home park is what's in effect regardless of which team
    is batting or pitching.
    """
    if not runs_scored or not runs_allowed or not games_played:
        return runs_scored, runs_allowed
    factor = park_run_factor / 100.0
    # Roughly half of a team's games are on the road, so their season totals
    # are already a blend of ~half neutral/road parks and ~half their own
    # park; scale by the sqrt of the factor to approximate "this one game's"
    # park effect without double-counting the home park's own contribution.
    game_factor = factor ** 0.5
    return runs_scored * game_factor, runs_allowed * game_factor


RUNS_PER_OUT = 0.75  # standard sabermetric approximation: each out above/below average is worth ~0.75 runs
DEFENSE_MAX_RUN_SWING = 0.15  # cap the defensive adjustment at +/-15% of runs allowed, so an outlier OAA can't dominate


def defense_adjusted_runs_allowed(runs_allowed, games_played, team_oaa, weight):
    """Scale a team's runs-allowed input by their season OAA (outs above
    average, fielding) -- good defense turns some batted balls that would
    otherwise be hits into outs, so part of a team's "runs allowed" total is
    really a fielding effect independent of the pitching staff. NOT
    point-in-time (see statcast.get_team_oaa's docstring); returns
    runs_allowed unchanged if OAA or games_played is unavailable.

    weight lets the auto-tuner scale down the raw runs-per-out conversion
    (which is a fixed sabermetric approximation, not something to search
    over directly) without touching the physical constant itself.
    """
    if not runs_allowed or not games_played or team_oaa is None:
        return runs_allowed
    runs_saved = team_oaa * RUNS_PER_OUT * weight
    swing = max(-DEFENSE_MAX_RUN_SWING, min(DEFENSE_MAX_RUN_SWING, runs_saved / runs_allowed))
    return runs_allowed * (1 - swing)


def handedness_adjustment(team_ops_vs_hand, team_season_ops, weight, cap):
    """Nudge win probability based on how a team's OPS against the
    opposing starter's throwing hand compares to their overall season OPS --
    a team that mashes lefties gets a boost when facing a lefty starter.
    """
    if team_ops_vs_hand is None or not team_season_ops:
        return 0.0
    ops_diff = team_ops_vs_hand - team_season_ops
    adj = (ops_diff / team_season_ops) * weight
    return max(-cap, min(cap, adj))


def rest_adjustment(days_rest, weight, cap):
    """Penalize a starter working on short rest (fewer than FULL_REST_DAYS
    days since their last start). Returns 0.0 if rest is normal/long or the
    pitcher's rest is unknown (season debut, reliever, TBD).
    """
    if days_rest is None or days_rest >= FULL_REST_DAYS:
        return 0.0
    shortfall = FULL_REST_DAYS - days_rest
    adj = -(shortfall / FULL_REST_DAYS) * weight
    return max(-cap, adj)


def bullpen_availability_adjustment(top_reliever_count, fatigued_count, weight, cap):
    """Penalize a team based on the fraction of their top (highest-leverage)
    relievers who threw a heavy-workload appearance very recently -- a proxy
    for "probably unavailable or gassed today," independent of the team's
    overall bullpen ERA. Returns 0.0 if no top relievers were identified
    (e.g. very early season, nobody has a save/hold yet).
    """
    if not top_reliever_count:
        return 0.0
    fraction_fatigued = fatigued_count / top_reliever_count
    return max(-cap, -fraction_fatigued * weight)


def travel_status(this_game_home_team_id, prev_game_home_team_id):
    """Classify a team's travel situation for today's game based on where
    they played yesterday. Returns one of:
      'none'      -- no game yesterday (off day), or stayed in the same city
      'same_zone' -- traveled to a new city, but within the same timezone
      'new_zone'  -- traveled to a new city AND crossed a timezone
    prev_game_home_team_id is None if the team didn't play yesterday.
    """
    if prev_game_home_team_id is None:
        return "none"
    if prev_game_home_team_id == this_game_home_team_id:
        return "none"  # same city yesterday and today (true even for the home team every day)
    prev_zone = parks.timezone_bucket(prev_game_home_team_id)
    this_zone = parks.timezone_bucket(this_game_home_team_id)
    return "new_zone" if prev_zone != this_zone else "same_zone"


def travel_adjustment(status, weight, cap):
    """Penalize a team for getaway-day travel -- a bigger penalty when it
    also crossed a timezone, since that compounds with the physical travel.
    """
    if status == "new_zone":
        return max(-cap, -weight)
    if status == "same_zone":
        return max(-cap, -weight * 0.4)
    return 0.0


MIN_H2H_GAMES = 3  # below this sample, head-to-head history is too noisy to use


def head_to_head_adjustment(wins, losses, weight, cap):
    """Nudge win probability based on this team's record against THIS
    opponent specifically this season, once there's enough of a sample
    (some matchups run hot/cold independent of overall team quality).
    Returns 0.0 below MIN_H2H_GAMES games played between the two teams.
    """
    games = wins + losses
    if games < MIN_H2H_GAMES:
        return 0.0
    win_pct = wins / games
    adj = (win_pct - 0.5) * weight
    return max(-cap, min(cap, adj))


MIN_SPLIT_GAMES = 10  # below this, a home-or-road split is too small a sample to trust


def home_road_split_adjustment(split_record, overall_win_pct, weight, cap):
    """Nudge win probability based on whether a team is unusually strong or
    weak specifically in this game's venue context (home or road) relative
    to their own overall record -- beyond the flat league-wide home field
    edge already applied elsewhere. split_record is (wins, losses) for
    whichever side (home or road) applies to this team in this game.
    Returns 0.0 below MIN_SPLIT_GAMES games in that split.
    """
    wins, losses = split_record
    games = wins + losses
    if games < MIN_SPLIT_GAMES or overall_win_pct is None:
        return 0.0
    split_win_pct = wins / games
    adj = (split_win_pct - overall_win_pct) * weight
    return max(-cap, min(cap, adj))


def moneyline_from_prob(p):
    """Convert a win probability into 'fair' American moneyline odds (no vig)."""
    if p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        return round(-100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def implied_prob_from_moneyline(ml):
    """Convert American moneyline odds into implied win probability."""
    ml = float(ml)
    if ml > 0:
        return 100 / (ml + 100)
    return -ml / (-ml + 100)


def _lineup_strength(team_id, season, game_pk, is_home, use_lineups):
    if not use_lineups:
        return None, None
    order = api.get_confirmed_lineup(game_pk, home=is_home)
    if not order:
        return None, None
    ops_values = [api.get_player_season_ops(pid, season) for pid in order]
    ops_values = [v for v in ops_values if v is not None]
    lineup_ops = sum(ops_values) / len(ops_values) if ops_values else None
    roster_avg_ops = api.get_team_season_roster_ops(team_id, season)
    return lineup_ops, roster_avg_ops


def compute_win_prob(inputs, weights: Weights = DEFAULT_WEIGHTS):
    """Pure function: given a dict of raw per-game inputs (see
    backtest.fetch_backtest_dataset / predict_game below for the shape),
    compute the home team's win probability. No network calls -- this is
    the function the weight tuner calls repeatedly.
    """
    park_factor = inputs.get("park_run_factor", 100)
    home_rs, home_ra = park_adjusted_runs(
        inputs["home_runs_scored"], inputs["home_runs_allowed"], inputs.get("home_games_played"), park_factor
    )
    away_rs, away_ra = park_adjusted_runs(
        inputs["away_runs_scored"], inputs["away_runs_allowed"], inputs.get("away_games_played"), park_factor
    )

    # Defense: scale each team's (already park-adjusted) runs-allowed by
    # their own OAA, so good fielding gets some credit currently folded
    # entirely into "runs allowed" alongside pitching quality.
    home_ra = defense_adjusted_runs_allowed(
        home_ra, inputs.get("home_games_played"), inputs.get("home_oaa"), weights.defense_weight
    )
    away_ra = defense_adjusted_runs_allowed(
        away_ra, inputs.get("away_games_played"), inputs.get("away_oaa"), weights.defense_weight
    )

    home_pyth = pythagorean_win_pct(home_rs, home_ra)
    away_pyth = pythagorean_win_pct(away_rs, away_ra)

    # Blend in the Statcast xwOBA-based win expectation when available. This
    # is a per-team blend (each team's raw-runs Pythagorean number is nudged
    # toward its own xwOBA-implied number), not a head-to-head calculation --
    # keeps the two teams' numbers independent until the final ratio step
    # below, consistent with how the rest of this function is structured.
    home_xwoba_pyth = xwoba_win_pct(inputs.get("home_xwoba"), inputs.get("home_xwoba_against"))
    away_xwoba_pyth = xwoba_win_pct(inputs.get("away_xwoba"), inputs.get("away_xwoba_against"))
    if home_xwoba_pyth is not None:
        w = weights.statcast_blend_weight
        home_pyth = (1 - w) * home_pyth + w * home_xwoba_pyth
    if away_xwoba_pyth is not None:
        w = weights.statcast_blend_weight
        away_pyth = (1 - w) * away_pyth + w * away_xwoba_pyth

    home_w, home_l = inputs["home_recent_form"]
    away_w, away_l = inputs["away_recent_form"]
    home_pyth = blend_recent_form(home_pyth, home_w, home_l, weights.recent_form_weight)
    away_pyth = blend_recent_form(away_pyth, away_w, away_l, weights.recent_form_weight)

    base_home_prob = home_pyth / (home_pyth + away_pyth) if (home_pyth + away_pyth) else 0.5

    home_adj = pitcher_adjustment(
        inputs["home_pitcher_era"], inputs["home_era"], weights.pitcher_weight, weights.pitcher_cap
    )
    away_adj = pitcher_adjustment(
        inputs["away_pitcher_era"], inputs["away_era"], weights.pitcher_weight, weights.pitcher_cap
    )

    bullpen_adj_diff = 0.0
    if inputs.get("home_bullpen_era") is not None or inputs.get("away_bullpen_era") is not None:
        bullpen_adj_diff = (
            bullpen_adjustment(inputs.get("home_bullpen_era"), inputs["home_era"], weights.bullpen_weight, weights.bullpen_cap)
            - bullpen_adjustment(inputs.get("away_bullpen_era"), inputs["away_era"], weights.bullpen_weight, weights.bullpen_cap)
        )

    lineup_adj_diff = 0.0
    if inputs.get("home_lineup_ops") is not None or inputs.get("away_lineup_ops") is not None:
        lineup_adj_diff = (
            lineup_adjustment(inputs.get("home_lineup_ops"), inputs.get("home_roster_ops"), weights.lineup_weight, weights.lineup_cap)
            - lineup_adjustment(inputs.get("away_lineup_ops"), inputs.get("away_roster_ops"), weights.lineup_weight, weights.lineup_cap)
        )

    # Handedness: home team's bat-vs-starter-hand edge minus away team's,
    # each compared against the OPPOSING starter (who they're actually facing).
    handedness_adj_diff = 0.0
    if inputs.get("home_ops_vs_away_hand") is not None or inputs.get("away_ops_vs_home_hand") is not None:
        handedness_adj_diff = (
            handedness_adjustment(inputs.get("home_ops_vs_away_hand"), inputs.get("home_season_ops"), weights.handedness_weight, weights.handedness_cap)
            - handedness_adjustment(inputs.get("away_ops_vs_home_hand"), inputs.get("away_season_ops"), weights.handedness_weight, weights.handedness_cap)
        )

    rest_adj_diff = 0.0
    if inputs.get("home_pitcher_rest_days") is not None or inputs.get("away_pitcher_rest_days") is not None:
        rest_adj_diff = (
            rest_adjustment(inputs.get("home_pitcher_rest_days"), weights.rest_weight, weights.rest_cap)
            - rest_adjustment(inputs.get("away_pitcher_rest_days"), weights.rest_weight, weights.rest_cap)
        )

    bullpen_avail_adj_diff = 0.0
    if inputs.get("home_top_reliever_count") or inputs.get("away_top_reliever_count"):
        bullpen_avail_adj_diff = (
            bullpen_availability_adjustment(
                inputs.get("home_top_reliever_count", 0), inputs.get("home_fatigued_reliever_count", 0),
                weights.bullpen_availability_weight, weights.bullpen_availability_cap,
            )
            - bullpen_availability_adjustment(
                inputs.get("away_top_reliever_count", 0), inputs.get("away_fatigued_reliever_count", 0),
                weights.bullpen_availability_weight, weights.bullpen_availability_cap,
            )
        )

    travel_adj_diff = 0.0
    if inputs.get("home_travel_status") or inputs.get("away_travel_status"):
        travel_adj_diff = (
            travel_adjustment(inputs.get("home_travel_status", "none"), weights.travel_weight, weights.travel_cap)
            - travel_adjustment(inputs.get("away_travel_status", "none"), weights.travel_weight, weights.travel_cap)
        )

    h2h_adj_diff = 0.0
    if inputs.get("home_h2h_record") or inputs.get("away_h2h_record"):
        home_h2h_w, home_h2h_l = inputs.get("home_h2h_record", (0, 0))
        away_h2h_w, away_h2h_l = inputs.get("away_h2h_record", (0, 0))
        h2h_adj_diff = (
            head_to_head_adjustment(home_h2h_w, home_h2h_l, weights.h2h_weight, weights.h2h_cap)
            - head_to_head_adjustment(away_h2h_w, away_h2h_l, weights.h2h_weight, weights.h2h_cap)
        )

    home_road_adj_diff = 0.0
    if inputs.get("home_home_split") or inputs.get("away_road_split"):
        home_split_w, home_split_l = inputs.get("home_home_split", (0, 0))
        away_split_w, away_split_l = inputs.get("away_road_split", (0, 0))
        home_road_adj_diff = (
            home_road_split_adjustment(
                (home_split_w, home_split_l), inputs.get("home_overall_win_pct"),
                weights.home_road_split_weight, weights.home_road_split_cap,
            )
            - home_road_split_adjustment(
                (away_split_w, away_split_l), inputs.get("away_overall_win_pct"),
                weights.home_road_split_weight, weights.home_road_split_cap,
            )
        )

    home_prob = (
        base_home_prob
        + home_adj - away_adj
        + bullpen_adj_diff
        + lineup_adj_diff
        + handedness_adj_diff
        + rest_adj_diff
        + bullpen_avail_adj_diff
        + travel_adj_diff
        + h2h_adj_diff
        + home_road_adj_diff
        + weights.home_field_edge
    )
    return max(PROB_FLOOR, min(PROB_CEIL, home_prob))


def _park_run_factor(home_team_id, game_datetime, use_weather):
    park = parks.get_park(home_team_id)
    factor = park.run_factor
    wind_mph = None
    if use_weather and not park.is_dome and game_datetime:
        wind_mph = weather.get_wind_speed_mph(park.lat, park.lon, game_datetime)
        amp = weather.wind_amplification_factor(wind_mph)
        if amp:
            # amplify the park's existing deviation from neutral (100)
            deviation = factor - 100
            factor = 100 + deviation * (1 + amp)
    return factor, wind_mph


def _statcast_inputs(team_id, as_of_date, use_statcast):
    if not use_statcast:
        return None, None
    batting = statcast.get_team_xwoba_as_of(team_id, as_of_date, side="batting")
    pitching = statcast.get_team_xwoba_as_of(team_id, as_of_date, side="pitching")
    return (batting["xwoba"] if batting else None), (pitching["xwoba"] if pitching else None)


def gather_game_inputs(game, season, as_of_date, use_bullpen=False, use_lineups=False,
                        use_park_factors=False, use_handedness=False, use_rest=False, use_weather=False,
                        use_statcast=False, use_defense=False, use_bullpen_availability=False,
                        use_travel=False, use_h2h=False, use_home_road_splits=False):
    """Fetch (and cache, via api.py) all raw per-game inputs the model needs.
    Separated from compute_win_prob so the same fetched inputs can be reused
    across many candidate weight sets during tuning without re-hitting the API.
    """
    # Point-in-time stats: scoped to what had actually happened before
    # as_of_date, not the team/pitcher's full current-season totals. For a
    # live prediction (as_of_date == today) this is effectively the same as
    # season-to-date; for a backtest date in the past, this is what makes
    # the evaluation honest instead of "seeing the future" via today's stats.
    home_rs, home_ra, home_gp, home_era = api.get_team_run_stats_as_of(game["home_team_id"], season, as_of_date)
    away_rs, away_ra, away_gp, away_era = api.get_team_run_stats_as_of(game["away_team_id"], season, as_of_date)

    home_w, home_l = api.get_team_recent_form(game["home_team_id"], season, as_of_date)
    away_w, away_l = api.get_team_recent_form(game["away_team_id"], season, as_of_date)

    home_pitcher_era = api.get_pitcher_era_as_of(game["home_pitcher_id"], season, as_of_date)
    away_pitcher_era = api.get_pitcher_era_as_of(game["away_pitcher_id"], season, as_of_date)

    home_bullpen_era = away_bullpen_era = None
    if use_bullpen:
        home_bullpen_era = api.get_bullpen_era_l30(game["home_team_id"], as_of_date)
        away_bullpen_era = api.get_bullpen_era_l30(game["away_team_id"], as_of_date)

    home_lineup_ops = away_lineup_ops = home_roster_ops = away_roster_ops = None
    if use_lineups:
        home_lineup_ops, home_roster_ops = _lineup_strength(
            game["home_team_id"], season, game["game_pk"], True, use_lineups
        )
        away_lineup_ops, away_roster_ops = _lineup_strength(
            game["away_team_id"], season, game["game_pk"], False, use_lineups
        )

    park_run_factor, wind_mph = (100, None)
    if use_park_factors:
        park_run_factor, wind_mph = _park_run_factor(game["home_team_id"], game.get("game_datetime"), use_weather)

    home_ops_vs_away_hand = away_ops_vs_home_hand = None
    home_season_ops = away_season_ops = None
    if use_handedness:
        away_hand = api.get_pitcher_hand(game["away_pitcher_id"])
        home_hand = api.get_pitcher_hand(game["home_pitcher_id"])
        if away_hand:
            home_ops_vs_away_hand = api.get_team_ops_vs_hand(game["home_team_id"], away_hand, season)
        if home_hand:
            away_ops_vs_home_hand = api.get_team_ops_vs_hand(game["away_team_id"], home_hand, season)
        home_season_ops = api.get_team_season_roster_ops(game["home_team_id"], season)
        away_season_ops = api.get_team_season_roster_ops(game["away_team_id"], season)

    home_pitcher_rest_days = away_pitcher_rest_days = None
    if use_rest:
        home_pitcher_rest_days = api.get_pitcher_days_rest(game["home_pitcher_id"], as_of_date, season)
        away_pitcher_rest_days = api.get_pitcher_days_rest(game["away_pitcher_id"], as_of_date, season)

    home_xwoba, home_xwoba_against = _statcast_inputs(game["home_team_id"], as_of_date, use_statcast)
    away_xwoba, away_xwoba_against = _statcast_inputs(game["away_team_id"], as_of_date, use_statcast)

    home_oaa = away_oaa = None
    if use_defense:
        home_oaa = statcast.get_team_oaa_value(game["home_team_id"], season)
        away_oaa = statcast.get_team_oaa_value(game["away_team_id"], season)

    home_top_relievers = home_fatigued = away_top_relievers = away_fatigued = 0
    if use_bullpen_availability:
        home_top_relievers, home_fatigued = api.get_bullpen_availability_penalty_inputs(
            game["home_team_id"], as_of_date, season
        )
        away_top_relievers, away_fatigued = api.get_bullpen_availability_penalty_inputs(
            game["away_team_id"], as_of_date, season
        )

    home_travel_status = away_travel_status = "none"
    if use_travel:
        # Today's game is played at game["home_team_id"]'s park regardless of
        # which team we're asking about -- travel_status compares that
        # against where each team played yesterday.
        home_prev = api.get_previous_day_game_location(game["home_team_id"], as_of_date)
        away_prev = api.get_previous_day_game_location(game["away_team_id"], as_of_date)
        home_travel_status = travel_status(game["home_team_id"], home_prev)
        away_travel_status = travel_status(game["home_team_id"], away_prev)

    home_h2h_record = away_h2h_record = (0, 0)
    if use_h2h:
        home_h2h_record = api.get_head_to_head_record(game["home_team_id"], game["away_team_id"], season, as_of_date)
        away_h2h_record = api.get_head_to_head_record(game["away_team_id"], game["home_team_id"], season, as_of_date)

    home_home_split, home_overall_win_pct = (0, 0), None
    away_road_split, away_overall_win_pct = (0, 0), None
    if use_home_road_splits:
        (home_home_split, home_road_split) = api.get_home_road_splits(game["home_team_id"], season, as_of_date)
        home_total = sum(home_home_split) + sum(home_road_split)
        if home_total:
            home_overall_win_pct = (home_home_split[0] + home_road_split[0]) / home_total

        (away_home_split, away_road_split) = api.get_home_road_splits(game["away_team_id"], season, as_of_date)
        away_total = sum(away_home_split) + sum(away_road_split)
        if away_total:
            away_overall_win_pct = (away_home_split[0] + away_road_split[0]) / away_total

    return {
        "home_runs_scored": home_rs, "home_runs_allowed": home_ra, "home_era": home_era,
        "home_games_played": home_gp,
        "away_runs_scored": away_rs, "away_runs_allowed": away_ra, "away_era": away_era,
        "away_games_played": away_gp,
        "home_recent_form": (home_w, home_l), "away_recent_form": (away_w, away_l),
        "home_pitcher_era": home_pitcher_era, "away_pitcher_era": away_pitcher_era,
        "home_bullpen_era": home_bullpen_era, "away_bullpen_era": away_bullpen_era,
        "home_lineup_ops": home_lineup_ops, "away_lineup_ops": away_lineup_ops,
        "home_roster_ops": home_roster_ops, "away_roster_ops": away_roster_ops,
        "park_run_factor": park_run_factor, "wind_mph": wind_mph,
        "home_ops_vs_away_hand": home_ops_vs_away_hand, "away_ops_vs_home_hand": away_ops_vs_home_hand,
        "home_season_ops": home_season_ops, "away_season_ops": away_season_ops,
        "home_pitcher_rest_days": home_pitcher_rest_days, "away_pitcher_rest_days": away_pitcher_rest_days,
        "home_xwoba": home_xwoba, "away_xwoba": away_xwoba,
        "home_xwoba_against": home_xwoba_against, "away_xwoba_against": away_xwoba_against,
        "home_oaa": home_oaa, "away_oaa": away_oaa,
        "home_top_reliever_count": home_top_relievers, "home_fatigued_reliever_count": home_fatigued,
        "away_top_reliever_count": away_top_relievers, "away_fatigued_reliever_count": away_fatigued,
        "home_travel_status": home_travel_status, "away_travel_status": away_travel_status,
        "home_h2h_record": home_h2h_record, "away_h2h_record": away_h2h_record,
        "home_home_split": home_home_split, "home_overall_win_pct": home_overall_win_pct,
        "away_road_split": away_road_split, "away_overall_win_pct": away_overall_win_pct,
    }


def predict_game(game, season, as_of_date, use_bullpen=False, use_lineups=False,
                  use_park_factors=False, use_handedness=False, use_rest=False, use_weather=False,
                  use_statcast=False, use_defense=False, use_bullpen_availability=False, use_travel=False,
                  use_h2h=False, use_home_road_splits=False, weights: Weights = DEFAULT_WEIGHTS):
    """Compute a full prediction for a single game dict (as returned by
    api.get_schedule). Returns the game dict merged with all model outputs.
    """
    inputs = gather_game_inputs(
        game, season, as_of_date,
        use_bullpen=use_bullpen, use_lineups=use_lineups,
        use_park_factors=use_park_factors, use_handedness=use_handedness,
        use_rest=use_rest, use_weather=use_weather, use_statcast=use_statcast,
        use_defense=use_defense, use_bullpen_availability=use_bullpen_availability, use_travel=use_travel,
        use_h2h=use_h2h, use_home_road_splits=use_home_road_splits,
    )
    home_prob = compute_win_prob(inputs, weights)

    return {
        **game,
        **inputs,
        "home_win_prob": home_prob,
        "away_win_prob": 1 - home_prob,
    }
