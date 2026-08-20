#!/usr/bin/env python3
"""
MLB Game Outcome Predictor
===========================
Pulls live data from the free, public MLB Stats API (no API key required)
and estimates win probabilities for a given day's games.

Model inputs:
  1. Pythagorean win expectation (runs scored / runs allowed, season-to-date)
  2. Recent form (last 10 games) blended with the season-long Pythagorean number
  3. Starting pitcher adjustment (probable starter's ERA vs. team's rotation-average ERA)
  4. Bullpen adjustment (last-30-days relief ERA vs. team's rotation-average ERA) -- opt-in via --bullpen
  5. Confirmed-lineup strength (posted lineup's avg OPS vs. team's season roster avg OPS) -- ON by default,
     disable via --no-lineups. Contributes nothing (a graceful no-op, not an error) until MLB posts the
     lineup, typically a few hours before game time -- re-run closer to first pitch to pick it up.
  6. Home field advantage (~54% baseline in MLB, applied as a fixed adjustment)
  7. Park factor (static run-scoring index for the home park) -- ON by default, disable via --no-park-factors
  8. Wind amplification of the park factor -- opt-in via --weather (requires park factors enabled)
  9. Handedness matchup (team OPS vs. the opposing starter's throwing hand) -- ON by default, disable via --no-handedness
  10. Starter rest (short-rest penalty) -- opt-in via --rest (weaker, noisier signal than the others)
  11. Statcast xwOBA blend (expected-stat-based win expectation, blended with
      the raw-runs Pythagorean number) -- ON by default, disable via
      --no-statcast. Has no effect until --backfill-statcast has been run
      once; see below.

Recommended signals (park factors, handedness, lineups, Statcast) are enabled
by default. Rest and weather are weaker/noisier signals and stay opt-in.

Output:
  - Predicted home/away win probability for each game
  - Implied "fair" moneyline odds
  - If you pass in the sportsbook's actual moneyline odds, it will calculate
    the edge (your model's probability vs. the market's implied probability)
  - A "Top Picks" summary: every game at or above a confidence floor (not a
    fixed count -- see --min-confidence), showing both raw model confidence
    and a calibration-adjusted confidence for transparency

Every prediction is automatically saved to a local history log (see --no-record
to skip). Once games are final, run --grade-picks to check saved predictions
against actual results and see the model's real-time track record -- overall
and for Top Picks specifically. Unlike backtesting, this has zero look-ahead
risk: it's not a reconstruction, it's literally "what did we say, and were we
right." --grade-picks also regenerates a local HTML report
(mlb_predictor/.history/picks_report.html) showing every Top Pick's raw and
calibration-adjusted confidence alongside its actual result -- open it in any
browser, no server needed. Use --report to regenerate it without grading.

Confidence calibration answers a different question: across a large sample of
historical games, when the model says "60% confident," does that side actually
win about 60% of the time? Run --backfill-history ONCE to pull several
seasons of historical games (takes a few minutes); after that, the plain daily
command silently tops up just the day(s) since the last run (normally fast --
see --no-calibration-update to skip), and --calibration prints the report.

IMPORTANT / HONEST DISCLAIMER:
  This is an educational statistical tool, not a guaranteed-win system.
  No public model beats the closing line consistently over a large sample.
  Sportsbook lines (especially from sharp books) already price in most of
  what a model like this can see. Use this to build intuition and sanity-check
  bets, not as a black box you blindly follow. Bet only what you can afford
  to lose, and treat any edge estimate with skepticism until backtested.

By default, every prediction run first auto-tunes the model's weights against
a rolling backtest of the trailing 30 days (see --tune-window / --no-tune).

Usage:
  pip install requests --break-system-packages
  python -m mlb_predictor.predictor                       # today's games, auto-tuned, park factors/handedness/lineups/statcast on by default
  python -m mlb_predictor.predictor --date 2026-08-05      # specific date
  python -m mlb_predictor.predictor --odds                 # prompts for sportsbook odds per game, for edge calc
  python -m mlb_predictor.predictor --bullpen               # also include bullpen L30 ERA adjustment
  python -m mlb_predictor.predictor --rest                    # also include starter short-rest penalty
  python -m mlb_predictor.predictor --weather                  # also amplify park factor by forecast wind
  python -m mlb_predictor.predictor --no-park-factors            # disable park factor adjustment
  python -m mlb_predictor.predictor --no-handedness                # disable handedness adjustment
  python -m mlb_predictor.predictor --no-lineups                    # disable lineup strength adjustment
  python -m mlb_predictor.predictor --bullpen --rest --weather   # everything enabled (lineups/park/handedness/statcast already on)
  python -m mlb_predictor.predictor --no-tune                 # skip auto-tuning, use fixed default weights
  python -m mlb_predictor.predictor --tune-window 14           # tune against the trailing 14 days instead of 30
  python -m mlb_predictor.predictor --backtest 2026-07-01 2026-07-31   # backtest a date range (fixed weights)
  python -m mlb_predictor.predictor --skill-trend                       # rolling Brier score across the season -- is current skill new or chronic?
  python -m mlb_predictor.predictor --no-tune-smoothing                # disable EMA damping of the daily auto-tune (on by default)
  python -m mlb_predictor.predictor --grade-picks              # grade saved predictions, show track record, update the HTML ledger
  python -m mlb_predictor.predictor --report                     # regenerate the HTML ledger without grading first
  python -m mlb_predictor.predictor --no-record                 # predict without saving to the history log
  python -m mlb_predictor.predictor --backfill-history            # ONE-TIME: pull several seasons for calibration (a few minutes)
  python -m mlb_predictor.predictor --backfill-history 2024-03-01  # backfill from a specific start date instead of the default
  python -m mlb_predictor.predictor --calibration                   # show the confidence calibration report
  python -m mlb_predictor.predictor --no-calibration-update            # skip the daily calibration top-up
  python -m mlb_predictor.predictor --backfill-statcast                  # ONE-TIME: pull Statcast data for the xwOBA blend
  python -m mlb_predictor.predictor --no-statcast                          # disable the Statcast xwOBA blend
  python -m mlb_predictor.predictor --clear-cache            # wipe the local API cache
  python -m mlb_predictor.predictor --scoreboard                       # live-track today's Top Picks (inning/outs/score) until they're all Final
"""

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta

try:
    import requests
except ImportError:
    print("This script requires the 'requests' library.")
    print("Install it with: pip install requests --break-system-packages")
    sys.exit(1)

from . import api, backtest, cache, calibration, history, live_odds, report, statcast, tuning_log
from .model import (
    DEFAULT_WEIGHTS, MIN_H2H_GAMES, devig_home_prob, implied_prob_from_moneyline,
    moneyline_from_prob, predict_game,
)

TUNE_WINDOW_DAYS = 30
TUNING_CACHE_TTL = 4 * 60 * 60  # rebuild the tuning dataset at most every 4 hours
TUNE_SMOOTHING_ALPHA = 0.35  # weight given to today's raw tune when --tune-smoothing is on; see backtest.blend_weights

SCOREBOARD_REFRESH_SECONDS = 30
# Safety valve: a postponed/suspended game's live feed can report "Scheduled"
# indefinitely (verified against a real postponed game_pk that never
# transitioned away from Preview/Scheduled) -- without this, --scoreboard
# could run forever. 7h comfortably covers a single day's slate including
# delays and extra innings.
SCOREBOARD_MAX_RUNTIME_SECONDS = 7 * 60 * 60


def get_tuned_weights(as_of_date: str, window_days: int, use_bullpen: bool, use_lineups: bool,
                       use_park_factors: bool, use_handedness: bool, use_rest: bool, use_statcast: bool,
                       use_defense: bool, use_bullpen_availability: bool, use_travel: bool, use_h2h: bool,
                       use_home_road_splits: bool, tune_smoothing: bool = True,
                       smoothing_alpha: float = TUNE_SMOOTHING_ALPHA):
    """Backtest the trailing `window_days` days ending the day before
    as_of_date and search for weights that minimize Brier score over that
    window. The assembled dataset (not just the result) is disk-cached so
    repeated runs on the same day don't re-fetch from the API every time.

    Note: weather isn't included in tuning -- forecasts aren't available for
    past dates, so the tuning window's park factors are always the static
    (non-wind-amplified) values regardless of --weather.

    If tune_smoothing is set, the freshly-tuned weights are exponentially
    blended against the last logged smoothed value (see backtest.blend_weights)
    before being returned/used for predictions, so a single noisy window can't
    swing live weights by itself the way h2h_weight/travel_weight roughly
    halved overnight between the Aug 13 and Aug 14 tuning runs. The raw
    (unsmoothed) result is still logged alongside the smoothed one either way.
    """
    end = date.fromisoformat(as_of_date) - timedelta(days=1)
    start = end - timedelta(days=window_days - 1)
    start_str, end_str = start.isoformat(), end.isoformat()

    cache_key = (
        f"tuning_dataset|{start_str}|{end_str}|bullpen={use_bullpen}|lineups={use_lineups}"
        f"|park={use_park_factors}|hand={use_handedness}|rest={use_rest}|statcast={use_statcast}"
        f"|defense={use_defense}|bullpen_avail={use_bullpen_availability}|travel={use_travel}"
        f"|h2h={use_h2h}|home_road={use_home_road_splits}"
    )
    dataset = cache.get(cache_key, TUNING_CACHE_TTL)
    if dataset is None:
        dataset = backtest.fetch_tuning_dataset(
            start_str, end_str, use_bullpen=use_bullpen, use_lineups=use_lineups,
            use_park_factors=use_park_factors, use_handedness=use_handedness, use_rest=use_rest,
            use_statcast=use_statcast, use_defense=use_defense,
            use_bullpen_availability=use_bullpen_availability, use_travel=use_travel, use_h2h=use_h2h,
            use_home_road_splits=use_home_road_splits,
        )
        cache.set(cache_key, dataset)

    baseline_score = backtest.evaluate_weights(dataset, DEFAULT_WEIGHTS)
    weights, score = backtest.tune_weights(dataset, DEFAULT_WEIGHTS)
    backtest.print_tuning_summary(weights, score, baseline_score, len(dataset))

    effective_weights = weights
    if tune_smoothing:
        previous = tuning_log.get_smoothed_weights_before(as_of_date)
        if previous is not None:
            effective_weights = backtest.blend_weights(weights, previous, smoothing_alpha)
            backtest.print_smoothing_summary(weights, effective_weights, smoothing_alpha)
        else:
            print("(--tune-smoothing: no prior logged weights yet -- using this run's raw tuned weights unchanged.)\n")

    tuning_log.upsert({
        "as_of_date": as_of_date,
        "window_start": start_str,
        "window_end": end_str,
        "window_days": window_days,
        "n_games": len(dataset),
        "baseline_brier": baseline_score,
        "tuned_brier": score,
        "raw_tuned_weights": tuning_log.weights_to_dict(weights),
        "smoothed_weights": tuning_log.weights_to_dict(effective_weights),
        "smoothing_enabled": tune_smoothing,
        "smoothing_alpha": smoothing_alpha if tune_smoothing else None,
        "tuned_at": datetime.now().isoformat(timespec="seconds"),
    })

    return effective_weights


def print_prediction(pred, ask_odds=False):
    print("=" * 70)
    print(f"{pred['away_team']} @ {pred['home_team']}  ({pred['status']})")
    print("-" * 70)
    print(f"  {pred['away_team']:<25} SP: {pred['away_pitcher'] or 'TBD':<20} "
          f"ERA: {pred['away_pitcher_era'] if pred['away_pitcher_era'] else 'N/A'}")
    print(f"  {pred['home_team']:<25} SP: {pred['home_pitcher'] or 'TBD':<20} "
          f"ERA: {pred['home_pitcher_era'] if pred['home_pitcher_era'] else 'N/A'}")

    away_w, away_l = pred["away_recent_form"]
    home_w, home_l = pred["home_recent_form"]
    print(f"  Recent form (L10): {pred['away_team']} {away_w}-{away_l}  |  {pred['home_team']} {home_w}-{home_l}")

    if pred.get("away_bullpen_era") is not None or pred.get("home_bullpen_era") is not None:
        print(f"  Bullpen ERA (L30): {pred['away_team']} {pred['away_bullpen_era'] or 'N/A'}  |  "
              f"{pred['home_team']} {pred['home_bullpen_era'] or 'N/A'}")

    if pred.get("away_lineup_ops") is not None or pred.get("home_lineup_ops") is not None:
        away_ops = f"{pred['away_lineup_ops']:.3f}" if pred.get("away_lineup_ops") else "not posted"
        home_ops = f"{pred['home_lineup_ops']:.3f}" if pred.get("home_lineup_ops") else "not posted"
        print(f"  Lineup OPS:        {pred['away_team']} {away_ops}  |  {pred['home_team']} {home_ops}")

    if pred.get("park_run_factor") is not None and pred["park_run_factor"] != 100:
        wind_note = f", wind {pred['wind_mph']:.0f} mph" if pred.get("wind_mph") is not None else ""
        print(f"  Park factor ({pred['home_team']} park): {pred['park_run_factor']:.1f}{wind_note}")

    if pred.get("home_ops_vs_away_hand") is not None or pred.get("away_ops_vs_home_hand") is not None:
        home_hand_ops = f"{pred['home_ops_vs_away_hand']:.3f}" if pred.get("home_ops_vs_away_hand") is not None else "N/A"
        away_hand_ops = f"{pred['away_ops_vs_home_hand']:.3f}" if pred.get("away_ops_vs_home_hand") is not None else "N/A"
        print(f"  OPS vs. opposing SP hand: {pred['away_team']} {away_hand_ops}  |  {pred['home_team']} {home_hand_ops}")

    if pred.get("home_pitcher_rest_days") is not None or pred.get("away_pitcher_rest_days") is not None:
        def fmt_rest(days):
            if days is None:
                return "N/A"
            return "10+" if days > 10 else str(days)  # beyond ~10 days it's a roster move, not "rest"
        print(f"  Starter rest (days): {pred['away_team']} {fmt_rest(pred.get('away_pitcher_rest_days'))}  |  "
              f"{pred['home_team']} {fmt_rest(pred.get('home_pitcher_rest_days'))}")

    if pred.get("home_xwoba") is not None or pred.get("away_xwoba") is not None:
        def fmt_xwoba(v):
            return f"{v:.3f}" if v is not None else "N/A (building sample)"
        print(f"  xwOBA (batting):     {pred['away_team']} {fmt_xwoba(pred.get('away_xwoba'))}  |  "
              f"{pred['home_team']} {fmt_xwoba(pred.get('home_xwoba'))}")
        print(f"  xwOBA (vs. pitching): {pred['away_team']} {fmt_xwoba(pred.get('away_xwoba_against'))}  |  "
              f"{pred['home_team']} {fmt_xwoba(pred.get('home_xwoba_against'))}")

    if pred.get("home_oaa") is not None or pred.get("away_oaa") is not None:
        away_oaa = pred.get("away_oaa")
        home_oaa = pred.get("home_oaa")
        print(f"  Defense (OAA):       {pred['away_team']} {away_oaa if away_oaa is not None else 'N/A':>4}  |  "
              f"{pred['home_team']} {home_oaa if home_oaa is not None else 'N/A':>4}")

    if pred.get("home_top_reliever_count") or pred.get("away_top_reliever_count"):
        away_top, away_fat = pred.get("away_top_reliever_count", 0), pred.get("away_fatigued_reliever_count", 0)
        home_top, home_fat = pred.get("home_top_reliever_count", 0), pred.get("home_fatigued_reliever_count", 0)
        print(f"  Bullpen availability: {pred['away_team']} {away_fat}/{away_top} top relievers fatigued  |  "
              f"{pred['home_team']} {home_fat}/{home_top} top relievers fatigued")

    if pred.get("home_travel_status", "none") != "none" or pred.get("away_travel_status", "none") != "none":
        print(f"  Travel:              {pred['away_team']} {pred.get('away_travel_status', 'none')}  |  "
              f"{pred['home_team']} {pred.get('home_travel_status', 'none')}")

    if pred.get("home_h2h_record") or pred.get("away_h2h_record"):
        hw, hl = pred.get("home_h2h_record", (0, 0))
        aw, al = pred.get("away_h2h_record", (0, 0))
        if hw + hl >= MIN_H2H_GAMES:
            print(f"  Head-to-head this season: {pred['away_team']} {aw}-{al}  |  {pred['home_team']} {hw}-{hl}")

    if pred.get("home_home_split") or pred.get("away_road_split"):
        hw, hl = pred.get("home_home_split", (0, 0))
        aw, al = pred.get("away_road_split", (0, 0))
        print(f"  Venue split:         {pred['away_team']} {aw}-{al} (road)  |  {pred['home_team']} {hw}-{hl} (home)")

    print()
    home_ml = moneyline_from_prob(pred["home_win_prob"])
    away_ml = moneyline_from_prob(pred["away_win_prob"])
    print(f"  Model win probability:  {pred['away_team']} {pred['away_win_prob']*100:.1f}%  "
          f"|  {pred['home_team']} {pred['home_win_prob']*100:.1f}%")
    print(f"  Model 'fair' moneyline: {pred['away_team']} {away_ml:+d}  "
          f"|  {pred['home_team']} {home_ml:+d}")

    if ask_odds:
        try:
            away_book = input(f"  Enter sportsbook moneyline for {pred['away_team']} (blank to skip): ").strip()
            home_book = input(f"  Enter sportsbook moneyline for {pred['home_team']} (blank to skip): ").strip()
            if away_book:
                book_prob = implied_prob_from_moneyline(away_book)
                edge = (pred["away_win_prob"] - book_prob) * 100
                print(f"  -> {pred['away_team']} market implies {book_prob*100:.1f}% | "
                      f"model edge: {edge:+.1f} pts")
                pred["away_market_edge"] = edge
            if home_book:
                book_prob = implied_prob_from_moneyline(home_book)
                edge = (pred["home_win_prob"] - book_prob) * 100
                print(f"  -> {pred['home_team']} market implies {book_prob*100:.1f}% | "
                      f"model edge: {edge:+.1f} pts")
                pred["home_market_edge"] = edge
        except ValueError:
            print("  (skipped - invalid odds entered)")
    print()


def _market_favorite_pick(pred, live_odds_map):
    """The validated RECOMMENDED-pick rule (see calibration.ROI_PICK_FAVORITE_PRICE_CAP
    for the full backing: 189 picks, +15.2% ROI, bootstrap-confirmed) --
    pick whichever side the MARKET's own devigged probability favors, not
    the model's, and recommend it unless that side is a heavily-juiced
    favorite. The model's own prediction plays no role in this decision.

    Returns None if there's no live-odds match for this game. Otherwise a
    dict: side ("home"/"away"), team (that side's name), ml (that side's
    market price), recommended (bool).
    """
    line = live_odds_map.get((pred["home_team"], pred["away_team"])) if live_odds_map else None
    if line is None:
        return None
    home_ml, away_ml = line
    market_home_prob = devig_home_prob(home_ml, away_ml)
    if market_home_prob is None:
        return None

    if market_home_prob >= 0.5:
        side, team, ml = "home", pred["home_team"], home_ml
    else:
        side, team, ml = "away", pred["away_team"], away_ml

    is_juiced_favorite = ml < 0 and abs(ml) > calibration.ROI_PICK_FAVORITE_PRICE_CAP
    return {"side": side, "team": team, "ml": ml, "recommended": not is_juiced_favorite}


def print_top_picks(preds, min_confidence=calibration.MIN_PICK_CONFIDENCE, live_odds_map=None):
    """Two independent things, printed as two independent sections -- they
    no longer gate each other the way they used to:

    1. RECOMMENDED PICKS: every game (not just Top Picks -- ALL of today's
       scheduled games) where _market_favorite_pick finds a live line and
       recommends it. This scans independently of the model's own opinion,
       by design (see _market_favorite_pick's docstring) -- a game the model
       wouldn't even call a Top Pick, or where the model favors the other
       side entirely, can still show up here if the market's own favorite is
       cheap enough.
    2. TOP PICKS: every game at or above min_confidence in the model's own
       RAW confidence (post-shrinkage, post-market-blend -- see
       model.Weights), ranked highest first, for transparency into what the
       model itself thinks. Shown alongside its calibration-adjusted number
       (calibration.get_adjusted_confidence) but purely informational now --
       it does not drive any recommendation.

    Returns (top_pick_game_pks, roi_pick_game_pks, market_ml_by_game_pk,
    roi_pick_side_by_game_pk) so callers can tag/record all of the above.
    market_ml_by_game_pk holds the RECOMMENDED side's own market price for
    ROI picks (falls back to the model-favored side's price for plain Top
    Picks, for the informational "ML" display); roi_pick_side_by_game_pk
    holds the actual team name recommended, which can differ from the
    model's own predicted_winner -- see history.py's roi_pick_side field.
    """
    ranked = []
    for pred in preds:
        if pred["home_win_prob"] >= pred["away_win_prob"]:
            favorite, prob = pred["home_team"], pred["home_win_prob"]
            underdog = pred["away_team"]
        else:
            favorite, prob = pred["away_team"], pred["away_win_prob"]
            underdog = pred["home_team"]
        ranked.append((prob, favorite, underdog, pred))
    ranked.sort(key=lambda x: x[0], reverse=True)
    top = [r for r in ranked if r[0] >= min_confidence]

    market_picks = {}
    for pred in preds:
        decision = _market_favorite_pick(pred, live_odds_map)
        if decision is not None:
            market_picks[pred["game_pk"]] = decision
    recommended_preds = [p for p in preds if market_picks.get(p["game_pk"], {}).get("recommended")]

    print("#" * 70)
    print(f"RECOMMENDED PICKS  (market's own favorite, priced -{calibration.ROI_PICK_FAVORITE_PRICE_CAP} or better)")
    print("#" * 70)
    if not live_odds_map:
        print("  No live odds available today -- this rule needs a real market price for every game.")
        print()
    elif not recommended_preds:
        print("  No games cleared the recommendation bar today -- no recommended picks.")
        print()
    else:
        for i, pred in enumerate(recommended_preds, start=1):
            d = market_picks[pred["game_pk"]]
            underdog = pred["away_team"] if d["team"] == pred["home_team"] else pred["home_team"]
            print(f"  {i}. {d['team']} to beat {underdog}")
            print(f"     market ML {d['ml']:+d}  (live odds)")
            if i < len(recommended_preds):
                print()
        print()

    print("#" * 70)
    print(f"TOP PICKS  (>= {min_confidence*100:.0f}% raw model confidence -- informational, "
          "not a recommendation)")
    print("#" * 70)
    if not top:
        print(f"  No games reached the {min_confidence*100:.0f}% confidence floor today.")
        print()
    else:
        for i, (prob, favorite, underdog, pred) in enumerate(top, start=1):
            ml = moneyline_from_prob(prob)
            ml_str = f"{ml:+d}" if ml is not None else "N/A"
            adjusted, adj_n = calibration.get_adjusted_confidence(prob)
            adj_str = f"{adjusted*100:5.1f}%" if adj_n > 0 else "  N/A "
            print(f"  {i}. {favorite} to beat {underdog}")
            print(f"     raw {prob*100:5.1f}%   adj {adj_str}   fair ML {ml_str:>6}")
            if i < len(top):
                print()
        print()

    top_pick_game_pks = {pred["game_pk"] for _, _, _, pred in top}
    roi_pick_game_pks = {pred["game_pk"] for pred in recommended_preds}

    top_favorite_by_game_pk = {pred["game_pk"]: favorite for _, favorite, _, pred in top}
    market_ml_by_game_pk = {}
    roi_pick_side_by_game_pk = {}
    for pred in preds:
        game_pk = pred["game_pk"]
        decision = market_picks.get(game_pk)
        if game_pk in roi_pick_game_pks:
            market_ml_by_game_pk[game_pk] = decision["ml"]
            roi_pick_side_by_game_pk[game_pk] = decision["team"]
        elif game_pk in top_pick_game_pks:
            line = live_odds_map.get((pred["home_team"], pred["away_team"])) if live_odds_map else None
            favorite = top_favorite_by_game_pk[game_pk]
            market_ml_by_game_pk[game_pk] = (
                (line[0] if favorite == pred["home_team"] else line[1]) if line is not None else None
            )

    return top_pick_game_pks, roi_pick_game_pks, market_ml_by_game_pk, roi_pick_side_by_game_pk


def _ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _format_first_pitch(game_datetime):
    """Convert a schedule entry's UTC gameDate (e.g. '2026-08-07T23:10:00Z')
    into a local H:MM AM/PM string, or 'TBD' if it's missing/unparseable.
    """
    if not game_datetime:
        return "TBD"
    try:
        dt = datetime.fromisoformat(game_datetime.replace("Z", "+00:00")).astimezone()
        return dt.strftime("%-I:%M %p")
    except (ValueError, TypeError):
        return "TBD"


def _live_lead(pick, state):
    """Return the team currently ahead in a live game's score, or None if
    the game hasn't started (no score yet) or is tied.
    """
    home_score, away_score = state["home_score"], state["away_score"]
    if home_score is None or away_score is None or home_score == away_score:
        return None
    return pick["home_team"] if home_score > away_score else pick["away_team"]


def _render_scoreboard(picks, states, schedule_by_pk, game_date):
    os.system("cls" if os.name == "nt" else "clear")
    now_str = datetime.now().strftime("%-I:%M:%S %p")

    print("=" * 80)
    print(f"LIVE SCOREBOARD -- Top Picks for {game_date}        "
          f"updated {now_str}  (refresh: {SCOREBOARD_REFRESH_SECONDS}s)")
    print("=" * 80)
    print()

    final_count = 0
    for i, pick in enumerate(picks, start=1):
        state = states[pick["game_pk"]]
        print(f"  {i}. {pick['away_team']} @ {pick['home_team']}")
        pick_line = f"     Pick:   {pick['predicted_winner']} to win (raw {pick['predicted_prob']*100:.1f}%)"

        if state["error"]:
            print("     Status: live data unavailable this cycle (will retry)")
            print(pick_line)
        elif state["is_final"]:
            final_count += 1
            leader = _live_lead(pick, state)
            result = "WIN" if leader == pick["predicted_winner"] else "LOSS"
            print("     Status: FINAL")
            print(f"     Score:  {pick['away_team']} {state['away_score']} - {pick['home_team']} {state['home_score']}")
            print(f"{pick_line}      Result: {result}")
        elif state["inning"] is None:
            g = schedule_by_pk.get(pick["game_pk"])
            first_pitch = _format_first_pitch(g["game_datetime"]) if g else "TBD"
            print(f"     Status: Scheduled -- first pitch {first_pitch}")
            print(pick_line)
        else:
            print(f"     Status: {state['inning_state']} {_ordinal(state['inning'])}, {state['outs']} out")
            print(f"     Score:  {pick['away_team']} {state['away_score']} - {pick['home_team']} {state['home_score']}")
            leader = _live_lead(pick, state)
            if leader is None:
                live_check = "tied" if state["home_score"] == state["away_score"] else "not started"
            else:
                live_check = f"leading {leader} -> {'CORRECT' if leader == pick['predicted_winner'] else 'INCORRECT'}"
            print(f"{pick_line}      Live check: {live_check}")

        if i < len(picks):
            print()
            print("-" * 80)
            print()

    print()
    print("=" * 80)
    remaining = len(picks) - final_count
    if remaining:
        print(f"  {remaining} in progress / scheduled, {final_count} Final. Waiting for all games to finish...")
    else:
        print(f"  All {final_count} game(s) Final.")
    print("=" * 80)


def _print_scoreboard_summary(game_date, incomplete):
    history.grade_predictions()
    picks = history.get_top_picks_for_date(game_date)

    print()
    print("=" * 80)
    print(f"FINAL SUMMARY -- Top Picks for {game_date}")
    print("=" * 80)

    wins = losses = 0
    for pick in picks:
        label = f"  {pick['predicted_winner']} ({pick['predicted_prob']*100:.1f}%)"
        if pick["graded"]:
            if pick["correct"]:
                wins += 1
                result = "WIN"
            else:
                losses += 1
                result = "LOSS"
            print(f"{label:<40} final: {pick['actual_winner']} won   -> {result}")
        else:
            print(f"{label:<40} -- not final yet (postponed/suspended?)")

    print("-" * 80)
    graded_total = wins + losses
    if graded_total:
        print(f"  Top Picks today: {wins}-{losses} ({wins / graded_total * 100:.1f}%)")
    else:
        print("  No picks graded yet.")
    print("=" * 80)
    if incomplete:
        print("(Stopped before every game reached Final -- rerun --grade-picks later to catch stragglers.)")


def run_scoreboard(game_date):
    picks = history.get_top_picks_for_date(game_date)
    if not picks:
        print(f"No Top Picks recorded for {game_date}.")
        print("Run the predictor for this date (without --no-record) to generate Top Picks first, then retry --scoreboard.")
        return

    schedule_by_pk = {g["game_pk"]: g for g in api.get_schedule(game_date)}

    print(f"Tracking {len(picks)} Top Pick(s) for {game_date}. "
          f"Refreshing every {SCOREBOARD_REFRESH_SECONDS}s -- Ctrl+C to stop early.\n")
    time.sleep(2)  # let the message above actually be readable before the first clear

    start = time.time()
    stopped_early = False
    try:
        while True:
            states = {p["game_pk"]: api.get_live_game_state(p["game_pk"]) for p in picks}
            _render_scoreboard(picks, states, schedule_by_pk, game_date)

            if all(states[p["game_pk"]]["is_final"] for p in picks):
                break
            if time.time() - start > SCOREBOARD_MAX_RUNTIME_SECONDS:
                print(f"\nStopping after {SCOREBOARD_MAX_RUNTIME_SECONDS // 3600}h without every "
                      "game reaching Final -- one or more may be postponed/suspended. "
                      "Check statsapi/MLB.com directly for those.")
                stopped_early = True
                break
            time.sleep(SCOREBOARD_REFRESH_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        stopped_early = True

    _print_scoreboard_summary(game_date, incomplete=stopped_early)


def main():
    parser = argparse.ArgumentParser(description="MLB game outcome predictor")
    parser.add_argument("--date", default=date.today().isoformat(),
                         help="Date to predict, format YYYY-MM-DD (default: today)")
    parser.add_argument("--odds", action="store_true",
                         help="Prompt for sportsbook odds per game to calculate edge")
    parser.add_argument("--bullpen", action="store_true",
                         help="Include bullpen (last 30 days) ERA adjustment")
    parser.add_argument("--no-lineups", dest="lineups", action="store_false",
                         help="Disable confirmed starting-lineup strength adjustment (on by default; a no-op contribution if the lineup isn't posted yet, e.g. early-morning runs -- re-run closer to game time to pick it up)")
    parser.add_argument("--no-park-factors", dest="park_factors", action="store_false",
                         help="Disable static park run-factor adjustment (on by default)")
    parser.add_argument("--no-handedness", dest="handedness", action="store_false",
                         help="Disable batter-vs-opposing-starter-hand OPS adjustment (on by default)")
    parser.set_defaults(park_factors=True, handedness=True, lineups=True)
    parser.add_argument("--rest", action="store_true",
                         help="Include starting pitcher short-rest penalty (off by default: weak signal)")
    parser.add_argument("--weather", action="store_true",
                         help="Amplify the park factor using forecast wind speed (requires --park-factors; no effect on past dates)")
    parser.add_argument("--min-confidence", type=float, default=None, metavar="PCT",
                         help=f"Minimum raw model confidence (0-100) for a game to appear in Top Picks "
                              f"(default: {calibration.MIN_PICK_CONFIDENCE*100:.0f}, set from real calibration data -- see README)")
    parser.add_argument("--backtest", nargs=2, metavar=("START_DATE", "END_DATE"),
                         help="Backtest the model over a historical date range instead of predicting a single day")
    parser.add_argument("--skill-trend", action="store_true",
                         help="Print a rolling trailing-window Brier score report across the season (is current "
                              "skill new or chronic?), instead of predicting a day. Slow: one dataset fetch per window.")
    parser.add_argument("--skill-trend-start", type=str, default=None, metavar="START_DATE",
                         help="First window's start date for --skill-trend (default: March 1 of the current year)")
    parser.add_argument("--skill-trend-step", type=int, default=14, metavar="DAYS",
                         help="Days between successive windows for --skill-trend (default: 14)")
    parser.add_argument("--no-tune", action="store_true",
                         help="Skip auto-tuning and use the model's fixed default weights")
    parser.add_argument("--tune-window", type=int, default=TUNE_WINDOW_DAYS,
                         help=f"Days of trailing history to auto-tune against (default: {TUNE_WINDOW_DAYS})")
    parser.add_argument("--no-tune-smoothing", dest="tune_smoothing", action="store_false",
                         help="Disable exponential smoothing of the daily auto-tune (on by default as of 2026-08-15, "
                              "validated over a 15-day out-of-sample check: aggregate Brier +0.0008 vs. raw, "
                              "day-over-day weight volatility -63.8%%). Each day's freshly-tuned weights are normally "
                              "blended against the last logged smoothed weights in mlb_predictor/.history/tuning_log.jsonl "
                              "instead of used unchanged, damping overnight swings from one noisy tuning window (see "
                              "backtest.blend_weights and --tune-smoothing-alpha). No effect the first time it's ever "
                              "run (no prior logged weights to smooth against).")
    parser.set_defaults(tune_smoothing=True)
    parser.add_argument("--tune-smoothing-alpha", type=float, default=TUNE_SMOOTHING_ALPHA, metavar="ALPHA",
                         help=f"Weight (0-1) given to today's raw tuned weights when --tune-smoothing is on; lower "
                              f"damps noise more but responds slower to a real regime shift (default: {TUNE_SMOOTHING_ALPHA})")
    parser.add_argument("--clear-cache", action="store_true",
                         help="Clear the local API response cache and exit")
    parser.add_argument("--grade-picks", action="store_true",
                         help="Check recorded predictions against final scores, print the real-time track record, and regenerate the Top Picks HTML ledger, instead of predicting a day")
    parser.add_argument("--report", action="store_true",
                         help="Regenerate the Top Picks HTML ledger from current data without grading first, instead of predicting a day")
    parser.add_argument("--scoreboard", action="store_true",
                         help="Live-track today's Top Picks (inning/outs/score vs. the model's pick), refreshing every "
                              f"{SCOREBOARD_REFRESH_SECONDS}s until all are Final, instead of predicting a day. Uses --date to pick the day (default: today).")
    parser.add_argument("--no-record", action="store_true",
                         help="Don't save this run's predictions to the history log")
    parser.add_argument("--no-live-odds", action="store_true",
                         help="Skip fetching live market odds (ODDS_API_KEY env var) for RECOMMENDED-pick expected "
                              "value -- falls back to the static confidence bar. Also skips the 1-credit API call.")
    parser.add_argument("--backfill-history", nargs="?", const="DEFAULT", default=None, metavar="START_DATE",
                         help=f"One-time bulk historical pull for confidence calibration (default: ~{calibration.DEFAULT_BACKFILL_START_DAYS_AGO} days back). "
                              "Passing an earlier date goes back further but takes longer and hasn't been validated beyond a few months in one call -- "
                              "see the README for a note on this.")
    parser.add_argument("--calibration", action="store_true",
                         help="Show the confidence calibration report (does a stated confidence level match the actual win rate?), instead of predicting a day")
    parser.add_argument("--no-calibration-update", action="store_true",
                         help="Skip the automatic daily calibration top-up (normally a fast no-op or single-day catch-up)")
    parser.add_argument("--no-statcast", dest="statcast", action="store_false",
                         help="Disable the Statcast xwOBA blend (on by default; has no effect until --backfill-statcast has been run)")
    parser.add_argument("--backfill-statcast", nargs="?", const="DEFAULT", default=None, metavar="START_DATE",
                         help=f"One-time bulk historical pull of Statcast data for the xwOBA blend (default: ~{statcast.DEFAULT_BACKFILL_START_DAYS_AGO} days back). "
                              "Pitch-level data is heavier than the game-level pulls elsewhere in this tool -- expect this to take longer than --backfill-history "
                              "for the same date range. See the README's timing note before going back further than the default.")
    parser.add_argument("--no-defense", dest="defense", action="store_false",
                         help="Disable the OAA (defense) adjustment to runs allowed (on by default). NOT point-in-time -- see README.")
    parser.add_argument("--bullpen-availability", action="store_true",
                         help="Include a penalty when a team's top relievers recently threw a heavy-workload appearance (off by default: adds several API calls per game)")
    parser.add_argument("--no-travel", dest="travel", action="store_false",
                         help="Disable the getaway-day/timezone travel penalty (on by default)")
    parser.add_argument("--no-h2h", dest="h2h", action="store_false",
                         help="Disable the head-to-head this-season matchup adjustment (on by default)")
    parser.add_argument("--no-home-road-splits", dest="home_road_splits", action="store_false",
                         help="Disable the team-specific home/road performance adjustment (on by default)")
    parser.set_defaults(statcast=True, defense=True, travel=True, h2h=True, home_road_splits=True)
    args = parser.parse_args()

    if args.clear_cache:
        n = cache.clear()
        print(f"Cleared {n} cached entries.")
        return

    if args.grade_picks:
        newly_graded = history.grade_predictions()
        if newly_graded:
            print(f"Graded {newly_graded} newly-completed prediction(s).\n")
        history.print_track_record()
        report_path = report.generate()
        print(f"Top Picks ledger updated: {report_path}")
        print(f"Open it: file://{report_path.resolve()}")
        return

    if args.report:
        report_path = report.generate()
        print(f"Top Picks ledger updated: {report_path}")
        print(f"Open it: file://{report_path.resolve()}")
        return

    if args.scoreboard:
        run_scoreboard(args.date)
        return

    if args.backfill_history is not None:
        backfill_start = None if args.backfill_history == "DEFAULT" else args.backfill_history
        resolved_start = backfill_start or calibration.default_backfill_start()
        print(f"Backfilling calibration history from {resolved_start} through yesterday...")
        print("This reuses the backtest engine over a wide range -- expect a few minutes for the default range, "
              "longer (untested at scale) for an earlier start date. See the README's timing note.\n")
        n = calibration.backfill(start_date=backfill_start)
        print(f"Backfill complete: {n} games added.\n")
        calibration.bucket_report()
        return

    if args.calibration:
        if not args.no_calibration_update:
            newly_added = calibration.top_up()
            if newly_added:
                print(f"Calibration data updated: {newly_added} newly-completed game(s) added.\n")
        calibration.bucket_report()
        return

    if args.backfill_statcast is not None:
        backfill_start = None if args.backfill_statcast == "DEFAULT" else args.backfill_statcast
        resolved_start = backfill_start or statcast.default_backfill_start()
        print(f"Backfilling Statcast data from {resolved_start} through yesterday, one day at a time...")
        print("This pulls pitch-level data per day -- expect this to take longer than --backfill-history "
              "for the same range. See the README's timing note.\n")
        n = statcast.backfill(start_date=backfill_start)
        print(f"Backfill complete: {n} team-day records added.\n")
        return

    use_weather = args.weather and args.park_factors
    if args.weather and not args.park_factors:
        print("Note: --weather has no effect with --no-park-factors; ignoring.\n")

    if args.backtest:
        start_date, end_date = args.backtest
        print(f"\nRunning backtest from {start_date} to {end_date}...\n")
        results = backtest.run_backtest(
            start_date, end_date, use_bullpen=args.bullpen, use_lineups=args.lineups,
            use_park_factors=args.park_factors, use_handedness=args.handedness, use_rest=args.rest,
            use_statcast=args.statcast, use_defense=args.defense,
            use_bullpen_availability=args.bullpen_availability, use_travel=args.travel, use_h2h=args.h2h,
            use_home_road_splits=args.home_road_splits,
        )
        backtest.print_backtest_report(start_date, end_date, results)
        return

    if args.skill_trend:
        start = args.skill_trend_start or f"{date.today().year}-03-01"
        end = (date.today() - timedelta(days=1)).isoformat()
        print(f"\nRunning skill trend from {start} to {end} "
              f"(window={args.tune_window}d, step={args.skill_trend_step}d)...\n")
        rows = backtest.skill_trend(
            start, end, window_days=args.tune_window, step_days=args.skill_trend_step,
            **calibration.CALIBRATION_FLAGS,
        )
        backtest.print_skill_trend_report(rows)
        return

    if not args.no_calibration_update and calibration.get_watermark() is not None:
        calibration.top_up()  # normally a same-day no-op or a single-day catch-up; silent, fast

    if args.statcast and statcast.get_watermark() is not None:
        statcast.top_up()  # same pattern; no-ops if already current, silent otherwise

    print(f"\nFetching MLB schedule for {args.date}...\n")
    games = api.get_schedule(args.date)

    if not games:
        print("No games found for that date (or the MLB API is unreachable).")
        return

    weights = DEFAULT_WEIGHTS
    if not args.no_tune:
        print(f"Auto-tuning weights against the trailing {args.tune_window} days...\n")
        weights = get_tuned_weights(
            args.date, args.tune_window, args.bullpen, args.lineups,
            args.park_factors, args.handedness, args.rest, args.statcast,
            args.defense, args.bullpen_availability, args.travel, args.h2h, args.home_road_splits,
            tune_smoothing=args.tune_smoothing, smoothing_alpha=args.tune_smoothing_alpha,
        )

    live_odds_map = None
    if not args.no_live_odds:
        live_odds_map = live_odds.fetch_live_odds(target_date=args.date)
        if live_odds_map is None:
            print("Live odds unavailable (no ODDS_API_KEY set, or the request failed) -- "
                  "predictions won't include the market blend, and recommendations fall back "
                  "to the static confidence bar.")
        else:
            print(f"Live odds: {len(live_odds_map)} game(s) matched a market line.")
        print()

    season = int(args.date[:4])
    preds = []
    for g in games:
        try:
            pred = predict_game(
                g, season, args.date, use_bullpen=args.bullpen, use_lineups=args.lineups,
                use_park_factors=args.park_factors, use_handedness=args.handedness,
                use_rest=args.rest, use_weather=use_weather, use_statcast=args.statcast,
                use_defense=args.defense, use_bullpen_availability=args.bullpen_availability,
                use_travel=args.travel, use_h2h=args.h2h, use_home_road_splits=args.home_road_splits,
                weights=weights, market_odds=live_odds_map,
            )
            print_prediction(pred, ask_odds=args.odds)
            preds.append(pred)
        except requests.RequestException as e:
            print(f"Could not fetch data for {g['away_team']} @ {g['home_team']}: {e}")

    if preds:
        min_confidence = (args.min_confidence / 100.0) if args.min_confidence is not None else calibration.MIN_PICK_CONFIDENCE
        top_pick_game_pks, roi_pick_game_pks, market_ml_by_game_pk, roi_pick_side_by_game_pk = print_top_picks(
            preds, min_confidence=min_confidence, live_odds_map=live_odds_map)
        if not args.no_record:
            history.record_predictions(args.date, preds, top_pick_game_pks, roi_pick_game_pks,
                                        market_ml_by_game_pk, roi_pick_side_by_game_pk)


if __name__ == "__main__":
    main()
