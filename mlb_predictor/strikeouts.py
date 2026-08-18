#!/usr/bin/env python3
"""
Strikeout Projections
======================
A standalone tool, completely separate from the win-probability model in
model.py/predictor.py -- it shares this project's data layer (api.py, the
same free MLB Stats API used everywhere else here) but has its own CLI
entry point and touches none of the win-probability code paths.

Method: pull each starter's season game log (strikeouts and batters faced
per start, from api.get_pitcher_game_log), compute their season strikeout
rate (K per batter faced -- a cleaner rate than K/9 since start length
varies), and project it onto their typical workload (average batters
faced per start).

An earlier version also adjusted the projection by how strikeout-prone
today's specific opponent is relative to the average opponent faced this
season. A backtest of 2,203 starts (2026-05-09 to 2026-08-06) found that
adjustment changed accuracy by essentially nothing (MAE 1.80 vs. 1.82
strikeouts with it off) -- noise, not signal -- so it was removed rather
than kept as unused complexity.

Usage:
  python -m mlb_predictor.strikeouts                          # today's probable starters
  python -m mlb_predictor.strikeouts --date 2026-08-05         # a specific date
  python -m mlb_predictor.strikeouts --backtest 2026-05-09 2026-08-06   # validate against actual results
  python -m mlb_predictor.strikeouts --grade                   # grade yesterday's projections (win/loss)
  python -m mlb_predictor.strikeouts --grade 2026-08-09         # grade a specific past date
  python -m mlb_predictor.strikeouts --calibration 2026-05-09 2026-08-06   # predicted vs. actual hit rate, by projected K
"""
import argparse
import math
from datetime import date, timedelta

from . import api
from .backtest import _iter_completed_games

# A calibration backtest (--calibration 2026-05-09 2026-08-06, 2203 starts)
# found projections of 3-5 strikeouts were well-calibrated or better (actual
# hit rate met or beat the stated probability), but 6+ ran systematically
# overconfident (actual hit rate fell increasingly short of predicted as the
# projected number climbed). Flagged in the daily list as a caution marker,
# not filtered out -- the projection itself isn't wrong, just less trustworthy.
CAUTION_K_THRESHOLD = 6


def _poisson_pmf(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)


def _poisson_at_least(k, lam):
    """P(X >= k) for X ~ Poisson(lam). Strikeouts-per-start is a discrete
    count over a fixed number of batters faced, which is exactly the kind
    of data a Poisson distribution is meant for -- this is the standard,
    well-established way to turn a rate projection into a hit probability
    for a count stat, not a bespoke formula invented for this tool.
    """
    if k <= 0:
        return 1.0
    return 1.0 - sum(_poisson_pmf(i, lam) for i in range(k))


def project_strikeouts(pitcher_id: int, season: int, as_of_date: str = None):
    """Return a projection dict, or None if there's not enough data.

    as_of_date=None (the default) means "use everything logged so far" --
    the live, day-of use case, where the season log naturally only contains
    what's already happened. Passing a real date restricts the pitcher's
    game log to starts strictly before it -- this is what makes
    backtesting honest, since a historical start is only projected from
    what was actually knowable at the time.
    """
    log = api.get_pitcher_game_log(pitcher_id, season)
    if as_of_date is not None:
        log = [g for g in log if g["date"] < as_of_date]
    if not log:
        return None

    total_k = sum(g["strikeouts"] for g in log)
    total_bf = sum(g["batters_faced"] for g in log)
    if total_bf == 0:
        return None

    season_k_rate = total_k / total_bf
    avg_batters_faced = total_bf / len(log)
    projected_k_mean = season_k_rate * avg_batters_faced
    projected_strikeouts = round(projected_k_mean)

    return {
        "starts": len(log),
        "season_k_rate": season_k_rate,
        "avg_batters_faced": avg_batters_faced,
        "projected_strikeouts": projected_strikeouts,
        # Probability of AT LEAST the projected number -- matches how real
        # strikeout props are framed ("Over 6.5 K"), computed from the
        # unrounded projected mean so display-rounding doesn't distort it.
        "prob_at_least": _poisson_at_least(projected_strikeouts, projected_k_mean),
    }


def print_projections(game_date: str):
    games = api.get_schedule(game_date)
    if not games:
        print(f"No games found for {game_date}.")
        return

    season = date.fromisoformat(game_date).year
    starters = []
    for g in games:
        starters.append((g["home_pitcher"], g["home_pitcher_id"], g["home_team"], g["away_team"]))
        starters.append((g["away_pitcher"], g["away_pitcher_id"], g["away_team"], g["home_team"]))

    ranked = []
    unavailable = []
    for pitcher_name, pitcher_id, team, opponent in starters:
        if pitcher_id is None:
            unavailable.append(f"{team} vs {opponent}: probable starter not announced yet")
            continue
        proj = project_strikeouts(pitcher_id, season)
        if proj is None:
            unavailable.append(f"{pitcher_name} ({team}) vs {opponent}: no starts logged yet this season")
            continue
        ranked.append((pitcher_name, team, opponent, proj))

    ranked.sort(key=lambda r: (r[3]["projected_strikeouts"], r[3]["season_k_rate"]), reverse=True)

    print("#" * 100)
    print(f"STRIKEOUT PROJECTIONS -- {game_date}  (ranked by projected strikeouts; separate tool, not part of the win-probability model)")
    print("#" * 100)
    print(f"  {'#':>2}  {'K':>3}  {'PITCHER':<22} {'TEAM':<22} {'OPPONENT':<25} {'% AT LEAST K':>13} {'K RATE':>7} {'STARTS':>7}")
    print(f"  {'--':>2}  {'---':>3}  {'-'*22} {'-'*22} {'-'*25} {'-'*13} {'-'*7} {'-'*7}")
    for i, (pitcher_name, team, opponent, proj) in enumerate(ranked, start=1):
        k = proj["projected_strikeouts"]
        k_str = f"{k}!" if k >= CAUTION_K_THRESHOLD else str(k)
        print(f"  {i:>2}. {k_str:>3}  {pitcher_name:<22} {team:<22} {'vs ' + opponent:<25} "
              f"{proj['prob_at_least']*100:>12.1f}% {proj['season_k_rate']*100:>6.1f}% {proj['starts']:>7}")

    if ranked:
        print()
        print(f"  ! = projected {CAUTION_K_THRESHOLD}+ strikeouts -- calibration backtest found these run overconfident")
        print(f"      (actual hit rate fell short of the stated probability); 3-5 has been the most reliable range.")

    if unavailable:
        print()
        print("  Not yet projectable:")
        for line in unavailable:
            print(f"    - {line}")
    print()


def run_backtest(start_date: str, end_date: str):
    """Validate projection accuracy against actual results, honestly --
    point-in-time throughout (see project_strikeouts' as_of_date behavior).

    Collects every probable starter who appeared in the schedule across the
    date range, then walks each one's own season game log: for every start
    in [start_date, end_date] that has at least one earlier start in the
    same log to project from, compute a projection using only what was
    knowable strictly before that start, and compare it to the actual
    strikeouts already recorded for that same start (both come from the
    same api.get_pitcher_game_log call -- no separate "actual results"
    fetch needed).
    """
    print(f"Collecting probable starters from {start_date} to {end_date}...")
    pitcher_ids = set()
    for day_str, season, g in _iter_completed_games(start_date, end_date):
        for key in ("home_pitcher_id", "away_pitcher_id"):
            pid = g.get(key)
            if pid is not None:
                pitcher_ids.add((pid, season))

    print(f"{len(pitcher_ids)} unique starting pitchers to evaluate...")

    errors = []
    for pitcher_id, season in pitcher_ids:
        log = api.get_pitcher_game_log(pitcher_id, season)
        for idx, start in enumerate(log):
            if not (start_date <= start["date"] <= end_date):
                continue
            if idx == 0:
                continue  # no prior starts to project from
            proj = project_strikeouts(pitcher_id, season, as_of_date=start["date"])
            if proj is None:
                continue
            errors.append(proj["projected_strikeouts"] - start["strikeouts"])

    if not errors:
        print("No projectable starts found in this range (not enough prior-start history).")
        return

    n = len(errors)
    mae = sum(abs(e) for e in errors) / n
    bias = sum(errors) / n
    within_1 = sum(1 for e in errors if abs(e) <= 1) / n
    within_2 = sum(1 for e in errors if abs(e) <= 2) / n

    print()
    print("=" * 70)
    print(f"STRIKEOUT PROJECTION BACKTEST: {start_date} to {end_date}")
    print("=" * 70)
    print(f"  Starts evaluated:       {n}")
    print(f"  Mean absolute error:    {mae:.2f} strikeouts")
    print(f"  Mean signed error:      {bias:+.2f} strikeouts  (positive = over-projecting)")
    print(f"  Within 1 strikeout:     {within_1*100:.1f}%")
    print(f"  Within 2 strikeouts:    {within_2*100:.1f}%")


def run_calibration(start_date: str, end_date: str):
    """Bucket historical projections by their projected strikeout count and
    compare the model's own stated % AT LEAST K (averaged within that
    bucket) against what actually happened -- same calibration concept
    calibration.py uses for the win-probability model (does a stated
    confidence level match the real-world rate?), just bucketed by
    projected K value here since that's the natural grouping for a
    count-based projection rather than a percentile band.
    """
    print(f"Collecting probable starters from {start_date} to {end_date}...")
    pitcher_ids = set()
    for day_str, season, g in _iter_completed_games(start_date, end_date):
        for key in ("home_pitcher_id", "away_pitcher_id"):
            pid = g.get(key)
            if pid is not None:
                pitcher_ids.add((pid, season))

    print(f"{len(pitcher_ids)} unique starting pitchers to evaluate...")

    buckets = {}
    for pitcher_id, season in pitcher_ids:
        log = api.get_pitcher_game_log(pitcher_id, season)
        for idx, start in enumerate(log):
            if not (start_date <= start["date"] <= end_date):
                continue
            if idx == 0:
                continue
            proj = project_strikeouts(pitcher_id, season, as_of_date=start["date"])
            if proj is None:
                continue
            k = proj["projected_strikeouts"]
            hit = start["strikeouts"] >= k
            buckets.setdefault(k, []).append((proj["prob_at_least"], hit))

    if not buckets:
        print("No projectable starts found in this range.")
        return

    print()
    print("=" * 70)
    print(f"STRIKEOUT PROJECTION CALIBRATION: {start_date} to {end_date}")
    print("=" * 70)
    print(f"  {'PROJ K':>6} {'N':>6} {'PREDICTED':>10} {'ACTUAL':>8} {'GAP':>8}")
    print(f"  {'-'*6} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")
    total_n = 0
    for k in sorted(buckets):
        entries = buckets[k]
        n = len(entries)
        total_n += n
        avg_pred = sum(p for p, _ in entries) / n
        actual_rate = sum(1 for _, hit in entries if hit) / n
        gap = actual_rate - avg_pred
        print(f"  {k:>6} {n:>6} {avg_pred*100:>9.1f}% {actual_rate*100:>7.1f}% {gap*100:>+7.1f}pt")
    print()
    print(f"  {total_n} starts total across {len(buckets)} projected-K buckets.")
    print("  Gap = actual - predicted: positive means the model was underconfident at that")
    print("  K value (it hit more than expected), negative means overconfident. Small-N")
    print("  buckets will bounce around -- weight the N column before trusting a single row.")


def grade_day(game_date: str):
    """Show each probable starter's projection from that day graded against
    what they actually threw -- reconstructed the same point-in-time way
    run_backtest does (as_of_date=game_date), not from any saved log, since
    a past start's data never changes after the fact. A projection "wins"
    if the pitcher actually struck out AT LEAST the projected number,
    matching the % AT LEAST K column shown in the daily projections.
    """
    games = api.get_schedule(game_date)
    if not games:
        print(f"No games found for {game_date}.")
        return

    season = date.fromisoformat(game_date).year
    starters = []
    for g in games:
        starters.append((g["home_pitcher"], g["home_pitcher_id"], g["home_team"], g["away_team"]))
        starters.append((g["away_pitcher"], g["away_pitcher_id"], g["away_team"], g["home_team"]))

    rows = []
    for pitcher_name, pitcher_id, team, opponent in starters:
        if pitcher_id is None:
            continue
        proj = project_strikeouts(pitcher_id, season, as_of_date=game_date)
        if proj is None:
            continue
        log = api.get_pitcher_game_log(pitcher_id, season)
        actual_entry = next((g for g in log if g["date"] == game_date), None)
        rows.append((pitcher_name, team, opponent, proj, actual_entry))

    if not rows:
        print(f"No gradable projections for {game_date} (no probable starters with prior-start history that day).")
        return

    rows.sort(key=lambda r: (r[3]["projected_strikeouts"], r[3]["season_k_rate"]), reverse=True)

    print("#" * 90)
    print(f"STRIKEOUT PROJECTIONS -- GRADED FOR {game_date}  (ranked by projected strikeouts)")
    print("#" * 90)
    print(f"  {'PITCHER':<22} {'TEAM':<22} {'OPPONENT':<25} {'PROJ':>5} {'% AT LEAST K':>13} {'ACTUAL':>7}  RESULT")
    print(f"  {'-'*22} {'-'*22} {'-'*25} {'-'*5} {'-'*13} {'-'*7}  {'-'*6}")

    wins = losses = pending = 0
    for pitcher_name, team, opponent, proj, actual_entry in rows:
        projected = proj["projected_strikeouts"]
        prob_str = f"{proj['prob_at_least']*100:.1f}%"
        if actual_entry is None:
            print(f"  {pitcher_name:<22} {team:<22} {'vs ' + opponent:<25} {projected:>5} {prob_str:>13} {'--':>7}  PENDING")
            pending += 1
            continue
        actual = actual_entry["strikeouts"]
        hit = actual >= projected
        wins += int(hit)
        losses += int(not hit)
        print(f"  {pitcher_name:<22} {team:<22} {'vs ' + opponent:<25} {projected:>5} {prob_str:>13} {actual:>7}  {'WIN' if hit else 'LOSS'}")

    print()
    graded = wins + losses
    if graded:
        pending_note = f"  ({pending} pending)" if pending else ""
        print(f"  Record: {wins}-{losses} ({wins / graded * 100:.1f}%){pending_note}")
    else:
        print(f"  All {pending} projection(s) still pending -- try again once those games are final.")


def main():
    parser = argparse.ArgumentParser(description="Standalone strikeout projections for probable starters")
    parser.add_argument("--date", default=date.today().isoformat(),
                         help="Date to project, format YYYY-MM-DD (default: today)")
    parser.add_argument("--backtest", nargs=2, metavar=("START_DATE", "END_DATE"),
                         help="Validate projections against actual results over a historical date range, instead of projecting a single day")
    parser.add_argument("--grade", nargs="?", const="YESTERDAY", default=None, metavar="DATE",
                         help="Grade a past day's projections against actual strikeouts (default: yesterday), instead of projecting a day or backtesting a range")
    parser.add_argument("--calibration", nargs=2, metavar=("START_DATE", "END_DATE"),
                         help="Bucket historical projections by projected K value and compare the model's stated "
                              "% AT LEAST K against the actual hit rate in each bucket, instead of projecting a day")
    args = parser.parse_args()

    if args.backtest:
        run_backtest(args.backtest[0], args.backtest[1])
        return

    if args.calibration:
        run_calibration(args.calibration[0], args.calibration[1])
        return

    if args.grade is not None:
        game_date = (date.today() - timedelta(days=1)).isoformat() if args.grade == "YESTERDAY" else args.grade
        grade_day(game_date)
        return

    print_projections(args.date)


if __name__ == "__main__":
    main()
