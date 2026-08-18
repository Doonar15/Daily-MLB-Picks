#!/usr/bin/env python3
"""
Model-vs-Market Edge Backtest
==============================
A standalone diagnostic, separate from the live predictor -- it never
changes how predictions are made. It answers one question: when this
model disagrees with the real sportsbook line by a lot, is that
disagreement actually informative (the model found something the market
missed), or is it usually just the model being wrong?

Inspired by a sibling NCAA football project's backtest, which found that
bucketing games by |model vs. market disagreement| showed accuracy flat
at ~50% and realized moneyline ROI getting *worse*, not better, as the
disagreement grew -- i.e. big edges were dominated by the model being
wrong, not real value. This script runs the same style of analysis on
real MLB games using this project's own default-weight model and a free,
already-integrated market odds source.

Data: the same free MLB odds file used in this session's earlier
market-blend experiment (https://shanemcd.org, moneyline only here --
run line/totals aren't used). Not committed to the repo; point --odds-file
at a local copy (download it yourself, see the earlier session notes) or
this project's default lookup path.

Method per game: get this model's home win probability via predict_game
(fixed DEFAULT_WEIGHTS, the same default signal set predictor.py uses
live), get the market's devigged home win probability from the odds
file, take the model's picked side (whichever it favors) and compute
edge = model_prob_for_that_side - market_prob_for_that_side. Bucket by
|edge| and report, per bucket: how often the model's pick actually won,
and the realized ROI from flat-betting that side at the market's actual
price (not the devigged price -- that's what you'd really be paid).

Usage:
  python -m mlb_predictor.market_edge 2026-05-09 2026-08-06 --odds-file /path/to/mlb-2026-odds.xlsx
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .backtest import MAX_WORKERS, _iter_completed_games
from .model import DEFAULT_WEIGHTS, implied_prob_from_moneyline, predict_game

# Matches predictor.py's actual live defaults (parser.set_defaults calls) --
# this is what makes the comparison meaningful: the model being evaluated
# here is the one you actually run every day, not a stripped-down version.
LIVE_DEFAULT_FLAGS = dict(
    use_bullpen=False, use_lineups=True, use_park_factors=True, use_handedness=True,
    use_rest=False, use_statcast=True, use_defense=True, use_bullpen_availability=False,
    use_travel=True, use_h2h=True, use_home_road_splits=True,
)

EDGE_BUCKETS = [(0, 5), (5, 10), (10, 15), (15, 100)]


def load_market_odds(xlsx_path: str):
    """Return {(date, home_team, away_team): (home_ml, away_ml)}. Both None
    for a matchup with no line posted (future game, or a gap in the source).
    """
    df = pd.read_excel(xlsx_path, sheet_name="Betting Odds")
    odds = {}
    for _, row in df.iterrows():
        if row.get("Status") != "Final":
            continue
        home_ml = row.get("Home ML")
        away_ml = row.get("Away ML")
        if pd.isna(home_ml) or pd.isna(away_ml):
            continue
        try:
            home_ml, away_ml = float(home_ml), float(away_ml)
        except (TypeError, ValueError):
            # Some source rows use a literal "-" placeholder instead of
            # leaving the cell blank (seen in the 2025 season file) --
            # pd.isna() doesn't catch that, so fall back to skipping
            # anything that isn't actually numeric.
            continue
        key = (str(row["Date"]), row["Home"], row["Away"])
        odds[key] = (home_ml, away_ml)
    return odds


def devig_home_prob(home_ml, away_ml):
    home_raw = implied_prob_from_moneyline(home_ml)
    away_raw = implied_prob_from_moneyline(away_ml)
    total = home_raw + away_raw
    if total <= 0:
        return None
    return home_raw / total


def profit_per_100_wagered(ml, won):
    """Flat-bet realized profit/loss on a $100 wager at these American odds."""
    if not won:
        return -100.0
    return float(ml) if ml > 0 else 10000.0 / abs(ml)


def run(start_date: str, end_date: str, odds_file: str):
    print(f"Loading market odds from {odds_file}...")
    odds = load_market_odds(odds_file)
    print(f"{len(odds)} games with a posted moneyline in the odds file.")

    print(f"Scoring model predictions from {start_date} to {end_date}...")

    def _score_one(day_str, season, g):
        key = (day_str, g["home_team"], g["away_team"])
        line = odds.get(key)
        if line is None:
            return "no_odds"
        home_ml, away_ml = line
        market_home_prob = devig_home_prob(home_ml, away_ml)
        if market_home_prob is None:
            return "no_odds"

        try:
            pred = predict_game(g, season, day_str, weights=DEFAULT_WEIGHTS, **LIVE_DEFAULT_FLAGS)
        except Exception:
            return None

        home_won = g["home_score"] > g["away_score"]
        model_home_prob = pred["home_win_prob"]

        if model_home_prob >= 0.5:
            model_prob, market_prob, ml, won = model_home_prob, market_home_prob, home_ml, home_won
        else:
            model_prob, market_prob = 1 - model_home_prob, 1 - market_home_prob
            ml, won = away_ml, not home_won

        edge = (model_prob - market_prob) * 100
        return {"edge": edge, "won": won, "profit": profit_per_100_wagered(ml, won)}

    rows = []
    no_odds = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(_score_one, day_str, season, g)
            for day_str, season, g in _iter_completed_games(start_date, end_date)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result == "no_odds":
                no_odds += 1
            elif result is not None:
                rows.append(result)

    print(f"{len(rows)} games scored with both a model prediction and a market line "
          f"({no_odds} completed games had no matching odds row).")
    if not rows:
        print("Nothing to report.")
        return

    print()
    print("=" * 78)
    print(f"MODEL vs. MARKET EDGE BACKTEST: {start_date} to {end_date}")
    print("=" * 78)
    print(f"  {'|EDGE| BUCKET':<16} {'N':>6} {'AVG EDGE':>10} {'HIT RATE':>10} {'ROI':>10}")
    print(f"  {'-'*16} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

    for lo, hi in EDGE_BUCKETS:
        bucket = [r for r in rows if lo <= abs(r["edge"]) < hi]
        if not bucket:
            continue
        n = len(bucket)
        avg_edge = sum(abs(r["edge"]) for r in bucket) / n
        hit_rate = sum(1 for r in bucket if r["won"]) / n
        roi = sum(r["profit"] for r in bucket) / (n * 100)
        label = f"{lo}-{hi}pt" if hi < 100 else f"{lo}pt+"
        print(f"  {label:<16} {n:>6} {avg_edge:>9.1f}pt {hit_rate*100:>9.1f}% {roi*100:>+9.1f}%")

    overall_hit = sum(1 for r in rows if r["won"]) / len(rows)
    overall_roi = sum(r["profit"] for r in rows) / (len(rows) * 100)
    print()
    print(f"  Overall: {len(rows)} picks, {overall_hit*100:.1f}% hit rate, {overall_roi*100:+.1f}% ROI")
    print()
    print("  ROI = flat $100-per-game bet on the model's picked side at the market's actual")
    print("  price. A rising ROI/hit-rate as the edge bucket grows would mean big disagreements")
    print("  are informative; flat or falling means they're dominated by the model being wrong.")


def main():
    parser = argparse.ArgumentParser(description="Backtest whether model-vs-market disagreement is informative")
    parser.add_argument("start_date")
    parser.add_argument("end_date")
    parser.add_argument("--odds-file", required=True, help="Path to the market odds xlsx file")
    args = parser.parse_args()
    run(args.start_date, args.end_date, args.odds_file)


if __name__ == "__main__":
    main()
