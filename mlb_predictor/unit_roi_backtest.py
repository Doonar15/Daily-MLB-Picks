#!/usr/bin/env python3
"""
Unit-Based ROI Backtest of Top Picks
=====================================
A standalone diagnostic, separate from the live predictor -- it never
changes how predictions are made. It answers a different question than
market_edge.py's model-vs-market analysis: if you had actually bet every
Top Pick this season (games clearing the live 58% raw-confidence bar) at
the closing moneyline, would you be up or down real money, and by how much?

Staking convention: flat $UNIT_SIZE-to-WIN per pick (not flat risk). A
favorite at -110 risks $5.50 to win $5; a dog at +150 risks $3.33 to win
$5. Every winning pick nets exactly +$UNIT_SIZE; only the risk (and thus
the loss on a miss) varies with the price. This mirrors how the user
already reasons about unit sizing for favorites, extended consistently to
dogs.

Data: same free MLB odds file as market_edge.py (https://shanemcd.org,
moneyline only). Not committed to the repo; defaults to this project's
local copy at mlb_predictor/data/mlb-2026-odds.xlsx.

Usage:
  python -m mlb_predictor.unit_roi_backtest 2026-03-25 2026-08-15
  python -m mlb_predictor.unit_roi_backtest 2026-03-25 2026-08-15 --unit-size 10 --min-confidence 60
"""
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import calibration
from .backtest import MAX_WORKERS, _iter_completed_games
from .market_edge import LIVE_DEFAULT_FLAGS, load_market_odds
from .model import DEFAULT_WEIGHTS, predict_game

DEFAULT_ODDS_FILE = Path(__file__).parent / "data" / "mlb-2026-odds.xlsx"

CONFIDENCE_TIERS = [(0.58, 0.60), (0.60, 0.70), (0.70, 1.001)]


def stake(ml: float, unit_size: float) -> float:
    """Dollars risked to win exactly unit_size on this moneyline."""
    if ml < 0:
        return unit_size * abs(ml) / 100.0
    return unit_size * 100.0 / ml


def unit_profit(ml: float, unit_size: float, won: bool) -> float:
    """Realized profit/loss on a flat-$unit_size-to-WIN bet at this price."""
    if won:
        return unit_size
    return -stake(ml, unit_size)


def expected_value(prob: float, ml: float, unit_size: float = 5.0) -> float:
    """Expected $ profit of a flat-$unit_size-to-win bet at this price, given
    a true win probability (ideally calibration-adjusted, not raw -- see
    calibration.ROI_PICK_CONFIDENCE's docstring for why raw overstates true
    win rate at 70%+ confidence). Positive means the price offers real value
    against that probability. Reuses stake(): EV = prob*unit_size (the win
    case) minus (1-prob)*stake (the loss case's risked amount).
    """
    return prob * unit_size - (1 - prob) * stake(ml, unit_size)


def score_all_games(start_date: str, end_date: str, odds_file: str, unit_size: float = 5.0):
    """Fetch model predictions + closing odds for every completed game ONCE,
    regardless of confidence level. Returns (rows, no_odds_count).

    Kept separate from run()/sweep() so a threshold/price-cutoff sweep can
    filter this same pre-fetched dataset in-memory many times over instead
    of re-running predict_game() (network + point-in-time stat fetches) once
    per candidate combination -- same fetch-once-search-many split backtest.py
    uses for weight tuning (fetch_tuning_dataset / evaluate_weights).
    """
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

        try:
            pred = predict_game(g, season, day_str, weights=DEFAULT_WEIGHTS, **LIVE_DEFAULT_FLAGS)
        except Exception:
            return None

        home_won = g["home_score"] > g["away_score"]
        model_home_prob = pred["home_win_prob"]

        if model_home_prob >= 0.5:
            side_prob, ml, won, is_favorite = model_home_prob, home_ml, home_won, home_ml < 0
        else:
            side_prob, ml, won, is_favorite = 1 - model_home_prob, away_ml, not home_won, away_ml < 0

        adjusted_confidence, adjusted_n = calibration.get_adjusted_confidence(side_prob)

        return {
            "date": day_str,
            "confidence": side_prob,
            "adjusted_confidence": adjusted_confidence,
            "adjusted_n": adjusted_n,
            "ml": ml,
            "won": won,
            "is_favorite": is_favorite,
            "risk": stake(ml, unit_size),
            "profit": unit_profit(ml, unit_size, won),
        }

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

    rows.sort(key=lambda r: r["date"])
    return rows, no_odds


def run(start_date: str, end_date: str, odds_file: str, unit_size: float = 5.0,
        min_confidence: float = calibration.MIN_PICK_CONFIDENCE):
    all_rows, no_odds = score_all_games(start_date, end_date, odds_file, unit_size)
    below_threshold = sum(1 for r in all_rows if r["confidence"] < min_confidence)
    rows = [r for r in all_rows if r["confidence"] >= min_confidence]

    print(f"{len(rows)} Top Picks scored (>= {min_confidence*100:.0f}% raw confidence; {below_threshold} completed "
          f"games had a market line but stayed below the bar, {no_odds} completed games had no matching odds row).")
    if not rows:
        print("Nothing to report.")
        return

    print()
    print("=" * 78)
    print(f"UNIT ROI BACKTEST: {start_date} to {end_date}  (${unit_size:.2f}-to-win per pick)")
    print("=" * 78)

    def _print_group(label, group):
        n = len(group)
        wins = sum(1 for r in group if r["won"])
        hit_rate = wins / n
        total_risk = sum(r["risk"] for r in group)
        net = sum(r["profit"] for r in group)
        roi = net / total_risk if total_risk else 0.0
        print(f"  {label:<24} {n:>4} picks   {wins}-{n-wins}   {hit_rate*100:>5.1f}% hit   "
              f"${total_risk:>8.2f} risked   ${net:>+9.2f} net   {roi*100:>+6.1f}% ROI")

    print(f"  {'BY CONFIDENCE TIER':<24}")
    for lo, hi in CONFIDENCE_TIERS:
        tier = [r for r in rows if lo <= r["confidence"] < hi]
        if not tier:
            continue
        label = f"{lo*100:.0f}-{hi*100:.0f}%" if hi < 1.001 else f"{lo*100:.0f}%+"
        _print_group(label, tier)

    print()
    print(f"  {'BY SIDE PICKED':<24}")
    favorites = [r for r in rows if r["is_favorite"]]
    dogs = [r for r in rows if not r["is_favorite"]]
    if favorites:
        _print_group("Favorites", favorites)
    if dogs:
        _print_group("Underdogs", dogs)

    print()
    print(f"  {'OVERALL':<24}")
    _print_group("All Top Picks", rows)
    print()
    print("  ROI = net profit / total dollars risked, flat-$-to-win staking at the closing")
    print("  moneyline. Losses on favorites cost more than a unit; losses on dogs cost less.")


# Grid searched by --sweep. A favorite_cutoff of None means "no price cap on
# favorites"; otherwise it excludes any favorite juicier than -cutoff (e.g.
# 150 excludes -175, -200, ... favorites but keeps every underdog untouched --
# the cutoff targets "don't lay a ton to win a little," which is a separate
# lever from raw confidence.
SWEEP_CONFIDENCE_GRID = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75]
SWEEP_FAVORITE_CUTOFFS = [None, 300, 250, 200, 175, 150, 130, 110]
MIN_SWEEP_SAMPLE = 20  # combos with fewer picks than this are noise, not signal


def _sweep_grid(all_rows, confidence_grid=SWEEP_CONFIDENCE_GRID, favorite_cutoffs=SWEEP_FAVORITE_CUTOFFS,
                 min_sample: int = MIN_SWEEP_SAMPLE, confidence_field: str = "confidence"):
    """Pure in-memory grid search over an already-scored rows list. Split out
    from sweep() so a multi-season pool (multi_sweep) can run the identical
    grid search over combined rows from several score_all_games() calls
    without re-fetching anything.

    confidence_field selects which per-row number the threshold is compared
    against: "confidence" (raw, the model's stated number) or
    "adjusted_confidence" (calibration.get_adjusted_confidence's real,
    historically-observed win rate for that raw bucket -- see
    calibration.ROI_PICK_CONFIDENCE's docstring for why raw overstates true
    win rate at 70%+). Rows whose bucket has no calibration data yet
    (adjusted_n == 0) are excluded from an adjusted-confidence sweep --
    there's nothing calibrated to threshold on for those.
    """
    if confidence_field == "adjusted_confidence":
        all_rows = [r for r in all_rows if r.get("adjusted_n", 0) > 0]

    results = []
    for threshold in confidence_grid:
        for cutoff in favorite_cutoffs:
            picks = [
                r for r in all_rows
                if r[confidence_field] >= threshold
                and (cutoff is None or not r["is_favorite"] or abs(r["ml"]) <= cutoff)
            ]
            n = len(picks)
            if n < min_sample:
                continue
            wins = sum(1 for r in picks if r["won"])
            total_risk = sum(r["risk"] for r in picks)
            net = sum(r["profit"] for r in picks)
            roi = net / total_risk if total_risk else 0.0
            results.append({
                "threshold": threshold, "favorite_cutoff": cutoff, "n": n, "wins": wins,
                "hit_rate": wins / n, "net": net, "roi": roi, "picks": picks,
                "confidence_field": confidence_field,
            })

    return results


def sweep(start_date: str, end_date: str, odds_file: str, unit_size: float = 5.0,
          confidence_grid=SWEEP_CONFIDENCE_GRID, favorite_cutoffs=SWEEP_FAVORITE_CUTOFFS,
          min_sample: int = MIN_SWEEP_SAMPLE, confidence_field: str = "confidence"):
    all_rows, no_odds = score_all_games(start_date, end_date, odds_file, unit_size)
    print(f"{len(all_rows)} completed games scored with a matching closing line "
          f"({no_odds} had no matching odds row).")
    if not all_rows:
        print("Nothing to sweep.")
        return []

    return _sweep_grid(all_rows, confidence_grid, favorite_cutoffs, min_sample, confidence_field)


def multi_sweep(specs, unit_size: float = 5.0, confidence_grid=SWEEP_CONFIDENCE_GRID,
                 favorite_cutoffs=SWEEP_FAVORITE_CUTOFFS, min_sample: int = MIN_SWEEP_SAMPLE,
                 confidence_field: str = "confidence"):
    """Same grid search as sweep(), pooled across several independently-scored
    date-range/odds-file pairs (e.g. multiple seasons) instead of one. Each
    spec is a dict with start_date/end_date/odds_file. Fetches each season
    once via score_all_games, then runs one grid search over the combined
    rows -- a cell only counts as a real signal here if it holds up once
    every season's games are mixed together, not just within one of them.
    """
    all_rows = []
    for spec in specs:
        rows, no_odds = score_all_games(spec["start_date"], spec["end_date"], spec["odds_file"], unit_size)
        print(f"  -> {len(rows)} games scored ({no_odds} without a matching line) from {spec['odds_file']}")
        all_rows.extend(rows)

    print(f"{len(all_rows)} completed games scored across {len(specs)} season(s) combined.")
    if not all_rows:
        print("Nothing to sweep.")
        return []

    return _sweep_grid(all_rows, confidence_grid, favorite_cutoffs, min_sample, confidence_field)


def bootstrap_cell(picks, iterations: int = 2000, seed: int = 0):
    """Resample a sweep cell's picks with replacement `iterations` times to
    see whether its point-estimate ROI holds up or is just noise from a small
    sample. Pure in-memory resampling of already-fetched (risk, profit) pairs
    -- no re-prediction, so it's cheap regardless of dataset size.

    Returns None for an empty cell. Otherwise a dict with the ROI
    distribution's mean, a 90% CI (5th/95th percentile), and the fraction of
    resamples that were net profitable -- a real edge should keep most of
    its CI on the positive side and a high profitable-fraction, not just a
    positive point estimate.
    """
    n = len(picks)
    if n == 0:
        return None

    rng = random.Random(seed)
    rois = []
    profitable = 0
    for _ in range(iterations):
        sample = [picks[rng.randrange(n)] for _ in range(n)]
        total_risk = sum(r["risk"] for r in sample)
        net = sum(r["profit"] for r in sample)
        rois.append(net / total_risk if total_risk else 0.0)
        if net > 0:
            profitable += 1

    rois.sort()
    lo = rois[int(0.05 * iterations)]
    hi = rois[min(int(0.95 * iterations), iterations - 1)]
    return {
        "roi_mean": sum(rois) / iterations,
        "roi_ci_low": lo,
        "roi_ci_high": hi,
        "p_profit": profitable / iterations,
    }


def add_bootstrap(results, iterations: int = 2000, seed: int = 0):
    """Mutate each sweep result in place, attaching its bootstrap_cell() stats
    under the "bootstrap" key."""
    for r in results:
        r["bootstrap"] = bootstrap_cell(r["picks"], iterations=iterations, seed=seed)
    return results


def print_sweep_report(results, unit_size: float = 5.0, min_sample: int = MIN_SWEEP_SAMPLE):
    has_bootstrap = bool(results) and results[0].get("bootstrap") is not None
    field = results[0]["confidence_field"] if results else "confidence"
    field_label = "ADJUSTED (calibrated)" if field == "adjusted_confidence" else "RAW"

    print()
    print("=" * 88)
    print(f"THRESHOLD / FAVORITE-JUICE SWEEP  (${unit_size:.2f}-to-win units, min {min_sample} picks per combo, "
          f"{field_label} confidence)")
    print("=" * 88)
    if not results:
        print("  No combination had enough picks to report.")
        return

    header = f"  {'CONFIDENCE >=':<14}{'FAV CUTOFF':<12}{'N':>6}{'RECORD':>10}{'HIT%':>8}{'NET $':>11}{'ROI':>9}"
    if has_bootstrap:
        header += f"{'90% ROI CI':>18}{'P(profit)':>11}"
    print(header)
    for r in sorted(results, key=lambda r: (r["threshold"], r["favorite_cutoff"] or 9999)):
        cutoff_label = f"-{r['favorite_cutoff']} or better" if r["favorite_cutoff"] else "none"
        record = f"{r['wins']}-{r['n']-r['wins']}"
        flag = "  <-- profitable" if r["net"] > 0 else ""
        line = (f"  {r['threshold']*100:>10.0f}%   {cutoff_label:<12}{r['n']:>6}{record:>10}"
                f"{r['hit_rate']*100:>7.1f}%{r['net']:>+10.2f} {r['roi']*100:>+7.1f}%")
        if has_bootstrap:
            b = r["bootstrap"]
            ci = f"[{b['roi_ci_low']*100:+.1f}%,{b['roi_ci_high']*100:+.1f}%]"
            line += f"{ci:>18}{b['p_profit']*100:>10.1f}%"
        print(line + flag)

    profitable = [r for r in results if r["net"] > 0]
    print()
    if not profitable:
        print("  No combination in this grid was net profitable.")
    else:
        best = max(profitable, key=lambda r: r["roi"])
        widest_sample = max(profitable, key=lambda r: r["n"])
        print(f"  {len(profitable)}/{len(results)} combinations were net profitable.")
        cutoff_label = f"-{best['favorite_cutoff']} or better" if best["favorite_cutoff"] else "no cap"
        print(f"  Best ROI:      >= {best['threshold']*100:.0f}% confidence, favorites {cutoff_label} "
              f"-> {best['n']} picks, {best['roi']*100:+.1f}% ROI, ${best['net']:+.2f} net")
        cutoff_label = f"-{widest_sample['favorite_cutoff']} or better" if widest_sample["favorite_cutoff"] else "no cap"
        print(f"  Largest sample: >= {widest_sample['threshold']*100:.0f}% confidence, favorites {cutoff_label} "
              f"-> {widest_sample['n']} picks, {widest_sample['roi']*100:+.1f}% ROI, ${widest_sample['net']:+.2f} net")
    print()
    print("  FAV CUTOFF excludes favorites juicier than -X (e.g. -150 or better means any favorite")
    print("  priced -151 or worse is skipped); underdogs are never excluded by this cutoff.")
    if has_bootstrap:
        print("  90% ROI CI / P(profit): 2000x resample-with-replacement of each cell's own picks. A cell")
        print("  whose CI straddles 0% (or whose P(profit) is barely above 50%) is likely noise, not edge --")
        print("  trust cells where the CI stays mostly positive and P(profit) is comfortably high.")


def main():
    parser = argparse.ArgumentParser(description="Backtest real-money ROI of Top Picks at closing moneylines")
    parser.add_argument("start_date", nargs="?", help="Required unless --season is given")
    parser.add_argument("end_date", nargs="?", help="Required unless --season is given")
    parser.add_argument("--odds-file", default=str(DEFAULT_ODDS_FILE), help="Path to the market odds xlsx file")
    parser.add_argument("--season", nargs=3, metavar=("START", "END", "ODDS_FILE"), action="append",
                         help="Add a season (date range + its own odds file) to pool together, repeatable. "
                              "With --sweep, runs one combined grid search across all given seasons' games "
                              "instead of the positional start_date/end_date/--odds-file.")
    parser.add_argument("--unit-size", type=float, default=5.0, help="Dollar amount one unit wins (default: 5.00)")
    parser.add_argument("--min-confidence", type=float, default=calibration.MIN_PICK_CONFIDENCE * 100,
                         help=f"Minimum raw confidence %% to count as a Top Pick (default: "
                              f"{calibration.MIN_PICK_CONFIDENCE*100:.0f})")
    parser.add_argument("--sweep", action="store_true",
                         help="Sweep confidence threshold x favorite-juice cutoff to find profitable combinations "
                              "instead of running a single report")
    parser.add_argument("--bootstrap", action="store_true",
                         help="With --sweep, add a bootstrap-resampled 90%% ROI confidence interval and "
                              "P(profit) to each cell, to distinguish real edges from small-sample noise")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000,
                         help="Resample count for --bootstrap (default: 2000)")
    parser.add_argument("--adjusted-confidence", action="store_true",
                         help="With --sweep, threshold on calibration-adjusted confidence (the real, "
                              "historically-observed win rate for that raw bucket) instead of raw confidence -- "
                              "raw overstates true win rate at 70%%+, see calibration.ROI_PICK_CONFIDENCE")
    args = parser.parse_args()
    confidence_field = "adjusted_confidence" if args.adjusted_confidence else "confidence"
    if args.season:
        if not args.sweep:
            parser.error("--season currently only supports --sweep (single-report `run` mode needs one date range)")
        specs = [{"start_date": s, "end_date": e, "odds_file": f} for s, e, f in args.season]
        results = multi_sweep(specs, unit_size=args.unit_size, confidence_field=confidence_field)
        if args.bootstrap:
            add_bootstrap(results, iterations=args.bootstrap_iterations)
        print_sweep_report(results, unit_size=args.unit_size)
    elif args.sweep:
        if not args.start_date or not args.end_date:
            parser.error("start_date and end_date are required unless --season is given")
        results = sweep(args.start_date, args.end_date, args.odds_file, unit_size=args.unit_size,
                         confidence_field=confidence_field)
        if args.bootstrap:
            add_bootstrap(results, iterations=args.bootstrap_iterations)
        print_sweep_report(results, unit_size=args.unit_size)
    else:
        if not args.start_date or not args.end_date:
            parser.error("start_date and end_date are required unless --season is given")
        run(args.start_date, args.end_date, args.odds_file, unit_size=args.unit_size,
            min_confidence=args.min_confidence / 100.0)


if __name__ == "__main__":
    main()
