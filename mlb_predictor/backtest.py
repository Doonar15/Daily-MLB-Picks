"""Backtest the model against historical MLB results using the same free
Stats API, and auto-tune the model's weights to minimize backtest error.

Point-in-time correctness: team run stats, pitcher ERA, and recent form are
all scoped to what had actually happened before the game being evaluated
(via api.get_team_run_stats_as_of / get_pitcher_era_as_of / a date-filtered
get_team_recent_form), not the team's current full-season totals. This is
what makes a backtest on, say, a June date reflect June's smaller sample
rather than quietly "seeing" the rest of the season.

Remaining caveats:
  - Bullpen ERA, confirmed lineups, and handedness splits (--bullpen,
    --lineups, --handedness) are NOT point-in-time -- they still use the
    team's current-season aggregate (or, for bullpen, a rolling 30-day
    window ending on the backtest date, which is point-in-time for bullpen
    specifically). Handedness splits and lineup OPS baselines use
    full-season data regardless of backtest date, since the free API's
    situational splits don't support byDateRange scoping.
  - Stats are date-granular, not time-of-day granular -- a game earlier in
    the day doesn't "come before" one later the same day.
  - Small sample sizes early in a season (or a short backtest window) make
    stats noisier and predictions less reliable, same as it would be live.
"""

import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, timedelta

from . import api
from .model import DEFAULT_WEIGHTS, Weights, compute_win_prob, gather_game_inputs, predict_game

# Point-in-time correctness means stats are cache-keyed per as_of_date, so a
# multi-day backtest can't reuse yesterday's cached team stats for today --
# it's genuinely a new API call per day per team. That makes a wide date
# range (a full season, multiple seasons) call-volume-bound rather than
# latency-bound: hundreds to thousands of small, independent, I/O-bound
# requests. A thread pool overlaps that network wait time instead of paying
# it serially. Kept modest (not "as many threads as possible") to stay
# reasonable against a free public API with no documented rate limit.
MAX_WORKERS = 8


def _brier_score(predicted_prob, actual_outcome):
    return (predicted_prob - actual_outcome) ** 2


def _iter_completed_games(start_date: str, end_date: str):
    """Yield (day_str, season, game) for every completed game (has a final
    score) in the date range. Schedule fetches happen serially since there's
    normally few dozen days at most, and each is already disk-cached; the
    real cost is the per-game stat fetches that follow, which callers
    parallelize themselves.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError("start_date must be <= end_date")

    day = start
    while day <= end:
        day_str = day.isoformat()
        season = day.year
        for g in api.get_schedule(day_str):
            # coded_status == "F" is the authoritative "game is actually
            # over" signal -- scores are non-null from the first pitch
            # onward (in-progress games included), so checking score
            # presence alone would wrongly treat a live game as final.
            if g.get("coded_status") != "F":
                continue
            if g.get("home_score") is not None and g.get("away_score") is not None:
                yield day_str, season, g
        day += timedelta(days=1)


def run_backtest(start_date: str, end_date: str, use_bullpen=False, use_lineups=False,
                  use_park_factors=False, use_handedness=False, use_rest=False, use_statcast=False,
                  use_defense=False, use_bullpen_availability=False, use_travel=False, use_h2h=False,
                  use_home_road_splits=False, weights: Weights = DEFAULT_WEIGHTS, market_odds=None):
    """market_odds, if given, is a {(home_team, away_team): (home_ml, away_ml)}
    lookup passed straight through to predict_game -- lets a backtest/
    calibration-backfill run reflect model.Weights.market_blend_weight for
    the (typically narrow) date range a historical odds file actually
    covers. Defaults to None (no market blend) so plain --backtest runs and
    the default calibration backfill range keep behaving exactly as before
    unless a caller explicitly supplies odds for a range known to have them.
    """
    def _predict_one(day_str, season, g):
        try:
            pred = predict_game(
                g, season, day_str, use_bullpen=use_bullpen, use_lineups=use_lineups,
                use_park_factors=use_park_factors, use_handedness=use_handedness, use_rest=use_rest,
                use_statcast=use_statcast, use_defense=use_defense,
                use_bullpen_availability=use_bullpen_availability, use_travel=use_travel, use_h2h=use_h2h,
                use_home_road_splits=use_home_road_splits, weights=weights, market_odds=market_odds,
            )
        except Exception:
            return None

        home_won = g["home_score"] > g["away_score"]
        return {
            "date": day_str,
            "matchup": f"{g['away_team']} @ {g['home_team']}",
            "home_win_prob": pred["home_win_prob"],
            "home_won": home_won,
            "correct_side": (pred["home_win_prob"] >= 0.5) == home_won,
            "brier": _brier_score(pred["home_win_prob"], 1.0 if home_won else 0.0),
        }

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(_predict_one, day_str, season, g)
            for day_str, season, g in _iter_completed_games(start_date, end_date)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    results.sort(key=lambda r: r["date"])
    return results


def summarize(results):
    if not results:
        return {
            "games": 0, "accuracy": None, "brier_score": None,
            "avg_confidence_when_correct": None, "avg_confidence_when_wrong": None,
        }

    n = len(results)
    correct = sum(1 for r in results if r["correct_side"])
    brier = sum(r["brier"] for r in results) / n

    def confidence(r):
        return r["home_win_prob"] if r["home_win_prob"] >= 0.5 else 1 - r["home_win_prob"]

    correct_confs = [confidence(r) for r in results if r["correct_side"]]
    wrong_confs = [confidence(r) for r in results if not r["correct_side"]]

    return {
        "games": n,
        "accuracy": correct / n,
        "brier_score": brier,
        "avg_confidence_when_correct": (sum(correct_confs) / len(correct_confs)) if correct_confs else None,
        "avg_confidence_when_wrong": (sum(wrong_confs) / len(wrong_confs)) if wrong_confs else None,
    }


def print_backtest_report(start_date, end_date, results):
    summary = summarize(results)
    print("=" * 70)
    print(f"BACKTEST REPORT: {start_date} to {end_date}")
    print("=" * 70)
    if summary["games"] == 0:
        print("No completed games found in this range.")
        return

    print(f"  Games evaluated:        {summary['games']}")
    print(f"  Accuracy (picked side): {summary['accuracy']*100:.1f}%")
    print(f"  Brier score:            {summary['brier_score']:.4f}  (0=perfect, 0.25=coin flip, lower is better)")
    if summary["avg_confidence_when_correct"] is not None:
        print(f"  Avg confidence (right): {summary['avg_confidence_when_correct']*100:.1f}%")
    if summary["avg_confidence_when_wrong"] is not None:
        print(f"  Avg confidence (wrong): {summary['avg_confidence_when_wrong']*100:.1f}%")
    print()
    print("  Note: core stats (runs, ERA, recent form) are point-in-time as of")
    print("  each backtest date. Bullpen/lineup/handedness stats (if enabled)")
    print("  are not -- see backtest.py module docstring for details.")
    print()


# ---------------------------------------------------------------------------
# Weight auto-tuning
#
# Tuning needs to score many candidate Weights against the same historical
# games. Re-fetching data per candidate would mean thousands of API calls, so
# instead we fetch each game's raw inputs ONCE (fetch_tuning_dataset) and then
# search over weights purely in-memory (evaluate_weights / tune_weights).
# ---------------------------------------------------------------------------

def fetch_tuning_dataset(start_date: str, end_date: str, use_bullpen=False, use_lineups=False,
                          use_park_factors=False, use_handedness=False, use_rest=False, use_statcast=False,
                          use_defense=False, use_bullpen_availability=False, use_travel=False, use_h2h=False,
                          use_home_road_splits=False):
    """Fetch raw model inputs + actual outcome for every completed game in
    the range, one time. Returns a list of (inputs, home_won) pairs suitable
    for repeated in-memory scoring against different Weights.

    Weather is intentionally never included here: Open-Meteo only serves a
    forecast window, so past-date lookups return None and the park factor
    falls back to its static value automatically -- there's no separate
    use_weather flag to pass through.
    """
    def _gather_one(day_str, season, g):
        try:
            inputs = gather_game_inputs(
                g, season, day_str, use_bullpen=use_bullpen, use_lineups=use_lineups,
                use_park_factors=use_park_factors, use_handedness=use_handedness, use_rest=use_rest,
                use_statcast=use_statcast, use_defense=use_defense,
                use_bullpen_availability=use_bullpen_availability, use_travel=use_travel, use_h2h=use_h2h,
                use_home_road_splits=use_home_road_splits,
            )
        except Exception:
            return None
        home_won = g["home_score"] > g["away_score"]
        return (inputs, home_won)

    dataset = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(_gather_one, day_str, season, g)
            for day_str, season, g in _iter_completed_games(start_date, end_date)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                dataset.append(result)

    return dataset


def evaluate_weights(dataset, weights: Weights):
    """Pure, no-network scoring of a candidate Weights against a pre-fetched
    dataset. Returns the mean Brier score (lower is better).
    """
    if not dataset:
        return None
    total = 0.0
    for inputs, home_won in dataset:
        prob = compute_win_prob(inputs, weights)
        total += _brier_score(prob, 1.0 if home_won else 0.0)
    return total / len(dataset)


# Candidate multipliers tried per weight dimension during search, relative to
# the current value. A coordinate-descent sweep (one dimension at a time,
# repeated for a couple of passes) is used instead of a full grid search --
# a full grid over 5 dimensions x 7 candidates would be 7**5 (~16800)
# evaluations; coordinate descent gets close in ~5 x 7 x passes.
_MULTIPLIERS = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
_TUNABLE_FIELDS = [
    "home_field_edge", "recent_form_weight",
    "pitcher_weight", "bullpen_weight", "lineup_weight",
    "handedness_weight", "rest_weight", "statcast_blend_weight",
    "defense_weight", "bullpen_availability_weight", "travel_weight",
    "h2h_weight", "home_road_split_weight", "shrinkage_factor", "sample_shrinkage_games",
]

# Absolute bounds per field, so a run of "improving" multiplier steps can't
# wander the weight to an implausible extreme (e.g. recent form dominating
# the season-long Pythagorean number). Centered around each default, wide
# enough for tuning to matter, narrow enough to stay sane.
_FIELD_BOUNDS = {
    "home_field_edge": (0.0, 0.09),
    "recent_form_weight": (0.0, 0.35),
    "pitcher_weight": (0.0, 0.25),
    "bullpen_weight": (0.0, 0.16),
    "lineup_weight": (0.0, 0.12),
    "handedness_weight": (0.0, 0.20),
    "rest_weight": (0.0, 0.08),
    "statcast_blend_weight": (0.0, 0.60),  # capped below 1.0 so xwOBA can't fully override real results
    "defense_weight": (0.0, 1.5),  # scales the fixed runs-per-out constant, not a probability weight itself
    "bullpen_availability_weight": (0.0, 0.08),
    "travel_weight": (0.0, 0.05),
    "h2h_weight": (0.0, 0.30),
    "home_road_split_weight": (0.0, 0.40),
    # Floor well above 0.0: a shrinkage_factor of 0 always predicts a coin
    # flip (Brier exactly 0.2500, the trivial baseline), so an unbounded
    # search on a noisy/small tuning window could otherwise "win" by erasing
    # all signal rather than genuinely compressing overconfidence.
    "shrinkage_factor": (0.15, 1.0),
    # Full-dataset sweep found K=10-15 minimized Brier overall (K past ~40
    # started giving back the gain by over-shrinking large, reliable
    # late-season samples) while the early-season-only slice kept improving
    # all the way to K=100 -- a single fixed value has to compromise across
    # very different sample sizes. Bounded well above that overall optimum
    # so the daily tuner can't drift into the region that only helps a
    # small early-season slice at the cost of everything else.
    "sample_shrinkage_games": (0.0, 40.0),
}


def tune_weights(dataset, base_weights: Weights = DEFAULT_WEIGHTS, passes: int = 2):
    """Coordinate-descent search over Weights to minimize Brier score on the
    given pre-fetched dataset. Returns (best_weights, best_brier_score).
    Falls back to base_weights if the dataset is empty.
    """
    best = base_weights
    best_score = evaluate_weights(dataset, best)
    if best_score is None:
        return base_weights, None

    for _ in range(passes):
        improved = False
        for field in _TUNABLE_FIELDS:
            current_value = getattr(best, field)
            lo, hi = _FIELD_BOUNDS[field]
            for mult in _MULTIPLIERS:
                candidate_value = max(lo, min(hi, round(current_value * mult, 4)))
                candidate = replace(best, **{field: candidate_value})
                score = evaluate_weights(dataset, candidate)
                if score is not None and score < best_score:
                    best, best_score = candidate, score
                    improved = True
        if not improved:
            break

    return best, best_score


def skill_trend(start_date: str, end_date: str, window_days: int = 30, step_days: int = 14, **flags):
    """Answer "is the model's current thin edge new or chronic?" by repeating
    the live auto-tuner's exact procedure (fresh coordinate descent from
    DEFAULT_WEIGHTS, scored via Brier) over a series of trailing windows
    stepped across a wide date range, instead of just the single trailing-30-
    day window get_tuned_weights() uses for live predictions.

    Each window is fetched independently (no reuse of get_tuned_weights()'s
    disk cache, which is only keyed for the single current window), so this
    is call-volume-heavy for a wide range -- expect roughly a window's worth
    of fetch_tuning_dataset() time per step, not an instant report.

    Returns a list of dicts (oldest window first): window_start, window_end,
    n_games, baseline_brier, tuned_brier.
    """
    rows = []
    window_end = date.fromisoformat(start_date) + timedelta(days=window_days - 1)
    final_end = date.fromisoformat(end_date)
    while window_end <= final_end:
        window_start = window_end - timedelta(days=window_days - 1)
        dataset = fetch_tuning_dataset(window_start.isoformat(), window_end.isoformat(), **flags)
        baseline = evaluate_weights(dataset, DEFAULT_WEIGHTS)
        _, tuned = tune_weights(dataset, DEFAULT_WEIGHTS)
        rows.append({
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "n_games": len(dataset),
            "baseline_brier": baseline,
            "tuned_brier": tuned,
        })
        window_end += timedelta(days=step_days)

    return rows


def print_skill_trend_report(rows):
    print("=" * 70)
    print("SKILL TREND  (rolling trailing-window Brier score, coin flip = 0.2500)")
    print("=" * 70)
    if not rows:
        print("  No windows evaluated.")
        print()
        return

    print(f"  {'window end':<12}{'n':>6}{'baseline':>12}{'tuned':>12}")
    for r in rows:
        if r["n_games"] == 0 or r["baseline_brier"] is None:
            print(f"  {r['window_end']:<12}{r['n_games']:>6}   no completed games in this window")
            continue
        print(f"  {r['window_end']:<12}{r['n_games']:>6}{r['baseline_brier']:>12.4f}{r['tuned_brier']:>12.4f}")

    print()
    print("  A window near 0.2500 has little real edge over a coin flip in that")
    print("  stretch, tuned or not. Compare early vs. recent windows to see")
    print("  whether current thin edge is a new dip or the season's norm.")
    print()


def blend_weights(raw: Weights, previous_smoothed: Weights, alpha: float) -> Weights:
    """Exponentially smooth a freshly-tuned Weights against the last logged
    smoothed value, so one noisy day's coordinate-descent result can't swing
    live weights by itself (e.g. h2h_weight/travel_weight roughly halving
    overnight between the Aug 13 and Aug 14 tuning runs). Only the tunable
    fields are blended; *_cap fields pass through from raw unchanged (tune_weights
    never touches them -- they're always DEFAULT_WEIGHTS' values).

    alpha is the weight given to the new raw value: alpha=1.0 is no smoothing
    (matches today's fresh-tune-every-day behavior), lower alpha damps faster-
    moving noise at the cost of slower response to a genuine regime shift.

    A convex blend (alpha in [0,1]) of two values already inside
    _FIELD_BOUNDS[field] is always inside that bound too, so no reclamping is
    mathematically required -- it's applied anyway as cheap insurance.
    """
    values = {}
    for field in _TUNABLE_FIELDS:
        blended = alpha * getattr(raw, field) + (1 - alpha) * getattr(previous_smoothed, field)
        lo, hi = _FIELD_BOUNDS[field]
        values[field] = max(lo, min(hi, round(blended, 4)))
    return replace(raw, **values)


def print_smoothing_summary(raw: Weights, smoothed: Weights, alpha: float):
    moved = []
    for field in _TUNABLE_FIELDS:
        raw_v, smoothed_v = getattr(raw, field), getattr(smoothed, field)
        if raw_v == 0:
            continue
        if abs(smoothed_v - raw_v) / abs(raw_v) > 0.01:
            moved.append((field, raw_v, smoothed_v))

    print("-" * 70)
    print(f"WEIGHT SMOOTHING  (alpha={alpha}, damping today's raw tune against prior smoothed history)")
    print("-" * 70)
    if not moved:
        print("  No tunable weight moved more than 1% after smoothing.")
    else:
        for field, raw_v, smoothed_v in moved:
            print(f"  {field:<24} raw tuned to {raw_v:.4f}  ->  using {smoothed_v:.4f}")
    print()


def print_tuning_summary(weights: Weights, score, baseline_score, games_used):
    print("-" * 70)
    print(f"AUTO-TUNED WEIGHTS  (from {games_used} recent completed games)")
    print("-" * 70)
    print(f"  home_field_edge:     {weights.home_field_edge:+.4f}")
    print(f"  recent_form_weight:  {weights.recent_form_weight:.4f}")
    print(f"  pitcher_weight:      {weights.pitcher_weight:.4f}")
    print(f"  bullpen_weight:      {weights.bullpen_weight:.4f}")
    print(f"  lineup_weight:       {weights.lineup_weight:.4f}")
    print(f"  handedness_weight:   {weights.handedness_weight:.4f}")
    print(f"  rest_weight:         {weights.rest_weight:.4f}")
    print(f"  statcast_blend_weight: {weights.statcast_blend_weight:.4f}")
    print(f"  defense_weight:      {weights.defense_weight:.4f}")
    print(f"  bullpen_availability_weight: {weights.bullpen_availability_weight:.4f}")
    print(f"  travel_weight:       {weights.travel_weight:.4f}")
    print(f"  h2h_weight:          {weights.h2h_weight:.4f}")
    print(f"  home_road_split_weight: {weights.home_road_split_weight:.4f}")
    if score is not None and baseline_score is not None:
        print(f"  Brier score: {score:.4f} tuned  vs.  {baseline_score:.4f} default weights")
    print()
