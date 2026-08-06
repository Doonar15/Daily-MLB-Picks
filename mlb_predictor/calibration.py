"""Confidence calibration: across a large sample of historical games, when
the model says "60% confident," is it actually right about 60% of the time?

This is different from both backtesting (a one-off report over a date range,
not persisted) and history.py's real-time track record (predictions made in
the moment, graded later). Calibration data is backtest-RECONSTRUCTED (same
look-ahead caveats as any backtest -- see backtest.py's docstring) but
accumulated permanently and incrementally, specifically to answer "is a
stated confidence level trustworthy," which needs a much bigger sample than
the real-time log will have for a long time.

Storage is a local JSON-lines file, one graded game per line, plus a
watermark date recording the most recent date already covered -- so a daily
top-up only ever has to backtest the small gap since the last run (normally
zero or one day) instead of re-scanning everything.
"""

import json
from datetime import date, timedelta
from pathlib import Path

from . import backtest

CALIBRATION_DIR = Path(__file__).parent / ".history"
CALIBRATION_FILE = CALIBRATION_DIR / "calibration.jsonl"
WATERMARK_FILE = CALIBRATION_DIR / "calibration_watermark.json"

# A measured 90-day/1,167-game backtest took ~3 minutes wall-clock with the
# thread pool. That doesn't hold up linearly at much larger scale -- an
# ~11-month test during development was still running past 30 minutes and was
# stopped rather than trusted blindly (likely cumulative request volume
# against a free API with no documented rate limit). Default to a safer ~4
# month window; pass an explicit earlier START_DATE to go back further,
# understanding it may take a long time and hasn't been validated at that
# scale -- consider running multiple smaller backfills (one per season/chunk)
# instead of one huge multi-season call.
DEFAULT_BACKFILL_START_DAYS_AGO = 120

# The calibration dataset should reflect what the plain daily command
# actually predicts with, so it stays a meaningful answer to "is today's
# default confidence level trustworthy." Matches predictor.py's defaults.
CALIBRATION_FLAGS = dict(
    use_park_factors=True, use_handedness=True, use_lineups=True, use_statcast=True,
    use_defense=True, use_travel=True, use_h2h=True, use_home_road_splits=True,
    use_bullpen=False, use_rest=False, use_bullpen_availability=False,
)

BUCKET_EDGES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.001]  # 1.001 so 100% sorts into the last bucket

# Minimum raw (model-predicted) confidence for a game to show up in Top
# Picks. Set from real calibration data: the 50-60% bucket showed only a
# -1.8pt gap (n=1021) -- close to honest -- while everything from 60% up
# showed real, larger overconfidence gaps that don't keep shrinking as
# stated confidence rises (60-70%: -5.8pt n=428, 70-80%: -11.3pt n=95).
# 58% sits inside the well-calibrated part of the 50-60% band rather than at
# its noisier low end. Threshold is intentionally on RAW confidence, not the
# adjusted number -- the whole point of showing both is that a raw 66% pick
# might turn out to have an adjusted confidence below this floor, and that
# gap is exactly what's worth seeing, not hiding by thresholding post-adjustment.
MIN_PICK_CONFIDENCE = 0.58

# Buckets used for Top Picks display/threshold purposes specifically (as
# opposed to bucket_report's fixed 10-point report buckets above). Only
# 50-60%/60-70% are active given the sample sizes seen so far -- 70%+ bands
# have too few games (n=95, n=9, n=2) to trust their measured gap, and
# widening the top band would just borrow the (also noisy) 70-80% gap for a
# high-stakes adjustment. Kept here, commented out, so re-enabling any tier
# is a one-line change once its sample size is large enough to trust --
# check bucket_report() output before flipping one on.
ADJUSTMENT_BUCKETS = [
    (0.50, 0.60),
    (0.60, 0.70),
    # (0.70, 0.80),  # n=95 as of last check -- re-enable once this has a few hundred+ games
    # (0.80, 0.90),  # n=9 as of last check -- far too small to trust
    # (0.90, 1.001), # n=2 as of last check -- pure noise right now
]


def _read_all():
    if not CALIBRATION_FILE.exists():
        return []
    records = []
    for line in CALIBRATION_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _append_all(new_records):
    if not new_records:
        return
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    with CALIBRATION_FILE.open("a") as f:
        for r in new_records:
            f.write(json.dumps(r) + "\n")


def get_watermark():
    """Return the last date already covered by the calibration dataset, or
    None if no backfill has run yet.
    """
    if not WATERMARK_FILE.exists():
        return None
    try:
        return json.loads(WATERMARK_FILE.read_text()).get("last_covered_date")
    except json.JSONDecodeError:
        return None


def _set_watermark(covered_through_date: str):
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    WATERMARK_FILE.write_text(json.dumps({"last_covered_date": covered_through_date}))


def _results_to_records(results):
    return [
        {
            "date": r["date"],
            "confidence": r["home_win_prob"] if r["home_win_prob"] >= 0.5 else 1 - r["home_win_prob"],
            "correct": r["correct_side"],
        }
        for r in results
    ]


def default_backfill_start():
    return (date.today() - timedelta(days=DEFAULT_BACKFILL_START_DAYS_AGO)).isoformat()


def backfill(start_date: str = None, end_date: str = None):
    """One-time (or re-run-to-extend) bulk historical pull. Backtests
    start_date (default: ~4 months ago) through end_date (default: yesterday)
    and saves every graded game to the calibration log, then sets the
    watermark. Safe to call again with an earlier start_date than before --
    results aren't deduped against the watermark here, so only call this for
    a genuinely new range (use top_up() for the routine "since last time"
    case).
    """
    start_date = start_date or default_backfill_start()
    end_date = end_date or (date.today() - timedelta(days=1)).isoformat()
    if start_date > end_date:
        return 0

    results = backtest.run_backtest(start_date, end_date, **CALIBRATION_FLAGS)
    records = _results_to_records(results)
    _append_all(records)

    existing_watermark = get_watermark()
    if existing_watermark is None or end_date > existing_watermark:
        _set_watermark(end_date)

    return len(records)


def top_up():
    """Fetch only the gap between the watermark and yesterday. Normally
    covers zero or one day (whatever's accumulated since the last run), so
    this is fast even though it reuses the same backtest machinery as a full
    backfill. Returns the number of newly-added games. No-op (and no API
    calls) if there's no backfill yet or nothing new to cover.
    """
    watermark = get_watermark()
    if watermark is None:
        return 0  # no backfill yet; --backfill-history hasn't been run

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    gap_start = (date.fromisoformat(watermark) + timedelta(days=1)).isoformat()
    if gap_start > yesterday:
        return 0  # already fully up to date

    return backfill(start_date=gap_start, end_date=yesterday)


def _bucket_actual_rate(records, lo, hi):
    """Return (actual_win_rate, n) for games in records with lo <= confidence
    < hi, or (None, 0) if there's no data in that range.
    """
    in_bucket = [r for r in records if lo <= r["confidence"] < hi]
    if not in_bucket:
        return None, 0
    n = len(in_bucket)
    actual_win_rate = sum(1 for r in in_bucket if r["correct"]) / n
    return actual_win_rate, n


def get_adjusted_confidence(raw_confidence):
    """Given a raw (model-predicted) confidence for the favored side, return
    (adjusted_confidence, bucket_n) using ADJUSTMENT_BUCKETS -- i.e. "what
    has this stated confidence level actually meant historically." Falls
    back to the raw confidence unchanged (with bucket_n=0) if raw_confidence
    falls outside the active buckets (e.g. below 50%, or in a currently
    disabled 70%+ tier) or there's no calibration data yet.
    """
    records = _read_all()
    for lo, hi in ADJUSTMENT_BUCKETS:
        if lo <= raw_confidence < hi:
            actual_rate, n = _bucket_actual_rate(records, lo, hi)
            if actual_rate is None:
                return raw_confidence, 0
            return actual_rate, n
    return raw_confidence, 0


def bucket_report():
    records = _read_all()
    watermark = get_watermark()

    print("=" * 70)
    print("CONFIDENCE CALIBRATION  (model-reconstructed predictions vs. actual results)")
    print("=" * 70)
    if not records:
        print("  No calibration data yet. Run --backfill-history to build it.")
        print()
        return

    buckets = []
    for i in range(len(BUCKET_EDGES) - 1):
        lo, hi = BUCKET_EDGES[i], BUCKET_EDGES[i + 1]
        in_bucket = [r for r in records if lo <= r["confidence"] < hi]
        buckets.append((lo, min(hi, 1.0), in_bucket))

    for lo, hi, in_bucket in buckets:
        label = f"{lo*100:.0f}-{hi*100:.0f}%"
        if not in_bucket:
            print(f"  {label:<10} no games in this range yet")
            continue
        n = len(in_bucket)
        actual_win_rate = sum(1 for r in in_bucket if r["correct"]) / n
        avg_predicted = sum(r["confidence"] for r in in_bucket) / n
        gap = (actual_win_rate - avg_predicted) * 100
        print(f"  {label:<10} n={n:<5} predicted avg {avg_predicted*100:.1f}%  ->  "
              f"actual {actual_win_rate*100:.1f}%  ({gap:+.1f} pts)")

    print()
    print(f"  Total games: {len(records)}  |  Data through: {watermark}")
    print("  A well-calibrated model has actual ~= predicted in every row.")
    print("  Positive gap = model is underconfident in that range; negative = overconfident.")
    print()
