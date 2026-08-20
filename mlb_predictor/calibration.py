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
# Picks. Re-derived after model.Weights.shrinkage_factor (0.45) was added to
# compress the final probability toward 50% -- see that field's docstring
# for why. Shrinkage compressed the whole raw-confidence scale into roughly
# 50-70% (down from 50-95% before), and largely fixed the overconfidence
# problem that motivated the original 0.58 floor: an offline sweep over
# confidence.jsonl (5,976 games, transform applied to the already-recorded
# pre-shrinkage confidence values, exact since shrinkage is the model's
# literal last step) showed the 50-55%/55-60%/60-65% shrunk bands all within
# a few points of honest (the old floor existed specifically because 60%+
# WASN'T). 55% sits just above the noisiest, least-differentiated part of
# the scale (50-52%, n=1782, essentially a coin flip) while keeping a broad,
# still-curated pool (~33% of games clear it) as Top Picks. Threshold is
# intentionally on RAW (post-shrinkage) confidence, not the adjusted number
# -- the whole point of showing both is that a raw 66% pick might turn out
# to have an adjusted confidence below this floor, and that gap is exactly
# what's worth seeing, not hiding by thresholding post-adjustment.
# RE-CHECKED after model.Weights.market_blend_weight and sample_shrinkage_games
# were added on top of shrinkage_factor: rebuilt calibration.jsonl under the
# full model (2025+2026, 4,448 games, market-blended for the ranges the local
# odds files cover) and re-ran the bucket breakdown. Still holds -- every
# 2pt band from 50-65% stayed within a few points of honest (the one outlier,
# 65-70% at +12.3pts, is n=19, noise). No change needed; 0.55 remains inside
# the well-calibrated range on the new (slightly narrower, ~50-69%) scale.
MIN_PICK_CONFIDENCE = 0.55

# Minimum raw confidence for a Top Pick to also count as a backtested ROI
# pick -- a much narrower, betting-specific bar than MIN_PICK_CONFIDENCE.
# Originally set to 0.75 from a RAW-confidence sweep (2025+2026 pooled,
# 4,118 games): only >=75% raw with no favorite cap showed a real edge
# (+4.2% ROI, 85.8% P(profit), 240 picks). Re-run on CALIBRATION-ADJUSTED
# confidence once the 70-80%/80-90% buckets had real sample sizes (n=481,
# n=98) told a sharper story: the 70-80% raw tier's true win rate is only
# 59.5% and showed NO edge anywhere (every cell net negative); the edge was
# entirely concentrated in the 80-90% raw tier (true win rate 69.4%) --
# +3.7% ROI/72.4% P(profit) uncapped (91 picks), up to +6.4%/75.9% P(profit)
# with a -250 favorite cap (60 picks). The old 0.75 bar was diluting that
# real, concentrated signal with a much larger pool of 70-80% picks that
# had no edge at all -- raised to 0.80 to isolate just the tier that
# actually showed one. Smaller sample than the original 75% finding (60-91
# picks vs. 240), so still the leading hypothesis, not a proven edge -- see
# unit_roi_backtest.py's docstring and its --adjusted-confidence sweep mode
# for the full methodology.
#
# RE-DERIVED after model.Weights.shrinkage_factor's addition compressed raw
# confidence into roughly 50-70%, making the old 0.80 mathematically
# unreachable. Re-ran the same RAW-confidence sweep (2025+2026 pooled, 4,116
# games) rather than --adjusted-confidence, since that mode draws its
# adjustment from calibration.jsonl -- which still reflects the PRE-shrinkage
# model and would've compared the new model's output against the old model's
# historical accuracy in the same numeric range, not a clean read. Same
# pattern as the original 0.75->0.80 story: edge is concentrated in one tier,
# not "higher is better" -- everything below 62% raw was net negative, then
# >=62% raw (no favorite cap) turned real: 181 picks, 67.4% hit rate, +5.2%
# ROI, 86.2% P(profit), 90% CI [-3.0%,+13.2%]. >=65% raw posts a better point
# estimate (+8.7-9.1% ROI) but on a third the sample (70 picks) -- a
# reasonable tightening later as more games accumulate, not the first move.
# See unit_roi_backtest.py's docstring and its --sweep mode for the full
# methodology.
#
# RETIRED as the live RECOMMENDED-pick decision as of ROI_PICK_FAVORITE_PRICE_CAP
# below: model.Weights.market_blend_weight's own testing found that model-vs-
# market disagreement (the entire premise this confidence-based bar and
# EV_ROI_MARGIN/the old EV method depended on) is NOT informative -- hit rate
# and ROI got worse, not better, as disagreement grew (market_edge.py's edge-
# bucket backtest). The validated replacement doesn't use the model's opinion
# for the decision at all. Left defined (not deleted) since unit_roi_backtest.py
# still references it in its own diagnostic sweep methodology/docstrings.
ROI_PICK_CONFIDENCE = 0.62

# The live RECOMMENDED-pick decision as of model.Weights.market_blend_weight
# (see its docstring in model.py): pick whichever side the MARKET's own
# devigged probability favors -- not the model's -- and recommend it unless
# that side is a heavily-juiced (expensive) favorite. No confidence
# threshold, no model opinion involved; the model's own favored side is
# irrelevant to this decision (predictor._market_favorite_pick).
#
# Validated via unit_roi_backtest.py's sweep methodology on plain raw market
# probability (2025+2026 pooled, 4,118 games): every cell below this cap was
# either unprofitable or too small a sample; >=50% confidence (true by
# construction -- it's whichever side is favored) with favorites priced -110
# or better turned real: 189 picks, 60.3% hit rate, +15.2% ROI, 2000x
# bootstrap 90% CI [+4.0%,+26.3%] (never crosses zero), P(profit)=98.6% --
# the tightest, best-supported edge found across every signal tested this
# project (see model.py's market_blend_weight docstring for the broader
# context). A price cap tighter than -110 wasn't sweep-validated as better;
# loosening it hasn't been tested and shouldn't be assumed safe.
ROI_PICK_FAVORITE_PRICE_CAP = 110

# Buckets used for Top Picks display/threshold purposes specifically (as
# opposed to bucket_report's fixed 10-point report buckets above), after a
# calibration backfill extending back through the full 2025 season:
# 70-80%: n=481, -14.3pt gap. 80-90%: n=98, -14.4pt gap -- both now real,
# usable samples with a consistent, similar-magnitude overconfidence gap
# to each other, so both are enabled. 90-100% stays disabled at n=27 --
# bigger than before but still thin, and its measured gap (-38.0pt, actual
# win rate barely above a coin flip) is extreme enough that it needs a
# larger sample before trusting it rather than treating it as a fluke.
# Kept here, commented out, so re-enabling it is a one-line change once its
# sample size is large enough to trust -- check bucket_report() output
# before flipping it on.
ADJUSTMENT_BUCKETS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),  # n=481 as of last check
    (0.80, 0.90),  # n=98 as of last check
    # (0.90, 1.001), # n=27 as of last check -- still too thin, extreme measured gap
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


def backfill(start_date: str = None, end_date: str = None, market_odds=None):
    """One-time (or re-run-to-extend) bulk historical pull. Backtests
    start_date (default: ~4 months ago) through end_date (default: yesterday)
    and saves every graded game to the calibration log, then sets the
    watermark. Safe to call again with an earlier start_date than before --
    results aren't deduped against the watermark here, so only call this for
    a genuinely new range (use top_up() for the routine "since last time"
    case).

    market_odds, if given, is passed straight through to run_backtest so the
    calibration data reflects model.Weights.market_blend_weight for a range
    a historical odds file actually covers (there's no free historical-odds
    API to cover an arbitrary range -- see model.py's market_blend_weight
    docstring). Defaults to None: routine top-ups of the live rolling window
    have no such file available and fall back to the pure (unblended) model,
    same as always.
    """
    start_date = start_date or default_backfill_start()
    end_date = end_date or (date.today() - timedelta(days=1)).isoformat()
    if start_date > end_date:
        return 0

    results = backtest.run_backtest(start_date, end_date, market_odds=market_odds, **CALIBRATION_FLAGS)
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
