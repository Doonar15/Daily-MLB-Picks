# MLB Game Outcome Predictor

Pulls live data from the free, public MLB Stats API (no API key required)
and estimates win probabilities for a given day's games.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m mlb_predictor.predictor                                     # today's games, auto-tuned, recommended signals on
python -m mlb_predictor.predictor --date 2026-08-05                   # specific date
python -m mlb_predictor.predictor --odds                              # prompts for sportsbook odds to calc edge
python -m mlb_predictor.predictor --bullpen                           # also include bullpen (L30 ERA) adjustment
python -m mlb_predictor.predictor --rest                              # also include starter short-rest penalty
python -m mlb_predictor.predictor --weather                           # also amplify park factor by forecast wind
python -m mlb_predictor.predictor --bullpen-availability              # also penalize fatigued top relievers
python -m mlb_predictor.predictor --no-park-factors                   # disable park run-factor adjustment
python -m mlb_predictor.predictor --no-handedness                     # disable handedness adjustment
python -m mlb_predictor.predictor --no-lineups                        # disable lineup strength adjustment
python -m mlb_predictor.predictor --no-defense                        # disable the OAA (defense) adjustment
python -m mlb_predictor.predictor --no-travel                         # disable the getaway-day/timezone travel penalty
python -m mlb_predictor.predictor --no-h2h                            # disable head-to-head this-season adjustment
python -m mlb_predictor.predictor --no-home-road-splits                # disable team-specific home/road adjustment
python -m mlb_predictor.predictor --bullpen --rest --weather --bullpen-availability  # everything enabled
python -m mlb_predictor.predictor --min-confidence 60                 # only show picks with raw confidence >= 60% (default: 58)
python -m mlb_predictor.predictor --no-tune                           # skip auto-tuning, use fixed default weights
python -m mlb_predictor.predictor --tune-window 14                    # tune against trailing 14 days instead of 30
python -m mlb_predictor.predictor --backtest 2026-07-01 2026-07-31    # backtest a historical date range
python -m mlb_predictor.predictor --grade-picks                       # grade saved predictions, show track record, update the HTML ledger
python -m mlb_predictor.predictor --report                            # regenerate the HTML ledger without grading first
python -m mlb_predictor.predictor --no-record                         # predict without saving to the history log
python -m mlb_predictor.predictor --backfill-history                  # ONE-TIME: pull historical games for calibration (see note below)
python -m mlb_predictor.predictor --backfill-history 2024-03-01       # backfill from a specific start date instead of the default
python -m mlb_predictor.predictor --calibration                       # show the confidence calibration report
python -m mlb_predictor.predictor --no-calibration-update              # skip the daily calibration top-up
python -m mlb_predictor.predictor --backfill-statcast                 # ONE-TIME: pull Statcast data for the xwOBA blend
python -m mlb_predictor.predictor --no-statcast                       # disable the Statcast xwOBA blend
python -m mlb_predictor.predictor --clear-cache                       # wipe the local API cache
```

Every run also prints a **Top Picks** summary: every game whose RAW model
confidence is at or above `--min-confidence` (default 58%, not a fixed
count) — some days that's 2 games, some days 8+. Each pick shows both the
raw confidence and a calibration-adjusted confidence (what that raw
confidence level has actually meant historically, per `--calibration`'s
data), for transparency about how much to trust the raw number. 58% was set
from real calibration data: the 50-60% band showed only a ~2pt gap between
predicted and actual (well-calibrated), while every band from 60% up showed
real, larger overconfidence that doesn't shrink as stated confidence rises —
see "Confidence calibration" below for the full picture and how to re-tune
this threshold as more data accumulates.

## Model inputs

1. **Pythagorean win expectation** — runs scored/allowed, season-to-date.
2. **Recent form** — last-10-games record, blended into the Pythagorean number.
3. **Starting pitcher** — probable starter's ERA vs. team rotation-average ERA.
4. **Bullpen** (`--bullpen`, off by default) — relief-only ERA over the last
   30 days vs. team ERA.
5. **Lineup strength** (**on by default**, disable via `--no-lineups`) —
   confirmed starting lineup's avg OPS vs. the team's full-roster season
   average OPS. MLB typically posts lineups only a few hours before game
   time, so an early-morning run will usually show no lineup data yet — this
   is a graceful no-op (falls back to contributing nothing), not an error.
   Re-run closer to first pitch to pick up the real lineup once it's posted.
6. **Home field advantage** — fixed ~54/46 baseline adjustment.
7. **Park factor** (**on by default**, disable via `--no-park-factors`) — a
   static per-park run-scoring index (100 = neutral; Coors Field ≈112,
   Oracle Park ≈90) scales both teams' expected runs for that game before
   the Pythagorean calculation. Sourced from a hardcoded table in
   [`mlb_predictor/parks.py`](mlb_predictor/parks.py) (refresh yearly — park
   factors drift, and teams occasionally play in temporary venues).
8. **Weather** (`--weather`, off by default, requires park factors enabled) —
   forecast wind speed at the park (via [Open-Meteo](https://open-meteo.com),
   free, no API key) amplifies the park factor's existing deviation from
   neutral when wind is significant (>15 mph). We don't have a reliable free
   source for each park's field orientation, so this doesn't resolve
   "blowing out" vs. "blowing in" — it's a coarse magnitude signal, not a
   precise physical model. Domed/retractable-roof parks are skipped. Only
   affects upcoming games; forecasts aren't available for past dates, so
   this has no effect on `--backtest` or auto-tuning.
9. **Handedness matchup** (**on by default**, disable via `--no-handedness`) —
   each team's season OPS against the opposing probable starter's throwing
   hand, compared to their overall season OPS. Uses OPS rather than wOBA
   since the free API's situational splits don't expose wOBA.
10. **Starter rest** (`--rest`, off by default) — penalizes a probable
    starter working on short rest (fewer than 4 days since their last
    start). Long layoffs (rehab assignments, roster moves) are treated as
    neutral, not a bonus.
11. **Statcast xwOBA blend** (**on by default**, disable via `--no-statcast`,
    requires `--backfill-statcast` first) — each team's season-to-date
    expected weighted on-base average (batting) and xwOBA-against
    (pitching), blended with the raw-runs Pythagorean number at a tunable
    weight. See "Statcast integration" below for the full design.
12. **Defense (OAA)** (**on by default**, disable via `--no-defense`) — each
    team's season-to-date Outs Above Average (fielding), sourced from
    Baseball Savant via `pybaseball`, scales the runs-allowed input before
    the Pythagorean calculation (good defense means some "runs allowed" are
    really a fielding effect, not pitching). **Not point-in-time**: unlike
    the xwOBA blend above, OAA is only available as a season-to-date/season-
    end snapshot from Savant (no raw per-play data to aggregate ourselves),
    so a backtest on a past date sees today's full-season OAA rather than
    what was true then — same limitation as handedness/lineups.
13. **Bullpen availability** (`--bullpen-availability`, off by default) —
    identifies each team's highest-leverage relievers (by saves + holds this
    season) and penalizes a team whose top reliever(s) threw a heavy-workload
    appearance (20+ pitches) within the last 2 days, as a proxy for "probably
    unavailable or fatigued today," independent of overall bullpen ERA. Off
    by default since it adds several extra API calls per game.
14. **Travel/getaway-day** (**on by default**, disable via `--no-travel`) —
    penalizes a team that played in a different city the day before, with a
    bigger penalty if that also crossed a timezone (derived from an explicit
    per-team timezone lookup table in `parks.py`, not a longitude formula —
    a naive longitude cutoff misclassifies cities like Chicago/Milwaukee that
    sit east of the true Eastern/Central line).
15. **Head-to-head history** (**on by default**, disable via `--no-h2h`) —
    each team's record against this specific opponent so far this season
    (point-in-time), nudging win probability once at least 3 games have been
    played between the two teams. Below that sample, contributes nothing.
16. **Home/road splits** (**on by default**, disable via `--no-home-road-splits`) —
    compares a team's own home-specific (or road-specific) win% this season
    against their own overall win%, nudging beyond the flat league-wide home
    field edge when a team is unusually strong or weak specifically in that
    venue context. Requires at least 10 games in that split.

**Why park factors, handedness, lineups, Statcast, defense, travel, h2h, and
home/road splits are on by default:** all are either backed by established
research, cheap to compute, or (for defense/travel/h2h/splits) added
alongside a real, measured backtest improvement during development — see
"Statcast integration" and the honest result noted there. Bullpen (ERA),
bullpen availability, rest, and weather are real but noisier/weaker signals
(or, for weather, hampered by a real data limitation — see above) and stay
opt-in so they don't dilute the auto-tuner's search with low-signal
dimensions unless you actually want them.

## Auto-tuning

By default, before predicting a day's games the model backtests the trailing
30 days (`--tune-window` to change), searches for weight values (home field
edge, recent-form weight, pitcher/bullpen/lineup/handedness/rest weights)
that minimize Brier score over that window, and uses the tuned weights for
that day's predictions. Only signals enabled via their flags (e.g.
`--handedness`) are included in tuning. The tuning dataset is disk-cached for
a few hours so repeated runs on the same day don't re-fetch from the API. Use
`--no-tune` to fall back to the fixed default weights instead. Tuned values
are bounded so the search can't drift to an implausible extreme (e.g. recent
form overwhelming the season-long number). Weather is never part of tuning —
forecasts don't exist for past dates, so tuning always uses the static park
factor.

## Caching

API responses (team stats, pitcher stats, bullpen stats, lineups, schedules,
handedness splits, rest days, weather, tuning datasets) are cached on disk
under `mlb_predictor/.cache/` with a TTL per data type (schedules refresh
hourly, season stats every 4 hours, lineups every 15 minutes, weather every
hour, tuning datasets every 4 hours, historical/completed data for 30 days).
Run with `--clear-cache` to wipe it.

## Backtesting

`--backtest START END` re-runs the model (with fixed default weights, not
auto-tuned) against completed historical games and reports accuracy and Brier
score.

**Point-in-time correctness:** team run stats, pitcher ERA, and recent form
are all scoped to what had actually happened before each backtest date (via
`stats=byDateRange` from opening day through the day before), not the team's
current full-season totals. This applies to auto-tuning too, since it reuses
the same data-gathering path. Head-to-head history and home/road splits (both
on by default) are also genuinely point-in-time, since they're computed from
schedule results directly. Things that are *not* point-in-time: bullpen ERA
already used a rolling 30-day window so it was point-in-time from the start;
lineup OPS, handedness splits, and OAA (defense) still use each team's
current full-season aggregate regardless of backtest date, since neither the
free MLB API's situational splits nor Baseball Savant's OAA leaderboard
support date-range scoping. Keep that in mind if you see those flags produce
unrealistically good backtest numbers on early-season dates — a lower
`--tune-window` also helps by keeping the tuning period recent, where the
season totals and the point-in-time truth are closer together anyway.

**Multi-season / large-range backtests:** `--backtest` and auto-tuning work
across year boundaries (`season` is derived per-day, so e.g.
`--backtest 2025-04-01 2026-08-01` correctly spans two seasons). Point-in-time
correctness means stats are cache-keyed per date, so a wide date range can't
reuse yesterday's cached team stats for today's game -- it's a genuinely new
API call per team per day. To keep that from being painfully slow over a full
season, `run_backtest` and `fetch_tuning_dataset` fetch games concurrently
(a small thread pool, since these are independent I/O-bound network calls).
`get_team_season_roster_ops` (used by `--handedness`, which is on by default)
also uses one hydrated roster call per team instead of one call per hitter.

**Honest timing note:** a measured 90-day range (1,167 games, default flags)
took about 3 minutes wall-clock with the thread pool. That doesn't scale
perfectly linearly to much larger ranges -- a real ~11-month test run during
development was still going after 30+ minutes and was intentionally stopped
rather than trusted blindly, likely from cumulative request volume against a
free public API with no documented rate limit (thousands of calls for a
season-plus of games). Budget accordingly for anything beyond a few months in
one range: a season at a time, run when you don't need the terminal for
anything else, is the safer way to build up a large history rather than one
huge multi-season call.

## Real-time track record

Every prediction run automatically saves each game's prediction (and whether
it was a Top Pick) to `mlb_predictor/.history/predictions.jsonl`, unless you
pass `--no-record`. This is different from backtesting: it's not a
reconstruction from historical stats, it's the literal prediction made at the
time, so grading it later has zero look-ahead risk.

Once games are final, run:

```bash
python -m mlb_predictor.predictor --grade-picks
```

This checks any ungraded saved predictions against final scores (via the same
MLB Stats API), marks each correct/incorrect, and prints your accuracy and
Brier score overall and for Top Picks specifically. Safe to run repeatedly —
already-graded entries aren't rescored, and predictions for games that
haven't finished yet are simply left for a future run to pick up.

This log is intentionally kept separate from `mlb_predictor/.cache/` and is
never touched by `--clear-cache` — it's a permanent record, not disposable
cached data. As of this version, this history is tracked but not yet fed back
into auto-tuning; once there's a meaningful sample of graded predictions (a
few weeks' worth), it's a natural next step to weigh real-time accuracy
alongside the backtest when tuning weights.

### HTML ledger

`--grade-picks` also regenerates a local, static HTML report —
`mlb_predictor/.history/picks_report.html` — showing every Top Pick with its
raw confidence, calibration-adjusted confidence, fair moneyline, and actual
result (Won / Lost / Pending), grouped by date with the most recent day on
top, plus a running season record. One rolling file, overwritten in full
each time — open it in any browser, no server needed. Run `--report` to
regenerate it without grading first (e.g. if you just want to re-view
current data). Games not yet final show as **Pending** and update on the
next `--grade-picks` run after they finish — normally the following day,
since most games are still in progress when you predict them that morning.

## Confidence calibration

A different question from the track record above: across a *large* sample of
games, when the model says "60% confident," does that side actually win about
60% of the time? The real-time log above will take weeks to accumulate enough
games to answer that reliably. Calibration instead uses the backtest engine
to reconstruct predictions over a wide historical range, once, and keeps that
dataset permanently (same look-ahead caveats as any backtest — see the
Backtesting section — but a much bigger sample to average over).

```bash
python -m mlb_predictor.predictor --backfill-history          # one-time: pull ~120 days by default
python -m mlb_predictor.predictor --calibration                # show the bucketed calibration report
```

`--backfill-history` uses `calibration.CALIBRATION_FLAGS`, which mirrors the
plain daily command's actual defaults (all the on-by-default signals, e.g.
park factors, handedness, lineups, Statcast, defense, travel, h2h, home/road
splits) and fixed (non-auto-tuned) default weights, so the report answers "is
the confidence level you see every day trustworthy" — not some other
configuration. It saves every graded game to
`mlb_predictor/.history/calibration.jsonl` plus a watermark date.

After the one-time backfill, **every plain daily run silently tops up** just
the gap since the last update (normally zero or one day, a few seconds) —
see `--no-calibration-update` to skip it. `--calibration` prints a report like:

```
50-60%     n=1021  predicted avg 54.6%  ->  actual 52.8%  (-1.8 pts)
60-70%     n=428   predicted avg 64.0%  ->  actual 58.2%  (-5.8 pts)
70-80%     n=95    predicted avg 73.4%  ->  actual 62.1%  (-11.3 pts)
```

A well-calibrated bucket has actual ≈ predicted. A positive gap means the
model is underconfident in that range (it happens more than it thinks); a
negative gap means overconfident. Small buckets (especially 80%+, which are
rare) will bounce around a lot until the sample grows — don't over-read a
single day's numbers, especially early on.

### Feeding calibration back into Top Picks

Two things in `calibration.py` are derived directly from this report, rather
than hardcoded:

- **`MIN_PICK_CONFIDENCE`** (default 0.58) — the floor `--min-confidence`
  uses by default. Set from real data: the 50-60% band was close to honest
  (small gap, huge sample) while every band from 60% up showed real,
  *larger* overconfidence that didn't shrink as stated confidence climbed —
  i.e. the model's most confident picks were not meaningfully more accurate
  in reality, just more confidently wrong. 58% sits inside the trustworthy
  part of the 50-60% band.
- **`ADJUSTMENT_BUCKETS`** — which confidence ranges are currently trusted
  enough to compute an adjusted confidence from (`get_adjusted_confidence`).
  Only 50-60% and 60-70% are active; 70-80%/80-90%/90-100% are present in
  the code **commented out**, since their sample sizes (n=95, n=9, n=2 as of
  last check) are too small to trust their measured gap. Top Picks shows
  "adjusted: N/A" for any raw confidence in a disabled range. Re-enable a
  tier by uncommenting its line once `--calibration` shows enough games in
  it — check the report before flipping one on, don't just uncomment on a
  schedule.

Top Picks (see Usage above) filters on **raw** confidence but displays both
numbers, so a 66% raw pick that's historically meant ~58% is visible as
such rather than silently hidden or silently trusted.

## Statcast integration

Adds Statcast-derived expected stats (via [`pybaseball`](https://github.com/jldbc/pybaseball),
which scrapes Baseball Savant) as a second, independent estimate of team
quality, blended with the existing raw-runs Pythagorean calculation rather
than replacing it.

```bash
python -m mlb_predictor.predictor --backfill-statcast          # one-time: pull ~120 days by default
python -m mlb_predictor.predictor                               # xwOBA blend included automatically after that
```

**How it works:** raw Statcast data is pitch-level (~2,000+ rows/day across
all games) — far more granular than useful for this model. Rather than
re-fetching and re-aggregating that on every prediction, `--backfill-statcast`
pulls each calendar day **once**, reduces it locally to one compact row per
team per day (xwOBA, barrel%, plate-appearance count), and stores only that
in `mlb_predictor/.history/statcast_team_days.jsonl` — the same
backfill-once-then-watermark-top-up pattern used for calibration. After the
first backfill, the plain daily command silently fetches only yesterday's new
data (about a second) before predicting; no flag or manual step needed.

**xwOBA formula:** matches how Baseball Savant computes it — for each
plate-appearance-ending pitch, `estimated_woba_using_speedangle` (Savant's
own expected-value estimate, populated on every PA outcome, not only batted
balls) when available, falling back to the actual `woba_value` on the rare
row where it's missing. Team xwOBA-against (a pitching-quality proxy) is the
same computation from the opposing team's plate appearances. This is
genuinely **point-in-time by construction**: because it's aggregated from raw
per-day pulls rather than `pybaseball`'s season-only leaderboard functions
(which only take a `year`, no date range), any as-of-date query only
includes days before that date — unlike `--handedness`/`--lineups`, which are
season-only regardless of backtest date.

**Why blended, not a full replacement:** raw runs scored/allowed are real
outcomes and act as a sanity anchor; xwOBA is Baseball Savant's own model of
"expected" performance and could in principle have quirks we don't fully
understand. `statcast_blend_weight` (default 0.35, tunable, capped at 0.60 so
it can never fully override actual results) controls how much the xwOBA-based
win expectation counts vs. the raw-runs Pythagorean number, and the
auto-tuner searches this weight against real backtest data just like every
other signal in this model.

**Honest result from initial testing:** on a small sample (3 days, 38 games,
only about a week of backfilled Statcast history at the time), enabling the
blend at its default weight made the backtest's Brier score very slightly
*worse* (0.2600 vs. 0.2587 without it) — not conclusive with that little
data, but a useful reminder that "more signals" isn't automatically "more
accurate," and exactly why this was built as blended and tunable rather than
a forced replacement. Run `--backtest` or `--calibration` with a proper
backfilled sample to judge for yourself whether it's earning its weight over
time; if the auto-tuner consistently drives `statcast_blend_weight` toward
zero, that's the tool telling you honestly that this signal isn't adding
value yet, not something to override.

**Multi-season note:** the backfill/watermark mechanism is season-boundary
safe (same as everything else in this project) — extending into next season,
or the season after that, is just re-running `--backfill-statcast` with a
later date range, not a redesign.

**Timing:** measured at roughly 0.5 seconds/day sequentially (pybaseball
handles pagination and retries internally) — much faster than the game-level
backtest pulls, since it's one bulk pitch-level request per day rather than
many small per-team API calls. A 120-day backfill should take roughly a
minute. Sequential, not parallelized like the other backtest machinery,
since `pybaseball`'s underlying scrape doesn't document safe concurrency the
way the MLB Stats API does.

## Defense, bullpen availability, travel, head-to-head, and home/road splits

Five additional signals, all following the same design principles as the
rest of the model — point-in-time where the data allows it, capped and
tunable, and off by default only when the cost or reliability doesn't
justify it. See items 12–16 in "Model inputs" above for the full description
of each. Two implementation notes worth calling out:

- **OAA (defense) is the one signal in this batch that isn't point-in-time**,
  for the same reason as handedness/lineups: Baseball Savant's OAA
  leaderboard only exposes a season-to-date/season-end snapshot (`year`
  parameter only, no date range), and there's no raw per-play fielding data
  we can aggregate ourselves the way xwOBA is built.
- **Travel timezone detection uses an explicit per-team lookup table, not a
  longitude formula.** A first attempt using a pure longitude cutoff
  misclassified Chicago and Milwaukee as Eastern time (they're Central) —
  the real US timezone boundary snakes well east of a straight meridian line
  in the upper Midwest. With only 30 fixed MLB cities, an explicit table
  (`parks.TIMEZONES`) is both simpler and actually correct, verified against
  all 30 team locations.

**Honest result from initial testing:** a backtest over a small sample (3
days, 38 games) comparing these 5 signals on vs. off showed a real
improvement — accuracy went from 44.7% to 57.9%, Brier score from 0.2593 to
0.2504. That's a tiny sample and not proof on its own, but it's a much larger
and more consistent swing than the Statcast blend's initial (inconclusive,
slightly negative) result above. As with every other signal in this model,
judge it for yourself with `--backtest` or `--calibration` over a larger
window as more history accumulates, and watch what the auto-tuner does with
each weight — that's the honest, ongoing check on whether a signal is
actually earning its place, not something to take on faith from a small
sample at launch.

## Disclaimer

This is an educational statistical tool, not a guaranteed-win system. No
public model beats the closing line consistently over a large sample.
Bet only what you can afford to lose.
