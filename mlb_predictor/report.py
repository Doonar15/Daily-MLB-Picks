"""Generates a local, static HTML report of ROI Picks results: what was
predicted, what actually happened, and a running record + units-based ROI
over time. Scoped to ROI Picks only (the real expected-value plays vs. the
market) -- the broader Top Picks pool isn't shown here at all.

Pure rendering layer -- reads history.py's predictions.jsonl (already
recorded and graded there) and calibration.py's live calibration data (for
the adjusted-confidence column). Doesn't record or grade anything itself;
run --grade-picks first so the data here is current.

One rolling file (REPORT_FILE), regenerated in full each call -- the most
recent date is shown first, with a running record and units total computed
across every graded ROI Pick. Open it in any browser; no server needed.
"""

import html
from collections import defaultdict
from datetime import date
from pathlib import Path

from . import calibration, history
from .market_edge import profit_per_100_wagered

REPORT_DIR = Path(__file__).parent / ".history"
REPORT_FILE = REPORT_DIR / "picks_report.html"

# .history/ is a dot-directory (hidden by Finder/most file browsers by
# default), which makes sense for the canonical/internal copy but is
# unfriendly for something meant to be opened by double-clicking. A second,
# identical copy is written to the project root -- always visible, no
# "show hidden files" toggle needed -- while REPORT_FILE above stays the
# source of truth that other code (and this module, on next regenerate)
# reads from.
VISIBLE_REPORT_FILE = Path(__file__).parent.parent / "picks_report.html"

# GitHub Pages only auto-serves a file literally named index.html as the
# site's homepage -- it won't recognize picks_report.html no matter its
# content. This third copy (identical content, root-level, same as
# VISIBLE_REPORT_FILE) exists purely so `git add index.html && git push`
# updates the live site. picks_report.html stays the one to double-click
# locally; index.html is the one that matters to git/GitHub.
PUBLISH_FILE = Path(__file__).parent.parent / "index.html"


def _roi_pick_records():
    return [r for r in history.get_all_predictions() if r.get("is_roi_pick")]


def _group_by_date(records):
    by_date = defaultdict(list)
    for r in records:
        by_date[r["date"]].append(r)
    return dict(sorted(by_date.items(), reverse=True))


def _roi_won(record):
    """Win/loss on the actual RECOMMENDED side (roi_pick_side), not
    necessarily the model's own predicted_winner -- these can differ (see
    history.record_predictions' docstring). Falls back to correct/
    predicted_winner for older records that predate roi_pick_side.
    """
    return record.get("roi_correct", record["correct"])


def _season_record(records):
    graded = [r for r in records if r["graded"]]
    if not graded:
        return None
    wins = sum(1 for r in graded if _roi_won(r))
    return wins, len(graded) - wins, len(graded)


def _day_record(day_records):
    graded = [r for r in day_records if r["graded"]]
    if not graded:
        return None
    wins = sum(1 for r in graded if _roi_won(r))
    return wins, len(graded) - wins


def _unit_profit(record):
    """Flat-1-unit-risk profit for one graded, priced pick: a loss always
    costs exactly 1u; a win pays out per the market price. Reuses
    market_edge.profit_per_100_wagered (same math, already written for the
    model-vs-market backtest) scaled from a $100 wager down to 1 unit.
    Returns None if this record has no stored market_ml (ungraded, or
    graded without a captured price -- e.g. a live-odds fetch failure that
    day) so callers can exclude it from the units total instead of silently
    counting it as a 0-profit pick.
    """
    if not record["graded"] or record.get("market_ml") is None:
        return None
    return profit_per_100_wagered(record["market_ml"], _roi_won(record)) / 100.0


def _units_summary(records):
    """(net_units, n_priced) across every graded record with a stored
    market_ml. n_priced is also the total units risked under flat-1u-risk
    staking, so net_units / n_priced is the ROI%.
    """
    profits = [p for p in (_unit_profit(r) for r in records) if p is not None]
    return sum(profits), len(profits)


def _result_pill(record):
    if not record["graded"]:
        return ("pending", "Pending")
    won = _roi_won(record)
    label = "Won" if won else "Lost"
    profit = _unit_profit(record)
    if profit is not None:
        label = f"{label} {profit:+.2f}u"
    return ("win" if won else "loss", label)


def _movement_note(record):
    """If this game's raw confidence moved across re-runs (e.g. once
    real lineups posted), show the trail: first-seen -> ... -> final.
    Returns '' if there's no history or only one distinct value recorded.
    """
    history_entries = record.get("probability_history") or []
    distinct = []
    for entry in history_entries:
        p = entry["predicted_prob"]
        if not distinct or distinct[-1] != p:
            distinct.append(p)
    if len(distinct) < 2:
        return ""
    trail = " &rarr; ".join(f"{p*100:.1f}%" for p in distinct)
    return f'<div class="pick-movement">moved during the day: {trail}</div>'


def _pick_row_html(i, record):
    home, away = record["home_team"], record["away_team"]
    favorite = record.get("roi_pick_side") or record["predicted_winner"]
    underdog = away if favorite == home else home
    # The model's own stated confidence in WHICHEVER side is being shown as
    # the pick here -- not necessarily predicted_prob, which is the model's
    # confidence in its own predicted_winner. Those differ exactly when
    # roi_pick_side != predicted_winner (see history.record_predictions'
    # docstring); re-deriving from home_win_prob keeps this number honest
    # for the side actually displayed, whichever one that is.
    raw_prob = record["home_win_prob"] if favorite == home else 1 - record["home_win_prob"]
    adjusted, adj_n = calibration.get_adjusted_confidence(raw_prob)
    market_ml = record.get("market_ml")
    adj_html = (
        f'<span><span class="metric-label">adj</span> <span class="adjusted-val">{adjusted*100:.1f}%</span></span>'
        if adj_n > 0 else
        '<span><span class="metric-label">adj</span> <span class="adjusted-val">N/A</span></span>'
    )
    pill_class, pill_label = _result_pill(record)
    ml_str = f"{market_ml:+.0f}" if market_ml is not None else "N/A"
    movement_html = _movement_note(record)

    return f"""
    <div class="pick-row">
      <div class="pick-num">{i}</div>
      <div class="pick-main">
        <div class="pick-matchup"><span class="fav">{html.escape(favorite)}</span> <span class="vs">to beat</span> {html.escape(underdog)}</div>
        <div class="pick-meta">
          <span><span class="metric-label">raw</span> <span class="raw-val">{raw_prob*100:.1f}%</span></span>
          {adj_html}
          <span><span class="metric-label">ML</span> {ml_str}</span>
        </div>
        {movement_html}
      </div>
      <div class="result-pill {pill_class}"><span class="dot"></span>{pill_label}</div>
    </div>"""


def _date_section_html(day_str, day_records):
    day_records = sorted(day_records, key=lambda r: r["predicted_prob"], reverse=True)
    record = _day_record(day_records)
    record_str = f"{record[0]}–{record[1]}" if record else "pending"
    rows = "\n".join(_pick_row_html(i, r) for i, r in enumerate(day_records, start=1))

    try:
        pretty_date = date.fromisoformat(day_str).strftime("%a %b %d %Y")
    except ValueError:
        pretty_date = day_str

    return f"""
  <section class="day-section">
    <div class="day-header">
      <div class="day-date">{html.escape(pretty_date)}</div>
      <div class="day-record">{record_str}</div>
    </div>
    <div class="picks">
      {rows}
    </div>
  </section>"""


def generate():
    """Render the full report to REPORT_FILE. Returns the file path.
    Safe to call with no data yet (renders an empty-state page).
    """
    records = _roi_pick_records()
    by_date = _group_by_date(records)
    season = _season_record(records)
    ungraded_count = sum(1 for r in records if not r["graded"])

    net_units, n_priced = _units_summary(records)
    if n_priced > 0:
        units_summary_cell = f"""
    <div class="summary-cell">
      <div class="summary-label">Units</div>
      <div class="summary-value accent">{net_units:+.2f}u</div>
      <div class="summary-sub">{net_units / n_priced * 100:+.1f}% ROI</div>
    </div>"""
    else:
        units_summary_cell = f"""
    <div class="summary-cell">
      <div class="summary-label">Units</div>
      <div class="summary-value accent">—</div>
      <div class="summary-sub">no priced picks yet</div>
    </div>"""

    most_recent_date = next(iter(by_date), None)
    graded_dates = len({d for d, recs in by_date.items() if _day_record(recs) is not None})

    if season:
        season_wins, season_losses, season_total = season
        season_pct = season_wins / season_total * 100
        season_summary = f"""
    <div class="summary-cell">
      <div class="summary-label">Record</div>
      <div class="summary-value accent">{season_wins}–{season_losses}</div>
      <div class="summary-sub">{season_pct:.1f}% overall</div>
    </div>{units_summary_cell}
    <div class="summary-cell">
      <div class="summary-label">Days tracked</div>
      <div class="summary-value">{graded_dates}</div>
      <div class="summary-sub">{len(by_date)} total, incl. pending</div>
    </div>
    <div class="summary-cell">
      <div class="summary-label">Picks graded</div>
      <div class="summary-value">{season_total}</div>
      <div class="summary-sub">{ungraded_count} still pending</div>
    </div>"""
    else:
        season_summary = f"""
    <div class="summary-cell">
      <div class="summary-label">Record</div>
      <div class="summary-value accent">—</div>
      <div class="summary-sub">no graded picks yet</div>
    </div>
    <div class="summary-cell">
      <div class="summary-label">Picks recorded</div>
      <div class="summary-value">{len(records)}</div>
      <div class="summary-sub">{ungraded_count} pending</div>
    </div>"""

    if by_date:
        sections = "\n".join(_date_section_html(d, recs) for d, recs in by_date.items())
    else:
        sections = """
  <div class="empty-state">
    No ROI Picks recorded yet. Run <code>python -m mlb_predictor.predictor</code>
    to generate today's picks, then <code>--grade-picks</code> once games are final.
  </div>"""

    date_line = most_recent_date or "no picks yet"

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ROI Picks Ledger</title>
<style>
:root {{
  --paper: #F7F5EF;
  --paper-raised: #FCFBF7;
  --ink: #1A1D1B;
  --ink-soft: #52564F;
  --ink-faint: #8B8E85;
  --rule: #DAD6C9;
  --rule-strong: #C4BFAE;
  --accent: #2C5F4A;
  --accent-soft: #E4ECE6;
  --gold: #96742F;
  --win: #23693F;
  --win-soft: #DEEBE1;
  --loss: #A63D33;
  --loss-soft: #F3DEDB;
  --pending: #8B8E85;
  --pending-soft: #EAE8E0;
  --shadow: 0 1px 2px rgba(26,29,27,0.04), 0 4px 12px rgba(26,29,27,0.05);
}}

@media (prefers-color-scheme: dark) {{
  :root {{
    --paper: #17191A;
    --paper-raised: #1E211F;
    --ink: #ECEAE2;
    --ink-soft: #B7B6AC;
    --ink-faint: #7C7F76;
    --rule: #33362F;
    --rule-strong: #43463B;
    --accent: #6FAE8E;
    --accent-soft: #223229;
    --gold: #D3B36A;
    --win: #6FAE8E;
    --win-soft: #1E2F24;
    --loss: #E28579;
    --loss-soft: #362121;
    --pending: #8B8E85;
    --pending-soft: #26282389;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 16px rgba(0,0,0,0.35);
  }}
}}

* {{ box-sizing: border-box; }}

body {{
  background: var(--paper);
  color: var(--ink);
  font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  margin: 0;
  padding: 40px 20px 80px;
  -webkit-font-smoothing: antialiased;
}}

.page {{
  max-width: 720px;
  margin: 0 auto;
}}

header {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 16px;
  border-bottom: 2px solid var(--ink);
  padding-bottom: 14px;
  margin-bottom: 4px;
  flex-wrap: wrap;
}}

.masthead {{
  font-family: Georgia, "Times New Roman", serif;
  font-weight: 700;
  font-size: 26px;
  letter-spacing: -0.01em;
  text-wrap: balance;
}}

.date-line {{
  font-size: 12.5px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--ink-faint);
  font-variant-numeric: tabular-nums;
}}

.summary-bar {{
  display: flex;
  gap: 0;
  margin: 20px 0 32px;
  border: 1px solid var(--rule);
  border-radius: 10px;
  overflow: hidden;
  background: var(--paper-raised);
  box-shadow: var(--shadow);
  flex-wrap: wrap;
}}

.summary-cell {{
  flex: 1;
  min-width: 140px;
  padding: 16px 18px;
  border-right: 1px solid var(--rule);
}}
.summary-cell:last-child {{ border-right: none; }}

.summary-label {{
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin-bottom: 6px;
}}

.summary-value {{
  font-family: Georgia, serif;
  font-weight: 700;
  font-size: 24px;
  font-variant-numeric: tabular-nums;
  color: var(--ink);
}}
.summary-value.accent {{ color: var(--accent); }}
.summary-sub {{
  font-size: 12px;
  color: var(--ink-soft);
  margin-top: 2px;
}}

.day-section {{ margin-bottom: 28px; }}

.day-header {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin: 0 0 10px 2px;
}}

.day-date {{
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-faint);
}}

.day-record {{
  font-size: 12.5px;
  font-weight: 700;
  color: var(--ink-soft);
  font-variant-numeric: tabular-nums;
}}

.picks {{ border-top: 1px solid var(--rule-strong); }}

.pick-row {{
  display: grid;
  grid-template-columns: 28px 1fr auto;
  align-items: center;
  gap: 14px;
  padding: 14px 2px;
  border-bottom: 1px solid var(--rule);
}}

.pick-num {{
  font-family: Georgia, serif;
  font-weight: 700;
  font-size: 15px;
  color: var(--ink-faint);
  font-variant-numeric: tabular-nums;
}}

.pick-main {{ min-width: 0; }}

.pick-matchup {{
  font-size: 15px;
  font-weight: 600;
  color: var(--ink);
  margin-bottom: 5px;
  line-height: 1.3;
}}
.pick-matchup .fav {{ color: var(--accent); }}
.pick-matchup .vs {{ color: var(--ink-faint); font-weight: 400; }}

.pick-meta {{
  display: flex;
  gap: 14px;
  font-size: 12.5px;
  color: var(--ink-soft);
  font-variant-numeric: tabular-nums;
  flex-wrap: wrap;
}}
.pick-meta .metric-label {{ color: var(--ink-faint); font-variant-numeric: normal; }}
.pick-meta .adjusted-val {{ color: var(--gold); font-weight: 600; }}
.pick-meta .raw-val {{ color: var(--ink); font-weight: 600; }}

.pick-movement {{
  font-size: 11.5px;
  color: var(--ink-faint);
  margin-top: 4px;
  font-variant-numeric: tabular-nums;
}}

.result-pill {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.02em;
  white-space: nowrap;
}}
.result-pill.win {{ background: var(--win-soft); color: var(--win); }}
.result-pill.loss {{ background: var(--loss-soft); color: var(--loss); }}
.result-pill.pending {{ background: var(--pending-soft); color: var(--pending); }}
.result-pill .dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}

.empty-state {{
  padding: 40px 20px;
  text-align: center;
  color: var(--ink-soft);
  font-size: 14px;
  border: 1px dashed var(--rule-strong);
  border-radius: 10px;
}}
.empty-state code {{
  background: var(--accent-soft);
  color: var(--accent);
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 12.5px;
}}

</style>
</head>
<body>
<div class="page">

  <header>
    <div class="masthead">ROI Picks Ledger</div>
    <div class="date-line">{html.escape(str(date_line))}</div>
  </header>

  <div class="summary-bar">{season_summary}
  </div>

  {sections}

</div>
</body>
</html>"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(doc)
    VISIBLE_REPORT_FILE.write_text(doc)  # same content, in a location Finder shows by default
    PUBLISH_FILE.write_text(doc)  # same content again, named for GitHub Pages
    return VISIBLE_REPORT_FILE
