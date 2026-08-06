"""Track record of the model's actual real-time predictions vs. what really
happened. Unlike backtesting (which reconstructs predictions after the fact
from historical stats), this records the prediction made in the moment, so
grading it against final scores later has zero look-ahead risk -- it's not
a reconstruction, it's literally "what did we say, and were we right."

Storage is a local JSON-lines file (one prediction per line), permanent --
never expired or touched by --clear-cache, since it's a log, not a cache.
"""

import json
from datetime import date
from pathlib import Path

from . import api

HISTORY_DIR = Path(__file__).parent / ".history"
HISTORY_FILE = HISTORY_DIR / "predictions.jsonl"


def _read_all():
    if not HISTORY_FILE.exists():
        return []
    records = []
    for line in HISTORY_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _write_all(records):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text("\n".join(json.dumps(r) for r in records) + ("\n" if records else ""))


def record_predictions(game_date: str, preds, top_pick_game_pks):
    """Append each of today's predictions to the history log, tagged with
    whether it was one of the day's Top Picks. Re-running the predictor for
    a date that hasn't been graded yet updates the entry for each game_pk in
    place (weights/lineups/pitchers can change closer to game time, so the
    latest pre-game prediction is what gets graded) -- but two things are
    preserved across re-runs rather than silently overwritten:

      - is_top_pick is STICKY: once a game clears the confidence floor on
        any run, it stays flagged as a Top Pick even if a later re-run's
        recalculated probability drops it back below the threshold. A pick
        that was shown to you shouldn't be able to un-become one just
        because real lineups/pitchers posted later that day and moved the
        number -- that's exactly the kind of swing worth being able to see
        and evaluate later, not erase.
      - probability_history accumulates every distinct raw confidence seen
        for that game that day, each with a timestamp, so the ledger can
        show "first seen 59.5%, later 55.6%" instead of just the final
        number. Grading still uses the LATEST prediction (predicted_prob /
        predicted_winner), since that's the fairest "final word" to judge --
        the history is additional context, not a replacement for it.

    Once an entry has been graded, it's left alone entirely -- re-predicting
    a past date never silently erases or alters a graded result.
    """
    existing = _read_all()
    by_key = {(r["date"], r["game_pk"]): r for r in existing}

    for pred in preds:
        key = (game_date, pred["game_pk"])
        prior = by_key.get(key)
        if prior is not None and prior["graded"]:
            continue  # never touch a graded entry

        if pred["home_win_prob"] >= pred["away_win_prob"]:
            predicted_winner, predicted_prob = pred["home_team"], pred["home_win_prob"]
        else:
            predicted_winner, predicted_prob = pred["away_team"], pred["away_win_prob"]

        now = date.today().isoformat()
        was_top_pick_before = bool(prior and prior.get("is_top_pick"))
        is_top_pick_now = pred["game_pk"] in top_pick_game_pks

        prob_history = list(prior["probability_history"]) if prior and "probability_history" in prior else (
            [{"predicted_prob": prior["predicted_prob"], "recorded_at": prior["recorded_at"]}] if prior else []
        )
        if not prob_history or prob_history[-1]["predicted_prob"] != predicted_prob:
            prob_history.append({"predicted_prob": predicted_prob, "recorded_at": now})

        by_key[key] = {
            "date": game_date,
            "game_pk": pred["game_pk"],
            "matchup": f"{pred['away_team']} @ {pred['home_team']}",
            "home_team": pred["home_team"],
            "away_team": pred["away_team"],
            "predicted_winner": predicted_winner,
            "predicted_prob": predicted_prob,
            "home_win_prob": pred["home_win_prob"],
            "is_top_pick": was_top_pick_before or is_top_pick_now,
            "probability_history": prob_history,
            "recorded_at": prior["recorded_at"] if prior else now,
            "graded": False,
            "actual_winner": None,
            "correct": None,
        }

    _write_all(list(by_key.values()))


def grade_predictions():
    """Check ungraded predictions for dates that have likely completed and
    mark them win/loss against final scores. Returns the number newly graded.
    Safe to call repeatedly -- already-graded entries are left alone, and
    entries without a final score yet (postponed, in progress) stay ungraded
    for a future call to pick up.
    """
    records = _read_all()
    ungraded_dates = {r["date"] for r in records if not r["graded"]}
    if not ungraded_dates:
        return 0

    scores_by_game_pk = {}
    for game_date in ungraded_dates:
        try:
            games = api.get_schedule(game_date)
        except Exception:
            continue
        for g in games:
            # Scores are non-null from the first pitch onward (in-progress
            # games included), so a not-None check alone is NOT enough to
            # know the game is actually over -- codedGameState == "F" is the
            # authoritative "this result is final" signal.
            if g.get("coded_status") != "F":
                continue
            if g.get("home_score") is not None and g.get("away_score") is not None:
                scores_by_game_pk[g["game_pk"]] = (g["home_team"], g["away_team"], g["home_score"], g["away_score"])

    newly_graded = 0
    for r in records:
        if r["graded"]:
            continue
        result = scores_by_game_pk.get(r["game_pk"])
        if result is None:
            continue
        home_team, away_team, home_score, away_score = result
        actual_winner = home_team if home_score > away_score else away_team
        r["graded"] = True
        r["actual_winner"] = actual_winner
        r["correct"] = actual_winner == r["predicted_winner"]
        newly_graded += 1

    if newly_graded:
        _write_all(records)
    return newly_graded


def get_all_predictions():
    """Return every recorded prediction (graded or not), each a dict with
    date, matchup, predicted_winner, predicted_prob, is_top_pick, graded,
    actual_winner, and correct. Public read accessor for other modules
    (e.g. report.py) that need the raw records rather than a summary.
    """
    return _read_all()


def summarize(top_picks_only=False):
    """Return win rate and Brier score over graded predictions, optionally
    restricted to entries that were a day's Top Pick.
    """
    records = [r for r in _read_all() if r["graded"]]
    if top_picks_only:
        records = [r for r in records if r["is_top_pick"]]

    if not records:
        return {"graded_games": 0, "accuracy": None, "brier_score": None}

    n = len(records)
    correct = sum(1 for r in records if r["correct"])
    brier = sum(
        (r["home_win_prob"] - (1.0 if r["actual_winner"] == r["home_team"] else 0.0)) ** 2
        for r in records
    ) / n

    return {"graded_games": n, "accuracy": correct / n, "brier_score": brier}


def print_track_record():
    ungraded = len([r for r in _read_all() if not r["graded"]])
    overall = summarize(top_picks_only=False)
    top_picks = summarize(top_picks_only=True)

    print("=" * 70)
    print("REAL-TIME TRACK RECORD  (actual predictions vs. actual results)")
    print("=" * 70)
    if overall["graded_games"] == 0:
        print("  No graded predictions yet.")
        if ungraded:
            print(f"  {ungraded} prediction(s) recorded but not yet gradeable (game not final).")
        print()
        return

    print(f"  All predictions:  {overall['graded_games']} graded, "
          f"{overall['accuracy']*100:.1f}% correct, Brier {overall['brier_score']:.4f}")
    if top_picks["graded_games"] > 0:
        print(f"  Top Picks only:    {top_picks['graded_games']} graded, "
              f"{top_picks['accuracy']*100:.1f}% correct, Brier {top_picks['brier_score']:.4f}")
    else:
        print("  Top Picks only:    no graded Top Picks yet")
    if ungraded:
        print(f"  ({ungraded} recorded prediction(s) still awaiting a final score)")
    print()
