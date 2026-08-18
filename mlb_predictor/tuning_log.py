"""Persisted history of the auto-tuner's daily output.

predictor.get_tuned_weights() re-derives weights from scratch every run via a
coordinate-descent search over a rolling trailing window (see backtest.py),
always starting from DEFAULT_WEIGHTS -- it never carries over the previous
day's result. That means day-to-day drift (how much a single day rolling in
or out of the window moves the tuned weights) was previously invisible: it
only lived in the 4-hour-TTL dataset cache and in stdout. This module gives
it a permanent, queryable home.

Storage is a local JSON-lines file, one calendar date per line, following the
same convention as calibration.py's calibration.jsonl.
"""

import json
from dataclasses import asdict
from pathlib import Path

from .model import Weights

TUNING_LOG_DIR = Path(__file__).parent / ".history"
TUNING_LOG_FILE = TUNING_LOG_DIR / "tuning_log.jsonl"


def _read_all():
    if not TUNING_LOG_FILE.exists():
        return []
    records = []
    for line in TUNING_LOG_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _write_all(records):
    TUNING_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with TUNING_LOG_FILE.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def upsert(record):
    """Write one day's tuning result, replacing any existing entry for the
    same as_of_date. Idempotent against re-running the CLI multiple times on
    the same calendar date.
    """
    records = [r for r in _read_all() if r.get("as_of_date") != record["as_of_date"]]
    records.append(record)
    records.sort(key=lambda r: r["as_of_date"])
    _write_all(records)


def get_smoothed_weights_before(as_of_date: str):
    """Return the most recently logged smoothed_weights strictly before
    as_of_date, as a Weights instance, or None if there's no prior entry
    (cold start -- e.g. the very first time smoothing runs).
    """
    prior = [r for r in _read_all() if r.get("as_of_date") < as_of_date and r.get("smoothed_weights")]
    if not prior:
        return None
    latest = max(prior, key=lambda r: r["as_of_date"])
    return Weights(**latest["smoothed_weights"])


def weights_to_dict(weights: Weights):
    return asdict(weights)
