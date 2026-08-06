"""Simple on-disk JSON cache with TTL, used to avoid re-hitting the MLB API
for data that doesn't change within a short window (team stats, pitcher
stats, etc.). Cache files live under mlb_predictor/.cache/ keyed by a
caller-supplied string key.

Writes are safe under concurrent access (backtest/tuning fetch games in
parallel via a thread pool -- see backtest.py) via write-to-temp-then-rename,
which is atomic on POSIX. Two threads racing to fill the same cache-miss key
just mean one redundant fetch, never a corrupted file.
"""

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"
DEFAULT_TTL_SECONDS = 4 * 60 * 60  # 4 hours


def _cache_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    return CACHE_DIR / f"{digest}.json"


def get(key: str, ttl_seconds: int = DEFAULT_TTL_SECONDS):
    """Return cached value for key, or None if missing/expired/corrupt."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - payload.get("cached_at", 0) > ttl_seconds:
        return None
    return payload.get("value")


def set(key: str, value):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)
    payload = {"cached_at": time.time(), "key": key, "value": value}
    tmp_path = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload))
    os.replace(tmp_path, path)  # atomic on POSIX -- safe under concurrent writers


def clear():
    """Delete all cached entries."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    return count


def cached(ttl_seconds: int = DEFAULT_TTL_SECONDS):
    """Decorator: cache a function's return value on disk, keyed by its
    name + args. Use for functions whose result is JSON-serializable.
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            key_parts = [fn.__name__] + [str(a) for a in args] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
            key = "|".join(key_parts)
            hit = get(key, ttl_seconds)
            if hit is not None:
                return hit
            result = fn(*args, **kwargs)
            if result is not None:
                set(key, result)
            return result
        wrapper.__wrapped__ = fn
        return wrapper
    return decorator
