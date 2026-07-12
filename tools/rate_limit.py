import time
import json
import logging
import asyncio
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# -----------------------------
# THREAD-SAFE GLOBAL STATE
# -----------------------------
_lock = threading.Lock()

LAST_CALL_TIMES: Dict[str, float] = {}
PENDING_CALLS: Dict[str, Dict[str, Any]] = {}  # key = tool_name
PROBE_TTL_SECONDS = 5.0   # auto-expire old probe entries

# Structure: RATE_LIMITS = { tool_name: (requests_per_window, cooldown_seconds) }
# Initialized by server.py at startup - do not modify directly
RATE_LIMITS: Dict[str, tuple] = {}

# -----------------------------
# INITIALIZATION
# -----------------------------
def initialize_rate_limiter(limits: Optional[Dict[str, tuple]] = None):
    """
    Initialize rate limiter state and optionally set rate limits.

    Args:
        limits: Dictionary mapping tool names to (requests_per_window, cooldown_seconds)
    """
    global LAST_CALL_TIMES, PENDING_CALLS, RATE_LIMITS
    with _lock:
        LAST_CALL_TIMES = {}
        PENDING_CALLS = {}
        if limits is not None:
            RATE_LIMITS.clear()
            RATE_LIMITS.update(limits)
            # Pre-populate LAST_CALL_TIMES with zeros for all tools
            for tool_name in limits.keys():
                LAST_CALL_TIMES[tool_name] = 0.0
    # intentionally silent to avoid stderr noise during client startup


# -----------------------------
# INTERNAL HELPERS
# -----------------------------
def _safe_hash_args(args: dict) -> str:
    """Stable hash for arguments. Avoids JSON failures."""
    try:
        return json.dumps(args, sort_keys=True)
    except Exception:
        return str(args)


def _expired(ts: float) -> bool:
    """Remove stale probe entries automatically."""
    return (time.perf_counter() - ts) > PROBE_TTL_SECONDS


def _cleanup_expired():
    """Remove old pending entries without affecting behavior."""
    with _lock:
        for tool, entry in list(PENDING_CALLS.items()):
            if "ts" not in entry or _expired(entry["ts"]):
                del PENDING_CALLS[tool]


# -----------------------------
# MAIN ENFORCER FUNCTION
# -----------------------------
async def enforce(tool_name: str, args: dict) -> bool:
    """
    Determine whether a tool call should be allowed.
    Implements:
      - cooldowns
      - probe-to-real-call detection
      - safe fallbacks
      - automatic cleanup

    Returns:
        True  = allowed
        False = interpreted as a probe
    """
    global LAST_CALL_TIMES, PENDING_CALLS

    # Clean old probes
    _cleanup_expired()

    # Auto-heal corrupted state (sandbox reset, accidental overwrites)
    if not isinstance(LAST_CALL_TIMES, dict) or not isinstance(PENDING_CALLS, dict):
        initialize_rate_limiter()

    # Extract cooldown
    cooldown = RATE_LIMITS.get(tool_name, (0, 0))[1]

    now = time.perf_counter()
    args_hash = _safe_hash_args(args)

    with _lock:
        last_called = LAST_CALL_TIMES.get(tool_name)

        # 1. If cooldown is zero → always allow
        if cooldown == 0:
            LAST_CALL_TIMES[tool_name] = now
            return True

        # 2. First time ever calling this tool → record + allow
        if last_called is None:
            LAST_CALL_TIMES[tool_name] = now
            return True

        # 3. Check cooldown
        if (now - last_called) < cooldown:
            # This may be a probe attempt
            pending = PENDING_CALLS.get(tool_name)

            # 3a. No pending → create probe entry → return False (probe)
            if pending is None:
                PENDING_CALLS[tool_name] = {
                    "args_hash": args_hash,
                    "args": args,
                    "ts": now,
                }
                return False   # signal: PROBE
            else:
                # 3b. Pending exists → check if args match → allow real call
                if pending["args_hash"] == args_hash:
                    # Convert probe → real call
                    del PENDING_CALLS[tool_name]
                    LAST_CALL_TIMES[tool_name] = now
                    return True
                else:
                    # 3c. Arguments differ → restart probe tracking
                    PENDING_CALLS[tool_name] = {
                        "args_hash": args_hash,
                        "args": args,
                        "ts": now,
                    }
                    return False

        # 4. Cooldown passed → allow normal call
        LAST_CALL_TIMES[tool_name] = now
        return True


def get_rate_limit_info(tool_name: str) -> dict:
    """
    Get rate limit info for a tool.

    Returns:
        dict with: cooldown, elapsed, retry_in (all in seconds), limited (bool)
    """
    if tool_name not in RATE_LIMITS:
        return {"cooldown": 0, "limited": False}

    _, cooldown = RATE_LIMITS.get(tool_name, (0, 0))
    if cooldown == 0:
        return {"cooldown": 0, "limited": False}

    with _lock:
        last_called = LAST_CALL_TIMES.get(tool_name, 0)
        now = time.perf_counter()
        elapsed = now - last_called if last_called else cooldown
        retry_in = max(0, cooldown - elapsed)

        return {
            "cooldown": cooldown,
            "elapsed": round(elapsed, 2),
            "retry_in": round(retry_in, 2),
            "limited": retry_in > 0
        }
