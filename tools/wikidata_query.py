# tools/wikidata_query.py

import re
import time
import hashlib
from collections import OrderedDict
from typing import Dict, List, Any, Optional

import httpx
import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------
# CONFIGURATION & SECURITY
# ------------------------------------------------------------------------------------

MAX_QUERY_LENGTH = 5000          # Max SPARQL query length
MAX_RESULTS = 1000               # Max results to return
QUERY_TIMEOUT = 15               # Seconds (reduced from 30)
_wikidata_client: Optional[httpx.AsyncClient] = None

# Dangerous SPARQL patterns to block
BLOCKED_PATTERNS = [
    r"\bDROP\b",                  # DROP operations
    r"\bDELETE\b",                # DELETE operations
    r"\bINSERT\b",                # INSERT operations
    r"\bCLEAR\b",                 # CLEAR operations
    r"\bCREATE\b",                # CREATE operations
    r"\bLOAD\b",                  # LOAD operations
    r"\bCOPY\b",                  # COPY operations
    r"\bMOVE\b",                  # MOVE operations
    r"\bADD\b",                   # ADD operations (SPARQL Update)
    r"SERVICE\s*<(?!https://query\.wikidata\.org)",  # Block SERVICE to non-Wikidata endpoints
]

# Pre-compile patterns
_BLOCKED_REGEX = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]


# ------------------------------------------------------------------------------------
# CACHE (copied from duckduckgo_search.py)
# ------------------------------------------------------------------------------------

class BoundedTTLCache:
    """LRU cache with TTL and max size."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 7200):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds

    def get(self, key: str):
        if key not in self._cache:
            return None
        value, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value):
        if key in self._cache:
            del self._cache[key]
        elif len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (value, time.time())


_QUERY_CACHE = BoundedTTLCache(max_size=50, ttl_seconds=300)


def _validate_sparql_query(query: str) -> Optional[str]:
    """
    Validate SPARQL query for security.
    Returns error message if invalid, None if OK.
    """
    if not query or not isinstance(query, str):
        return "Query must be a non-empty string."

    query_clean = query.strip()

    if len(query_clean) < 10:
        return "Query too short (min 10 chars)."

    if len(query_clean) > MAX_QUERY_LENGTH:
        return f"Query too long (max {MAX_QUERY_LENGTH} chars)."

    # Check for blocked patterns
    for pattern in _BLOCKED_REGEX:
        if pattern.search(query_clean):
            return f"Query contains blocked operation. Only SELECT queries allowed."

    # Must contain SELECT (read-only queries only)
    if not re.search(r"\bSELECT\b", query_clean, re.IGNORECASE):
        return "Only SELECT queries are allowed. Query must contain SELECT."

    return None  # Query is valid


# ------------------------------------------------------------------------------------
# INTERNAL WORKER (async httpx)
# ------------------------------------------------------------------------------------

def _get_wikidata_client() -> httpx.AsyncClient:
    global _wikidata_client
    if _wikidata_client is None or _wikidata_client.is_closed:
        _wikidata_client = httpx.AsyncClient(
            headers={"User-Agent": "web-explorer/1.0 (+https://github.com/cutlerbenjamin1-cmd/web-explorer)"},
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            timeout=None,  # per-request timeout set on calls
        )
    return _wikidata_client


async def _run_sparql_query_async(query: str) -> List[Dict[str, Any]]:
    endpoint_url = "https://query.wikidata.org/sparql"

    client = _get_wikidata_client()
    response = await client.get(
        endpoint_url,
        params={"query": query, "format": "json"},
        timeout=QUERY_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    results: List[Dict[str, Any]] = []

    for binding in data["results"]["bindings"]:
        row = {}
        for var, value in binding.items():
            row[var] = value.get("value")
        results.append(row)

        # Cap results
        if len(results) >= MAX_RESULTS:
            break

    return results


# ------------------------------------------------------------------------------------
# TOOL FUNCTION
# ------------------------------------------------------------------------------------

async def wikidata_query(query: str) -> Dict[str, Any]:
    """
    Query Wikidata using SPARQL and return parsed results.

    Only SELECT queries are allowed (read-only).
    Max query length: 5000 chars. Max results: 1000.

    :param query: SPARQL query string.
    """
    # Validate query
    validation_error = _validate_sparql_query(query)
    if validation_error:
        logger.warning(f"[HUMAN] Wikidata query blocked: {validation_error}")
        return {"success": False, "error": validation_error}

    cache_key = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
    cached = _QUERY_CACHE.get(cache_key)
    if cached:
        return cached

    try:
        result_data = await _run_sparql_query_async(query)

        truncated = len(result_data) >= MAX_RESULTS

        response = {
            "success": True,
            "query": query[:200] + "..." if len(query) > 200 else query,
            "results": result_data,
            "count": len(result_data),
            "truncated": truncated,
        }
        _QUERY_CACHE.set(cache_key, response)  # Cache successful response
        return response
    except httpx.TimeoutException:
        return {"success": False, "error": f"Query timed out after {QUERY_TIMEOUT} seconds."}
    except httpx.RequestError as e:
        return {"success": False, "error": f"Request failed: {e}"}
    except Exception as e:
        logger.error(f"[HUMAN] Wikidata query error: {e}")
        return {"success": False, "error": str(e)}
