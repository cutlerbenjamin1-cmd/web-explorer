import asyncio
import time
import logging
import ipaddress
import socket
import os
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import httpx
import re
from html.parser import HTMLParser

from collections import OrderedDict
import tools.rate_limit as rate_limit

logger = logging.getLogger(__name__)


# --------------------------------------------------------------
# Crawl Cache
# --------------------------------------------------------------

class BoundedTTLCache:
    """LRU cache with TTL and max size."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 1200):
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


_CRAWL_CACHE = BoundedTTLCache(max_size=100, ttl_seconds=1200)  # 20 min TTL


# --------------------------------------------------------------
# Security: SSRF Protection
# --------------------------------------------------------------

# Allowed ports for HTTP/HTTPS only
ALLOWED_PORTS = {80, 443, 8080, 8443}

# Private/internal IP ranges to block
BLOCKED_IP_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),        # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),     # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),    # Private Class C
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("169.254.0.0/16"),    # Link-local
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),          # IPv6 private
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
    ipaddress.ip_network("0.0.0.0/8"),         # "This" network
]

MAX_RESPONSE_BYTES = int(os.getenv("WEB_CRAWL_MAX_BYTES", str(2_000_000)))  # ~2MB cap for HTML/text
MAX_PDF_BYTES = int(os.getenv("WEB_CRAWL_MAX_PDF_BYTES", str(50_000_000)))  # 50MB cap for PDF bodies
HTTP_LIMITS = httpx.Limits(max_keepalive_connections=10, max_connections=20)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/webp,*/*;q=0.8"
    ),
    "Connection": "keep-alive",
    "DNT": "1"
}

_HTTP_CLIENT: Optional[httpx.AsyncClient] = None


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is in a private/blocked range."""
    try:
        ip = ipaddress.ip_address(ip_str)
        for network in BLOCKED_IP_RANGES:
            if ip in network:
                return True
        return False
    except ValueError:
        return False


def _validate_url_security(url: str) -> Optional[str]:
    """
    Validate URL for SSRF protection.
    Returns error message if invalid, None if OK.
    """
    try:
        parsed = urlparse(url)

        # Protocol check - only HTTP/HTTPS allowed
        if parsed.scheme.lower() not in ("http", "https"):
            return f"Invalid protocol '{parsed.scheme}'. Only http/https allowed."

        # Get hostname
        hostname = parsed.hostname
        if not hostname:
            return "URL has no hostname."

        # Port check
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80

        if port not in ALLOWED_PORTS:
            return f"Port {port} not allowed. Allowed ports: {ALLOWED_PORTS}"

        # Resolve hostname to check for private IPs
        try:
            # Get all IP addresses for hostname
            addr_info = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
            for family, socktype, proto, canonname, sockaddr in addr_info:
                ip = sockaddr[0]
                if _is_private_ip(ip):
                    return f"URL resolves to private/internal IP ({ip}). Blocked for security."
        except socket.gaierror:
            # DNS resolution failed - let httpx handle it
            pass

        return None  # URL is safe

    except Exception as e:
        return f"URL validation error: {e}"

# --------------------------------------------------------------
# Minimal additions from Universal Scraper
# --------------------------------------------------------------

def _modify_reddit_url(url: str) -> str:
    """
    Convert reddit urls to old reddit for cleaner static html.
    """
    match = re.match(r"^(https?://)(www\.)?(reddit\.com)(.*)$", url, re.IGNORECASE)
    if match:
        protocol = match.group(1)
        path_and_query = match.group(4)
        return f"{protocol}old.reddit.com{path_and_query}"
    return url


def extract_youtube_video_id(url: str) -> str | None:
    """
    Detect YouTube urls. Used only to prevent naive html crawling.
    """
    match = re.search(
        r"(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/|v/)|youtu\.be/)([^&?\n]+)",
        url
    )
    if match:
        return match.group(1)
    return None


# --------------------------------------------------------------
# Built in HTML text extractor
# --------------------------------------------------------------

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_stack = []

        self.exclude_tags = {
            "script", "style", "nav", "footer", "header",
            "aside", "button", "img", "noscript", "form",
            "iframe", "svg", "path", "meta", "link"
        }

    def handle_starttag(self, tag, attrs):
        if tag in self.exclude_tags:
            self.skip_stack.append(tag)

    def handle_endtag(self, tag):
        if self.skip_stack and self.skip_stack[-1] == tag:
            self.skip_stack.pop()

    def handle_data(self, data):
        if not self.skip_stack and data.strip():
            self.text_parts.append(data.strip())

    def get_text(self):
        text = " ".join(self.text_parts)
        text = " ".join(text.split())
        return text[:15000]


# --------------------------------------------------------------
# JS challenge / bot-protection detection
# --------------------------------------------------------------

_CHALLENGE_HTML_MARKERS = [
    "cf-browser-verification", "__cf_chl_", "cf_chl_prog",
    "cf-challenge-", "challenges.cloudflare.com", "cf-turnstile",
    "id=\"challenge-form\"", "id=\"challenge-running\"",
    "_Incapsula_Resource", "incap_ses_",
    "datadome", "dd.createCaptchaFrame",
    "px-captcha", "_pxhd",
    "akamai-bm-telemetry",
]

_CHALLENGE_TEXT_PHRASES = [
    "just a moment",
    "checking your browser",
    "checking if the site connection is secure",
    "enable javascript and cookies to continue",
    "please verify you are a human",
    "verify you are human",
    "please turn javascript on",
    "please enable javascript",
    "pardon our interruption",
    "one more step",
    "attention required",
    "access denied",
    "before you continue",
    "ray id",
    "performance & security by cloudflare",
    "ddos protection by",
]


def _detect_js_challenge(raw_html: str, extracted_text: str) -> Optional[str]:
    """Returns a description of the detected challenge, or None if clean."""
    html_lower = raw_html[:50000].lower()
    for marker in _CHALLENGE_HTML_MARKERS:
        if marker in html_lower:
            return f"bot-protection marker in HTML: {marker}"

    text_lower = extracted_text.lower()
    if len(extracted_text) < 2000:
        hits = sum(1 for phrase in _CHALLENGE_TEXT_PHRASES if phrase in text_lower)
        if hits >= 2:
            matched = [p for p in _CHALLENGE_TEXT_PHRASES if p in text_lower]
            return f"challenge page phrases detected: {', '.join(matched[:3])}"

    return None


# --------------------------------------------------------------
# Content extraction (trafilatura primary, TextExtractor fallback)
# --------------------------------------------------------------

def _extract_content(raw_html: str, url: str = "", summary: bool = True) -> tuple:
    """Returns (text, extraction_quality: str) where quality is 'good', 'fallback', or 'empty'"""
    MAX_SUMMARY = 5000
    MAX_FULL = 15000
    MIN_USEFUL = 200

    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        head = int(limit * 0.7)
        tail = limit - head - 80
        return text[:head] + f"\n\n[...{len(text) - head - tail:,} chars omitted...]\n\n" + text[-tail:]

    limit = MAX_SUMMARY if summary else MAX_FULL
    raw_len = len(raw_html)

    # Tier 1: trafilatura (article-aware extraction)
    traf_text = None
    try:
        import trafilatura
        traf_text = trafilatura.extract(
            raw_html,
            url=url or None,
            fast=summary,
            favor_precision=summary,
            favor_recall=not summary,
            include_comments=False,
            include_tables=True,
            include_links=False,
            deduplicate=True,
        )
    except Exception:
        pass

    # Accept trafilatura if it returned substantial content relative to page size
    traf_len = len(traf_text.strip()) if traf_text else 0
    traf_trustworthy = traf_len > MIN_USEFUL and (raw_len < 10000 or traf_len / raw_len > 0.01)
    if traf_trustworthy:
        return _truncate(traf_text.strip(), limit), "good"

    # Tier 2: BS4 smart extraction (article/main with noise removal)
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw_html, 'html.parser')

        for selector in [
            'script', 'style', 'nav', 'footer', 'header', 'aside',
            'noscript', 'iframe', 'form', 'button', 'svg',
            '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]',
            '.nav', '.navbar', '.header', '.footer', '.sidebar', '.menu',
            '.advertisement', '.ads', '.ad', '.social', '.share',
            '.cookie', '.popup', '.modal', '.overlay',
            '#nav', '#header', '#footer', '#sidebar', '#menu',
        ]:
            for el in soup.select(selector):
                el.decompose()

        main_content = None
        for selector in [
            'article', 'main', '[role="main"]',
            '.article', '.post', '.content', '.entry-content',
            '.post-content', '.article-content', '.main-content',
            '#content', '#main', '#article',
            '.markdown-body', '.blob-wrapper', '.readme',
        ]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.body or soup

        text_parts = []
        for el in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'td', 'th', 'pre', 'code', 'blockquote']):
            t = el.get_text(separator=' ', strip=True)
            if t and len(t) > 2:
                if el.name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                    t = '#' * int(el.name[1]) + ' ' + t
                text_parts.append(t)

        text = '\n\n'.join(text_parts)
        if len(text) < 200:
            text = main_content.get_text(separator=' ', strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        if text and len(text) > MIN_USEFUL:
            return _truncate(text, limit), "good"
    except Exception:
        pass

    # Accept short trafilatura output if it had anything at all
    if traf_text and len(traf_text.strip()) > 50:
        return _truncate(traf_text.strip(), limit), "fallback"

    # Tier 3: TextExtractor (last resort)
    try:
        extractor = TextExtractor()
        extractor.feed(raw_html)
        text = extractor.get_text()
        if text:
            return _truncate(text, limit), "fallback"
    except Exception:
        pass
    return "", "empty"


# --------------------------------------------------------------
# PDF handling
# --------------------------------------------------------------

def _is_pdf_url(url: str) -> bool:
    """URL-shape-only PDF detection. Cheap, no network needed.

    Matches '.pdf' suffix and known paper-host '/pdf/' conventions that
    don't carry a .pdf extension (arxiv, ACL anthology, OpenReview, SSRN).
    """
    u = url.lower().split("?", 1)[0].split("#", 1)[0]
    if u.endswith(".pdf"):
        return True
    if "/pdf/" in u and (
        "arxiv.org" in u
        or "aclanthology.org" in u
        or "openreview.net" in u
        or "papers.ssrn.com" in u
    ):
        return True
    return False


def _is_pdf_response(response, content_bytes: bytes, url: str) -> bool:
    """Detect whether a fetched response is a PDF (header / magic / URL shape)."""
    ctype = (response.headers.get("content-type") or "").lower()
    if "application/pdf" in ctype:
        return True
    if content_bytes[:5] == b"%PDF-":
        return True
    return _is_pdf_url(url)


def _pdf_bytes_to_markdown(content_bytes: bytes, url: str) -> str:
    """Convert in-memory PDF bytes to a markdown body string (no frontmatter).

    Uses PyMuPDF. Output is page-segmented with '## Page N' headers.
    Harvester prepends its own YAML frontmatter on save.
    """
    import fitz  # PyMuPDF - lazy import
    import unicodedata

    doc = fitz.open(stream=content_bytes, filetype="pdf")
    try:
        parts = []
        for i, page in enumerate(doc, 1):
            text = page.get_text("text")
            if not text.strip():
                continue
            text = unicodedata.normalize("NFKC", text)
            text = re.sub(r"\x00", "", text)
            text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", text)
            text = re.sub(r"-\n(\w)", r"\1", text)
            text = re.sub(r"\n{4,}", "\n\n\n", text)
            text = re.sub(r" {3,}", "  ", text)
            parts.append(f"\n---\n## Page {i}\n\n{text.strip()}\n")
        return "".join(parts).strip()
    finally:
        doc.close()


# --------------------------------------------------------------
# Shared HTTP client
# --------------------------------------------------------------

def _get_http_client() -> httpx.AsyncClient:
    """Return a shared AsyncClient to reuse connections across crawls."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(
            limits=HTTP_LIMITS,
            headers=DEFAULT_HEADERS,
            follow_redirects=True,
        )
    return _HTTP_CLIENT


# --------------------------------------------------------------
# Core crawl logic
# --------------------------------------------------------------

async def web_crawl(
    url: str,
    max_depth: int = 3,
    timeout_seconds: int = 15,
    harvest: bool = False,
    harvest_dest: str = "",
    summary: bool = True,
) -> Dict[str, Any]:
    """
    Web crawler that retrieves a page and embeds cleaned visible text.
    Now includes minimal YouTube detection and Reddit normalization.
    
    Args:
        url: Target URL to crawl
        max_depth: Max crawl depth (unused currently)
        timeout_seconds: Request timeout
        harvest: If True, save content to disk and return metadata only
        harvest_dest: Subfolder under Resource Docs/harvested/ (e.g., "docs/")
    """

    if not isinstance(url, str) or not url.strip():
        return {
            "success": False,
            "error": "Invalid URL parameter (must be non empty string)"
        }

    if len(url) > 1000:
        return {
            "success": False,
            "error": "URL exceeds maximum allowed length"
        }

    # Security: SSRF protection - validate URL before any network calls
    security_error = _validate_url_security(url)
    if security_error:
        logger.warning(f"[HUMAN] Web crawl blocked - SSRF protection: {security_error}")
        return {
            "success": False,
            "error": security_error
        }

    # Check cache first. Key includes harvest flag + dest so a non-harvest
    # preview never satisfies a harvest call (and vice versa).
    cache_key = f"{url}|h={1 if harvest else 0}|d={harvest_dest or ''}"
    cached = _CRAWL_CACHE.get(cache_key)
    if cached is not None:
        logger.info(f"[HUMAN] Web crawl cache HIT {url} harvest={bool(harvest)}")
        return {**cached, "cached": True}

    # New: YouTube detection block
    video_id = extract_youtube_video_id(url)
    if video_id:
        return {
            "success": False,
            "error": "This URL is a YouTube link. HTML crawl is not supported. Use a transcript tool instead."
        }

    # New: Reddit normalization
    url = _modify_reddit_url(url)

    cooldown = rate_limit.RATE_LIMITS.get("web", (0, 5))[1]
    if cooldown != 0:
        allowed = await rate_limit.enforce("web", {"url": url})
        if not allowed:
            info = rate_limit.get_rate_limit_info("web")
            return {
                "success": False,
                "error": f"Rate limit: 1 request per {info['cooldown']}s. Retry in {info['retry_in']}s."
            }

    logger.info(f"[HUMAN] Web crawl GET {url}")

    max_retries = 3
    backoff_base = 2
    client = _get_http_client()

    for attempt in range(max_retries):
        try:
            response = await client.get(url, timeout=timeout_seconds)
            content_bytes = await response.aread()

            logger.info(
                f"[HUMAN] Web crawl attempt {attempt+1}/{max_retries} "
                f"status={response.status_code}, bytes={len(content_bytes) if content_bytes else 0}"
            )

            is_pdf = _is_pdf_response(response, content_bytes, url)
            effective_cap = MAX_PDF_BYTES if is_pdf else MAX_RESPONSE_BYTES

            if len(content_bytes) > effective_cap:
                return {
                    "success": False,
                    "error": f"Response too large ({len(content_bytes)} bytes). Max allowed is {effective_cap} bytes."
                }

            if response.status_code == 200:
                if is_pdf:
                    try:
                        pdf_md = await asyncio.to_thread(_pdf_bytes_to_markdown, content_bytes, url)
                    except Exception as e:
                        logger.error(f"[HUMAN] PDF extraction failed for {url}: {e}")
                        return {
                            "success": False,
                            "error": f"PDF extraction failed: {e}",
                            "url": url,
                        }

                    if not pdf_md or len(pdf_md) < 200:
                        return {
                            "success": False,
                            "error": (
                                f"PDF extracted <200 chars (likely image-only/scanned). "
                                f"bytes={len(content_bytes)}"
                            ),
                            "url": url,
                        }

                    if harvest and harvest_dest:
                        from tools.harvester import process as harvest_process
                        harvest_result = await harvest_process(
                            content=pdf_md,
                            url=url,
                            dest=harvest_dest,
                            format="md",
                            source_tool="web",
                        )
                        if harvest_result.get("success"):
                            _CRAWL_CACHE.set(cache_key, harvest_result)
                        return harvest_result

                    # Non-harvest PDF: return truncated body + metadata (don't burn context)
                    preview_cap = 5000 if summary else 15000
                    preview = pdf_md[:preview_cap]
                    if len(pdf_md) > preview_cap:
                        preview += f"\n\n[...{len(pdf_md) - preview_cap:,} chars omitted — use harvest=True to save full PDF to disk...]"
                    result = {
                        "success": True,
                        "url": url,
                        "content": preview,
                        "content_length": len(pdf_md),
                        "content_type": "pdf",
                        "timestamp": time.time(),
                    }
                    _CRAWL_CACHE.set(cache_key, result)
                    return result

                raw_content = content_bytes.decode(
                    response.encoding or response.charset or "utf-8",
                    errors="replace",
                )

                if raw_content.strip():
                    # Harvest mode: save raw HTML to disk, skip extraction
                    if harvest and harvest_dest:
                        from tools.harvester import process as harvest_process
                        harvest_result = await harvest_process(
                            content=raw_content,
                            url=url,
                            dest=harvest_dest,
                            format="md",
                            source_tool="web",
                        )
                        if harvest_result.get("success"):
                            _CRAWL_CACHE.set(cache_key, harvest_result)
                        return harvest_result

                    clean_content, quality = await asyncio.to_thread(
                        _extract_content, raw_content, url, summary
                    )

                    challenge = _detect_js_challenge(raw_content, clean_content)
                    if challenge:
                        logger.info(f"[HUMAN] Web crawl JS challenge detected on {url}: {challenge}")
                        return {
                            "success": False,
                            "error": (
                                f"Bot protection detected ({challenge}). "
                                f"Page content is behind a JS challenge, not real article text. "
                                f"Use the fx tool to bypass: fx(action='navigate', url='{url}') then fx(action='text'). "
                                f"Remember to fx(action='close') when done."
                            ),
                            "url": url,
                        }

                    raw_len = len(raw_content)
                    clean_len = len(clean_content)
                    if quality != "good" and raw_len > 5000 and clean_len < 1000:
                        return {
                            "success": False,
                            "error": (
                                f"Page returned {raw_len:,} bytes of HTML but no article content detected "
                                f"({clean_len} chars of nav/boilerplate only). "
                                "Likely JS-rendered \u2014 article content is loaded client-side. "
                                "Use fx tool to bypass: fx(action='navigate', url='...') then fx(action='text'). "
                                "Remember to fx(action='close') when done."
                            ),
                            "url": url,
                        }

                    result = {
                        "success": True,
                        "url": url,
                        "content": clean_content,
                        "content_length": clean_len,
                        "timestamp": time.time()
                    }
                    _CRAWL_CACHE.set(cache_key, result)
                    return result
                else:
                    return {
                        "success": False,
                        "error": "Empty content received — This may be bot protection serving a challenge page. Try the fx tool to bypass: fx(action='navigate', url='...') then fx(action='text'). Remember to fx(action='close') when done."
                    }
            else:
                # Non-200 status code - handle appropriately
                logger.warning(f"[HUMAN] Web crawl got HTTP {response.status_code} on attempt {attempt+1}")
                if response.status_code >= 500:
                    # Server errors - retry
                    if attempt < max_retries - 1:
                        sleep_time = backoff_base ** attempt
                        await asyncio.sleep(sleep_time)
                        continue
                # Client errors (4xx) or final attempt - return error
                hint = ""
                if response.status_code in (403, 401, 429):
                    hint = " — This may be bot protection (Cloudflare, Datadome, etc). Try the fx tool to bypass: fx(action='navigate', url='...') then fx(action='text'). Remember to fx(action='close') when done."
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: Server returned non-success status{hint}"
                }

        except httpx.TimeoutException as e:
            logger.error(f"[HUMAN] Web crawl TIMEOUT attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff_base ** attempt
                await asyncio.sleep(sleep_time)
                continue
            return {
                "success": False,
                "error": f"Timeout after {max_retries} attempts: {str(e)}"
            }

        except httpx.RequestError as e:
            logger.error(f"[HUMAN] Web crawl HTTP ERROR attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff_base ** attempt
                await asyncio.sleep(sleep_time)
                continue
            return {
                "success": False,
                "error": f"HTTP request failed after {max_retries} attempts: {str(e)}"
            }

        except Exception as e:
            logger.error(f"[HUMAN] Web crawl UNEXPECTED ERROR attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff_base ** attempt
                await asyncio.sleep(sleep_time)
                continue
            return {
                "success": False,
                "error": f"Failed after {max_retries} attempts: {str(e)}"
            }

    return {
        "success": False,
        "error": "Unknown failure during crawl operation"
    }


logger.debug("Web crawler module loaded successfully")
