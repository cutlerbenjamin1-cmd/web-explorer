"""
Harvester module - saves fetched web content server-side, returns metadata only.

Reduces context waste by processing and saving content without passing it
back through the AI's context window.

Usage:
    from tools.harvester import process
    result = await process(content, url, "anthropic_docs/", format="md", source_tool="web")
"""
import asyncio
import html
import logging
import os
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Base path for harvested content
HARVEST_BASE = Path(os.getenv("HARVEST_ROOT", str(Path(__file__).resolve().parent.parent / "harvested")))


# =============================================================================
# Site-Specific Content Selectors
# =============================================================================
# Maps domain patterns to CSS-style selectors for main content extraction.
# Strips sidebar/nav chrome before conversion. Uses regex on raw HTML.

SITE_SELECTORS = {
    # ChatGPT - uses <main> for content, strip UI chrome after conversion
    "chatgpt.com": {
        "strip_patterns": [
            r'<nav[^>]*>.*?</nav>',  # Navigation
            r'<aside[^>]*>.*?</aside>',  # Sidebars
        ],
        "extract_pattern": r'<main[^>]*>(.*?)</main>',
        # Post-conversion text patterns to strip (applied to markdown output)
        "post_strip": [
            r'^ChatGPT[^\n]*Share[^\n]*\n+',  # "ChatGPT 5.1 Share Add people" header
            r'^Skip to content[^\n]*\n+',
        ],
    },
    # Anthropic docs
    "docs.anthropic.com": {
        "extract_pattern": r'<main[^>]*>(.*?)</main>',
        "strip_patterns": [r'<nav[^>]*>.*?</nav>'],
    },
    # OpenAI platform docs
    "platform.openai.com": {
        "extract_pattern": r'<main[^>]*>(.*?)</main>',
        "strip_patterns": [r'<nav[^>]*>.*?</nav>', r'<aside[^>]*>.*?</aside>'],
    },
    # GitHub
    "github.com": {
        "extract_pattern": r'<article[^>]*class="[^"]*markdown-body[^"]*"[^>]*>(.*?)</article>',
        "strip_patterns": [],
    },
    # LangChain docs
    "python.langchain.com": {
        "extract_pattern": r'<article[^>]*>(.*?)</article>',
        "strip_patterns": [r'<nav[^>]*>.*?</nav>'],
    },
    # Generic fallback patterns (tried in order)
    "_fallback": {
        "extract_patterns": [
            r'<main[^>]*>(.*?)</main>',
            r'<article[^>]*>(.*?)</article>',
            r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>',
            r'<div[^>]*id="content"[^>]*>(.*?)</div>',
        ],
        "strip_patterns": [
            r'<nav[^>]*>.*?</nav>',
            r'<aside[^>]*>.*?</aside>',
            r'<footer[^>]*>.*?</footer>',
            r'<header[^>]*>.*?</header>',
        ],
    },
}


def extract_main_content(html_content: str, url: str) -> str:
    """
    Extract main content from HTML using site-specific selectors.
    Falls back to full content if no selector matches.
    """
    from urllib.parse import urlparse
    
    domain = urlparse(url).netloc.lower()
    # Remove www. prefix
    if domain.startswith("www."):
        domain = domain[4:]
    
    # Find matching selector config
    config = None
    for site_pattern, site_config in SITE_SELECTORS.items():
        if site_pattern == "_fallback":
            continue
        if site_pattern in domain:
            config = site_config
            logger.debug(f"Using selector for {site_pattern}")
            break
    
    # Apply strip patterns first (remove nav/sidebar chrome)
    working_content = html_content
    strip_patterns = []
    
    if config:
        strip_patterns = config.get("strip_patterns", [])
    else:
        # Use fallback strip patterns
        strip_patterns = SITE_SELECTORS["_fallback"]["strip_patterns"]
    
    for pattern in strip_patterns:
        working_content = re.sub(pattern, "", working_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Try to extract main content
    if config and "extract_pattern" in config:
        match = re.search(config["extract_pattern"], working_content, re.DOTALL | re.IGNORECASE)
        if match:
            # For ChatGPT, we want ALL article matches, not just the first
            if "chatgpt" in domain:
                articles = re.findall(config["extract_pattern"], working_content, re.DOTALL | re.IGNORECASE)
                if articles:
                    logger.info(f"Extracted {len(articles)} conversation messages")
                    return "\n\n".join(articles)
            else:
                logger.info(f"Extracted main content using site selector")
                return match.group(1) if match.lastindex else match.group(0)
    
    # Try fallback patterns
    for pattern in SITE_SELECTORS["_fallback"]["extract_patterns"]:
        match = re.search(pattern, working_content, re.DOTALL | re.IGNORECASE)
        if match:
            logger.info(f"Extracted main content using fallback pattern")
            return match.group(1) if match.lastindex else match.group(0)
    
    # No extraction worked, return stripped content
    logger.debug(f"No content selector matched, using full content")
    return working_content


# =============================================================================
# HTML to Markdown Converter (adapted from scripts/convert_devdocs.py)
# =============================================================================

class HTMLToMarkdown(HTMLParser):
    """Convert HTML to Markdown, preserving code blocks and structure."""
    
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.in_pre = False
        self.in_code = False
        self.in_script = False  # Skip script content
        self.in_style = False   # Skip style content
        self.title = None
        self.in_title = False
        self.current_href = None  # Track href for anchor tags
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Skip script/style blocks entirely
        if tag == "script":
            self.in_script = True
            return
        elif tag == "style":
            self.in_style = True
            return
        
        if tag == "title":
            self.in_title = True
        elif tag == "pre":
            self.in_pre = True
            # Check for language hint
            lang = attrs_dict.get("class", "").replace("language-", "").split()[0] if attrs_dict.get("class") else ""
            self.text_parts.append(f"\n```{lang}\n")
        elif tag == "code" and not self.in_pre:
            self.in_code = True
            self.text_parts.append("`")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            self.text_parts.append("\n" + "#" * level + " ")
        elif tag == "p":
            self.text_parts.append("\n\n")
        elif tag == "br":
            self.text_parts.append("\n")
        elif tag == "li":
            self.text_parts.append("\n- ")
        elif tag == "a":
            href = attrs_dict.get("href", "")
            if href and not href.startswith("#"):
                self.current_href = href
                self.text_parts.append("[")
            else:
                self.current_href = None
        elif tag == "strong" or tag == "b":
            self.text_parts.append("**")
        elif tag == "em" or tag == "i":
            self.text_parts.append("*")
        elif tag == "blockquote":
            self.text_parts.append("\n> ")
            
    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False
            return
        elif tag == "style":
            self.in_style = False
            return
        
        if tag == "title":
            self.in_title = False
        elif tag == "pre":
            self.in_pre = False
            self.text_parts.append("\n```\n")
        elif tag == "code" and not self.in_pre:
            self.in_code = False
            self.text_parts.append("`")
        elif tag == "a":
            if self.current_href:
                self.text_parts.append(f"]({self.current_href})")
                self.current_href = None
            # else: no href tracked, don't emit broken link
        elif tag == "strong" or tag == "b":
            self.text_parts.append("**")
        elif tag == "em" or tag == "i":
            self.text_parts.append("*")
            
    def handle_data(self, data):
        # Skip script/style content
        if self.in_script or self.in_style:
            return
        
        if self.in_title and not self.title:
            self.title = data.strip()
        elif self.in_pre:
            self.text_parts.append(data)
        else:
            # Clean whitespace for non-pre content
            cleaned = " ".join(data.split())
            if cleaned:
                self.text_parts.append(cleaned + " ")
                
    def get_markdown(self) -> str:
        result = "".join(self.text_parts)
        # Clean up excessive newlines
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()
    
    def get_title(self) -> Optional[str]:
        return self.title


def html_to_markdown(content: str, url: str = "") -> tuple[str, Optional[str]]:
    """
    Convert HTML content to Markdown.
    
    Args:
        content: Raw HTML content
        url: Source URL (used for site-specific content extraction)
    
    Returns:
        Tuple of (markdown_content, title)
    """
    # Extract main content first if URL provided
    config = None
    if url:
        content = extract_main_content(content, url)
        # Get config for post-processing
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        for site_pattern, site_config in SITE_SELECTORS.items():
            if site_pattern != "_fallback" and site_pattern in domain:
                config = site_config
                break
    
    content = html.unescape(content)
    parser = HTMLToMarkdown()
    try:
        parser.feed(content)
        result = parser.get_markdown()
        
        # Apply post-strip patterns if configured
        if config and "post_strip" in config:
            for pattern in config["post_strip"]:
                result = re.sub(pattern, "", result, flags=re.MULTILINE)
        
        return result.strip(), parser.get_title()
    except Exception as e:
        logger.warning(f"HTML parsing failed, falling back to regex strip: {e}")
        # Fallback: strip all tags
        text = re.sub(r"<[^>]+>", "", content)
        return text.strip(), None


def extract_title(content: str, format: str) -> Optional[str]:
    """Extract title from content based on format."""
    if format == "html":
        # Try <title> tag
        match = re.search(r"<title[^>]*>([^<]+)</title>", content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Try first <h1>
        match = re.search(r"<h1[^>]*>([^<]+)</h1>", content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    elif format == "md":
        # First # header
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
    elif format == "txt":
        # First non-empty line
        for line in content.split("\n"):
            line = line.strip()
            if line:
                return line[:100]
    return None


def url_to_slug(url: str, max_len: int = 100) -> str:
    """Convert URL to a safe filename slug. Includes netloc to avoid collisions."""
    parsed = urlparse(url)
    # Include netloc to avoid collisions across hosts
    netloc = parsed.netloc.replace(".", "_").replace(":", "_")
    path = parsed.path.strip("/")
    
    if path:
        # Combine netloc and path
        combined = f"{netloc}__{path}" if netloc else path
    else:
        combined = netloc or "index"
    
    # Replace separators with underscores
    slug = re.sub(r"[/\\]", "_", combined)
    # Remove or replace unsafe chars
    slug = re.sub(r"[^a-zA-Z0-9_\-.]", "_", slug)
    # Collapse multiple underscores
    slug = re.sub(r"_+", "_", slug)
    # Truncate
    if len(slug) > max_len:
        slug = slug[:max_len]
    
    return slug.strip("_") or "index"


# =============================================================================
# Main Process Function
# =============================================================================

async def process(
    content: str,
    url: str,
    dest: str,
    format: str = "md",
    source_tool: str = "unknown",
    filename: Optional[str] = None,
) -> dict:
    """
    Process and save harvested content.
    
    Args:
        content: Raw content (HTML, text, etc.)
        url: Source URL
        dest: Destination subfolder under HARVEST_BASE (e.g., "anthropic_docs/")
        format: Output format - "md" (convert HTML), "html" (save as-is), "txt"
        source_tool: Which tool fetched this ("web", "fx", "github")
        filename: Optional explicit filename (without extension)
    
    Returns:
        Dict with: success, saved_to, title, size_bytes, source_url, source_tool
    """
    try:
        # Validate dest - no path traversal
        if ".." in dest or dest.startswith("/") or dest.startswith("\\"):
            return {
                "success": False,
                "error": f"Invalid dest path: {dest}"
            }
        
        # Determine output directory
        dest = dest.strip("/\\")
        output_dir = HARVEST_BASE / dest
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        if filename:
            slug = re.sub(r"[^a-zA-Z0-9_\-.]", "_", filename)
        else:
            slug = url_to_slug(url)
        
        # Determine extension
        ext_map = {"md": ".md", "html": ".html", "txt": ".txt"}
        ext = ext_map.get(format, ".txt")
        
        # Process content based on format
        title = None
        output_content = content
        
        if format == "md":
            # Detect if content is HTML
            if "<html" in content.lower() or "<body" in content.lower() or "<div" in content.lower():
                output_content, title = html_to_markdown(content, url)
            else:
                # Already markdown or plain text
                title = extract_title(content, "md")
                output_content = content
        elif format == "html":
            title = extract_title(content, "html")
            output_content = content
        else:  # txt
            # Strip HTML if present
            if "<" in content and ">" in content:
                output_content = re.sub(r"<[^>]+>", "", content)
            title = extract_title(output_content, "txt")
        
        # Fallback title from URL
        if not title:
            title = slug.replace("_", " ").title()
        
        # Write file
        output_path = output_dir / f"{slug}{ext}"
        
        # Add header with metadata
        header = f"---\nsource: {url}\nharvested: {time.strftime('%Y-%m-%d %H:%M:%S')}\ntool: {source_tool}\n---\n\n"
        final_content = header + output_content
        
        # Write with UTF-8
        await asyncio.to_thread(
            output_path.write_text,
            final_content,
            encoding="utf-8",
            errors="replace"
        )
        
        size_bytes = len(final_content.encode("utf-8", errors="replace"))
        
        logger.info(f"[HARVEST] Saved {size_bytes} bytes to {output_path}")
        
        return {
            "success": True,
            "saved_to": str(output_path),
            "title": title,
            "size_bytes": size_bytes,
            "source_url": url,
            "source_tool": source_tool,
            "format": format,
        }
        
    except Exception as e:
        logger.exception(f"Harvest failed for {url}")
        return {
            "success": False,
            "error": str(e),
            "source_url": url,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

async def harvest_html(content: str, url: str, dest: str, source_tool: str = "unknown") -> dict:
    """Harvest HTML content, converting to Markdown."""
    return await process(content, url, dest, format="md", source_tool=source_tool)


async def harvest_raw(content: str, url: str, dest: str, ext: str = "txt", source_tool: str = "unknown") -> dict:
    """Harvest raw content without conversion."""
    format_map = {".md": "md", ".html": "html", ".htm": "html"}
    fmt = format_map.get(ext.lower(), "txt")
    return await process(content, url, dest, format=fmt, source_tool=source_tool)
