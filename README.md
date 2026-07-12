# web-explorer

External web tools for any MCP client. Search the web, crawl pages, drive an
anti-detect Firefox browser past bot walls, and pull structured data from
Wikipedia, Wikidata, GitHub, and YouTube.

Two things worth stating up front:

- **Free APIs only.** No paid services, no metered quotas. It uses DuckDuckGo,
  Wikipedia, Wikidata, the public GitHub API, and YouTube's own caption
  endpoints. The only credential any tool reads is a GitHub token, which is free
  and which only `github` uses (for code search and issue actions; plain file
  and repo reads work without one). See Credentials below.
- **Runs off your own IP.** Every request goes out from the machine running the
  server, straight to the source. No proxy, no relay, no third-party scraping
  service in the middle reselling your traffic.

## Tools

| Tool | What it does |
|------|--------------|
| `web` | DuckDuckGo search and direct page crawl with readable-text extraction. News search too. |
| `fx` | Anti-detect Firefox (Camoufox) for bot-protected sites (Cloudflare, Datadome, verification walls). Persistent profile keeps cookies and solved challenges across restarts. Optional install, see below. |
| `wiki` | Wikipedia search and full-article fetch. |
| `wikidata_query` | Wikidata SPARQL queries for structured entity data (people, places, orgs, relationships). |
| `github` | Fetch files, analyze repo structure, search code, and read or manage issues via the public GitHub API. Code search and issue actions need a free `GITHUB_TOKEN`; file and repo reads do not. |
| `youtube` | Transcripts and metadata for any video. youtube-transcript-api first, yt-dlp fallback. Any URL shape or bare video id. |

Every tool takes an `action` argument, except `wikidata_query` which takes a
SPARQL `query`. Call `list_tools` from your client for the exact schema of every
action and parameter. What each action does:

### `web`

- `search` - web results as title, url, and snippet. `news=true` switches to the
  DuckDuckGo news endpoint (adds date and source per result); `timelimit` filters
  to `d`, `w`, or `m`.
- `crawl` - fetch one URL and extract readable article text (trafilatura, with a
  BeautifulSoup fallback), including PDFs. `summary=true` (default) returns a
  short version, `summary=false` the fuller text.

Results are cached for 20 minutes. If a page is behind bot protection, `crawl`
points you at `fx`.

### `fx` (optional install)

A real Camoufox (hardened Firefox) session for sites that block plain fetches.

- `read` - navigate, wait out JS challenges, extract clean text, auto-close. The
  common one-shot case.
- Step-by-step automation: `navigate`, `click`, `fill`, `type`, `press`,
  `scroll`, `wait`, `extract` (raw HTML), `text` (body text), `evaluate` (run
  JavaScript), `screenshot` (element or `full_page`), `cookies` / `add_cookies`,
  and `close`.
- `status` - report the current session.

Options include `visible` (show the window), `stealth` (humanized cursor), and
`load_images`. The profile persists to disk; see the note under Safety.

### `wiki`

- `search` - article titles matching a query.
- `page` - full article text for a title.

Cached for one hour.

### `wikidata_query`

Run a read-only SPARQL query against Wikidata for structured entity data. Limits:
5000 character query, 15 second timeout, 1000 results. Update operations (INSERT,
DELETE, DROP, and so on) are rejected.

### `github`

- `fetch` - raw file content. `analyze` - repo metadata, languages, file count.
- `search` - code search (requires a token).
- Issues: `get_issue`, `list_issues`, `create_issue`, `comment_issue`,
  `update_issue`. The write actions require a token. See Credentials.

Cached for 10 minutes.

### `youtube`

- `transcript` - captions only. `metadata` - title, channel, duration,
  description, chapters. `info` - both at once (the common case).
- `list_languages` - available caption tracks. `search` - find a phrase within a
  transcript, returning timestamped matches with `t=` deep links.

Transcript `format` can be `text` (default), `timestamped`, `srt`, `vtt`, or
`json`. Long transcripts auto-paginate with `segment_offset` and `max_segments`.

### Saving context: `harvest=True`

`web`, `fx`, `github`, and `youtube` accept `harvest=True`. Instead of returning
a large page or transcript into your context window, the content is written to
`harvested/<harvest_dest>/` as clean markdown and only the file path plus a short
summary come back. Point `HARVEST_ROOT` at a different directory if you want.

## Install

Python 3.10 or newer.

```
pip install -r requirements.txt
```

That covers the server and the five free-API tools (`web`, `wiki`,
`wikidata_query`, `github`, `youtube`).

### Optional: the `fx` browser

The anti-detect browser is a heavier dependency, so it is opt-in. The server and
the other tools work fine without it; `fx` just returns a clear "not installed"
message until you add it.

```
pip install -r requirements-fx.txt
python -m camoufox fetch
```

`python -m camoufox fetch` is a one-time download of a patched Firefox build
(about 150 MB). camoufox also pulls in playwright and browserforge.

## Use it with an MCP client

Speaks MCP over stdio. Point your client at `server.py`:

```json
{
  "mcpServers": {
    "web-explorer": {
      "command": "python",
      "args": ["/absolute/path/to/web-explorer/server.py"]
    }
  }
}
```

## Running over HTTP (optional)

To serve this over MCP streamable HTTP instead of stdio, use
[mcp-http-bridge](https://github.com/cutlerbenjamin1-cmd/mcp-http-bridge): run it
from this directory and point your client at the bridge. Keep the bridge's tool
filter tight if you expose it beyond localhost, since `fx` can drive a real
browser.

## Safety

- **Untrusted-content banner.** Everything these tools return is data from
  outside your machine. Each response is tagged as external and untrusted so an
  agent does not treat page content as instructions.
- **SSRF protection.** `web(crawl)` refuses private and loopback address ranges,
  so a crawled URL cannot be pointed back at your internal network.
- **Polite rate limiting.** Per-tool cooldowns space out requests so you do not
  hammer a host from your own IP and trip a 429.
- **The `fx` profile is real browser state.** `fx` keeps a persistent Firefox
  profile at `data/fx_profile/`. Anything you do in it sticks: if you sign into a
  site through `fx`, that session and its cookies are written to disk there in the
  clear, exactly like a normal browser profile. That is what lets logins and
  solved Cloudflare challenges survive a restart. The folder is gitignored so it
  will not be committed, but treat it like a password store: do not share or copy
  it, and delete `data/fx_profile/` to wipe every saved session.

## Credentials

Only the `github` tool reads a credential, and it is an ordinary free GitHub
personal access token.

- Without a token: `github` file fetch and repo analyze work, subject to
  GitHub's anonymous rate limit (about 60 requests per hour).
- With a token: that limit rises to 5000 per hour, and `github` code search plus
  the issue actions (create, comment, update) become available. They return a
  clear "token required" error otherwise.

Set it in the environment before launching the server:

```
set GITHUB_TOKEN=ghp_your_token_here        # Windows
export GITHUB_TOKEN=ghp_your_token_here      # macOS / Linux
```

No other tool needs an account or key. DuckDuckGo, Wikipedia, Wikidata, and
YouTube are all used unauthenticated.

## Environment variables

Every variable below is optional.

| Variable | Effect |
|----------|--------|
| `GITHUB_TOKEN` | GitHub auth. See Credentials above. |
| `HARVEST_ROOT` | Directory for `harvest=True` output. Default: `./harvested`. |
| `YOUTUBE_COOKIES_BROWSER` | Browser name (for example `firefox`) so yt-dlp can borrow its cookies when a video is IP-blocked. |
| `YOUTUBE_COOKIES_PROFILE` | Browser profile path for the cookies above. |
| `YOUTUBE_BIND_IP` | Local source IP to bind outbound YouTube requests to. |
| `MCP_DEBUG` | Set to `true` for stderr debug logging. |
| `OUTPUT_MAX_CHARS` | Hard cap on a single tool response. Default: 15000. |
| `GITHUB_MAX_RESPONSE_BYTES` | Max GitHub file size fetched. Default: about 2MB. |
| `WEB_CRAWL_MAX_BYTES` | Max HTML or text size for `web(crawl)`. Default: about 2MB. |
| `WEB_CRAWL_MAX_PDF_BYTES` | Max PDF size for `web(crawl)`. Default: about 50MB. |

## License

MIT. See LICENSE.
