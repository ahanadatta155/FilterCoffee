# Nemotron Scout

Describe what you want in plain English and get a verified table from one public web
page. Scout can resolve a named site or general topic to a single page before scraping.

## Current stage: one-page discovery and extraction

Implemented: configuration for Nebius, OpenRouter, NVIDIA, TinyFish, and optional Tavily;
model discovery;
an OpenAI-compatible model adapter; key/model CLI diagnostics; and Pydantic v2
contracts. The adapter handles retries, JSON parsing, model fallback, and per-call
token/cost/latency accounting. Nemotron classifies a free-form request, TinyFish Search
locates candidate pages, and the extraction pipeline reads one selected page. YouTube
requests use yt-dlp for video search, metadata, and available subtitles.

Also implemented: guarded HTTP fetching, robots rules, rate limiting, HTML cleaning,
JSON-LD/tables, repeated-item detection, and bounded page representations. The product
fixture contains 12 cards; article, JavaScript shell, canvas, blocked, malformed,
empty, Unicode, and JSON-LD fixtures are included.

Implemented core API: automatic schema planning, evidence-backed extraction,
normalization/validation, one repair pass, SQLite page/result cache, and an optional
Playwright Chromium fallback for JavaScript-rendered pages. Canvas or nearly empty
rendered pages can use screenshot-based Nemotron vision. Tests use real local Chromium
and HTML parsing plus mocked HTTP/model responses. A live 20-book extraction from
Books to Scrape was verified with the OpenRouter free model (one call, reported cost
$0.00).

OpenRouter's free Nemotron IDs were verified against its live `/models` endpoint:
`nvidia/nemotron-3.5-lightning:free`, `nvidia/nemotron-3.5-super-vl:free`, and
`nvidia/nemotron-3-ultra-550b-a55b:free`. A minimal free chat check returned HTTP
200 at zero cost. Nebius and NVIDIA model IDs remain configuration requests until
the startup resolver checks them against the selected provider.

## Setup (PowerShell)

Python 3.12 or 3.13 and `uv` are required. The project currently uses Python 3.13.

```powershell
uv sync
# Only if .env does not exist yet:
Copy-Item .env.example .env
# Edit your existing .env: set OPENROUTER_KEY, GEMINI_KEY, and TINYFISH_API_KEY.
uv run python scripts/check_env.py
uv run python -c "import scout; print('ok')"
```

For JavaScript pages, install Chromium once with `uv run playwright install chromium`
or double-click `install_browser.bat`. This downloads browser binaries (a few hundred
MB); Node.js/npm are not required.

`check_env.py` only prints whether each key is present or absent. It does not
authenticate with any provider. It exits with code 1 when all model-provider keys are
absent; a search key alone cannot run the extraction model.
Never paste API keys into terminal commands or commit `.env`.

Without `uv`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e . pytest respx ruff
.venv\Scripts\python scripts/check_env.py
.venv\Scripts\python -m pytest -q
```

## Configuration

The project-root `.env` is loaded with `python-dotenv`; existing shell variables
take precedence. `load_settings(env_file=...)` also accepts an explicit path.
`SCOUT_PROVIDER` accepts `auto`, `nebius`, `openrouter`, or `nvidia`. In auto mode,
the priority is Nebius, OpenRouter, then NVIDIA. An explicit provider requires its
own key and **never** falls back to another provider.

For free development, use `SCOUT_PROVIDER=openrouter` and `OPENROUTER_KEY` (or
`OPENROUTER_API_KEY`). This avoids choosing Nebius if its key is later added.
Missing credentials produce an actionable error with provider links. Do not
copy `.env.example` over an existing `.env` containing your key.

All model IDs, base URLs, and the user agent can be overridden using the variables
in `.env.example`. Provider-specific model defaults apply unless a model override
is set. Localhost access and Ultra default to disabled; their flags accept only
`0` or `1`. Configuration loading makes no network requests. API keys are held
as masked `SecretStr` values.

## Model discovery and adapter

```powershell
uv run python -m scout models
# Equivalent diagnostic:
uv run python scripts/list_models.py
```

These commands query `/models` only; they do not run inference. Exact IDs are
resolved case-insensitively, then by family (`lightning`/`nano`, `super-vl`/`super`,
`ultra`). Missing Ultra is nonfatal while disabled. The mapping, capabilities,
and price table are printed without keys.

OpenRouter is **free-only** in this version: overrides must end in `:free`, and
discovery filters out models with missing, invalid, or nonzero advertised prices,
including request/image fees. Neither model nor provider fallback can select a
paid OpenRouter model. An unexpected reported charge is recorded and stops further
requests on that client; this is a software safeguard, not a provider billing
guarantee. Free endpoints can still be rate-limited or unavailable. Check your
provider's privacy settings before sending page content; use public demo data.

| Role | OpenRouter default | Advertised input/output per 1M tokens |
|---|---|---|
| Fast | `nvidia/nemotron-3.5-lightning:free` | $0 / $0 |
| Strong/vision | `nvidia/nemotron-3.5-super-vl:free` | $0 / $0 |
| Optional Ultra | `nvidia/nemotron-3-ultra-550b-a55b:free` | $0 / $0 |

The adapter sends JSON mode only when supported or capabilities are unknown; an
unsupported-format 400 gets one retry without it. Lightning/Ultra free variants
do not advertise JSON mode. Model-not-found gets at most one same-provider model
fallback; vision never falls back to a known text-only model. Ultra must be
explicitly enabled and requested; automatic escalation belongs to the future pipeline.

429, 5xx, and connection errors get three retries (four attempts total) with
1/2/4-second backoff. SDK automatic retries are disabled to avoid multiplied
requests. A 402 stops immediately with `budget exhausted`. Every inference attempt
is recorded, including failed calls and truncated output. Failed HTTP requests
without usage are recorded as zero with an explicit unknown-usage warning, not
treated as proven free. Missing usage is estimated at roughly one token per four
characters and labeled; images and hidden reasoning cannot be estimated accurately.

Reported provider cost is preferred. Nebius fallback estimates use the spec's
unverified $0.06/$0.24, $0.30/$0.90, and $1/$3 per-million input/output rates for
the exact configured model IDs; verify these before presenting. Unknown prices
are labeled unknown rather than presented as confirmed zero billing.

`scout.llm` exposes the injectable `LLMClient` protocol, `LLMResponse`, `ModelClient`,
and `parse_json`. `ModelClient.chat(stage, messages, role="fast", max_tokens=4096)`
returns text and usage; `client.calls` contains per-attempt telemetry. Reasoning
models may consume output tokens before producing visible JSON. Empty or
length-truncated output raises a friendly error without hiding its token cost.

## Contracts

`scout.contracts` defines `FieldSpec`, `ExtractionPlan`, `Evidence`, `ExtractedItem`,
`ModelCall`, `RunMeta`, and `ExtractResult`. Plans require 1–12 uniquely named
snake_case fields. Supported field types: `string`, `number`, `integer`, `boolean`,
`date`, `url`, and `list[string]`. Confidence values must be finite and within
0–1; token counts, costs, and durations cannot be negative.

These contracts validate shape, not factual accuracy. The separate extraction
validator checks evidence and computes deterministic confidence scores.

## Page ingestion

`RunOptions(fetch_backend="auto")` selects TinyFish Fetch when `TINYFISH_API_KEY`
is set; otherwise it uses the local guarded HTTP/Playwright path. YouTube URLs are
handled by yt-dlp in either case. `fetch_backend="http"` explicitly selects the local
page reader, and `fetch_backend="tinyfish"` requires the TinyFish key. `force_browser`
selects local HTTP/Chromium and cannot be combined with explicit TinyFish fetching.
Injected test fetchers retain their existing behavior.

TinyFish receives the public URL and returns extracted HTML, not the full original
DOM. Scout verifies quotes against that returned content and labels the source
`tinyfish`. It does not independently verify TinyFish's server-side robots handling.
Original and final URLs are checked for unsafe addresses. The local HTTP robots
check described below applies to the local reader, not TinyFish or yt-dlp.
No blocked request automatically switches to a different provider or browser.

```text
URL -> DNS/address checks -> robots.txt -> throttled HTTP + manual redirects
    -> cleaned text / JSON-LD / tables -> repeated blocks -> compact representation
    -> plan -> extract -> verify -> repair once -> result/cache
```

`scout.fetch.Fetcher` is a context manager with `fetch(url, ignore_robots=False)`.
The returned `FetchedPage` includes the final URL, HTML, status, and warnings.
It does not require any model credentials. A custom HTTP client/resolver can be
injected for offline tests; production defaults use a DNS-pinning transport.

Only HTTP(S) URLs without credentials are allowed. Private, link-local, multicast,
reserved, metadata, and mixed public/private DNS answers are rejected. Connections
use a validated IP with the original HTTP Host and TLS hostname; system proxies
are disabled. Every redirect is validated (maximum five), including robots redirects.
Robots rules are checked at each page hop and cached per origin for one hour.
404 means allowed, 401/403 means denied, and transient robots failures fail closed.
The process-wide default limiter permits one request/second/domain, including
robots and redirects. HTTP timeout is 20 seconds; advertised body size and streamed
decoded bytes are capped at 5 MiB. Non-HTML content returns a friendly error.

`allow_localhost=True` on `Fetcher` allows literal loopback IPs or localhost names
resolving only to loopback, not arbitrary private networks or external hostnames
resolving to local addresses. This generates a warning. `ignore_robots=True` also
generates a warning and is intended only for sites you own. A 403 stops with a
blocked-site error; a 429 requests a wait. Neither is used to attempt a bot bypass.

`scout.clean.clean_page(html, url)` extracts page signals, JSON-LD, tables, article
text, and normalized source text. Script/style/SVG/noscript/template and explicitly
hidden content are removed. Navigation/footer content is excluded from candidate
groups. Safe links, labels, and other evidence-bearing attributes are retained in
source text. CSS visibility detection is heuristic, not a rendered-browser guarantee.

`scout.represent.find_candidate_groups(page)` groups siblings by tag and sorted
class signature, requires at least three similarly sized blocks, scores text,
counts, links, prices, and images, and retains the top three groups in stable order.
`build_representation(page, max_chars=60000, max_items=100)` prioritizes JSON-LD,
tables, groups, then main text; it never sends full HTML. A 120000-character cap
can be supplied for stronger models. These are structural heuristics, not extracted
or verified model results.

Short scripted/app-root pages, JavaScript noscript messages, and canvas-only pages
are marked for rendering. When enabled, the Playwright fallback runs Chromium in a
fresh headless context, with bounded navigation and a loopback proxy that validates
and pins outbound IP addresses. Unsafe/private destinations are blocked, non-GET/HEAD
browser requests are denied, and service workers are disabled. The initial
URL is checked with the existing robots rules before the browser starts. Blocked-page
detection is heuristic; CAPTCHA, login, paywall, and bot-bypass behavior is unsupported.
Only the loaded page is rendered; automatic pagination and infinite scrolling are
not performed.

Canvas-only or nearly empty rendered pages use a bounded in-memory screenshot. It is
sent to the configured Nemotron vision model (OpenRouter free model in the runner),
never to Gemini. Screenshot-derived values have a lower confidence cap and are marked
with screenshot evidence; they are not DOM-verifiable and should be checked manually.
Screenshots are not written to disk. Normal text extraction still uses Gemini.

## Python extraction API

```python
from scout import extract

result = extract(
    "https://books.toscrape.com/",
    "Extract each product's name, price, rating, and stock status.",
)
for item in result.items:
    print(item.data, item.confidence)
    print([(e.field, e.quote) for e in item.evidence])
```

Optional schema input accepts either `{"fields": [...]}` or standard JSON Schema
with `properties` and `required`. JSON Schema conversion supports string, number,
integer, boolean, and arrays of strings. Without a schema, the configured model
proposes 1–12 fields from page signals. The call returns `ExtractResult` even on
configuration, fetch, planning, or model failure; inspect `result.meta.warnings`.

Each model item is normalized against its field type; evidence quotes are checked
against normalized page text and then checked for value support. Missing or
unverified values become `null`; deterministic confidence is blended with model
confidence. List items are deduplicated by their first three non-null values. A
required-field, type, evidence, or low-confidence issue triggers at most one
strong-model repair; repair output merges by item index so omitted valid fields
survive. Screenshot-derived values are checked against their visual evidence quotes,
not the DOM text.

`data/cache.db` stores static, rendered, TinyFish, or YouTube source documents for 6
hours and results for 24 hours (WAL mode). Cached source documents preserve final URL,
status and warnings, including missing-subtitle notices. Page and result caches are
separated by selected source; old cache entries are not deleted.
Result identity includes URL, instruction, schema, provider/model settings, and
relevant options. `RunOptions(force_refresh=True)` bypasses reads but still writes.
The SQLite schema reserves a 7-day model-response cache, but automatic LLM response
reuse is not enabled yet. The page cache contains fetched HTML locally; treat it
as runtime data even though `data/` is gitignored.

## One-click live test (Windows)

Double-click `run_scout.bat` in the project folder. It asks what you want to find.
Nemotron interprets the request: an explicitly supplied URL is read directly, a named
website is searched with a site restriction, and a topic without a site uses general
web search. TinyFish returns candidates (Scout considers up to five); Nemotron picks one actual result
URL. Scout then scrapes only that page with the existing evidence checks. Press Enter
on an empty prompt to use the Books to Scrape sample. Gemini 3.5 Flash-Lite extracts
text; Nemotron handles planning, candidate selection, repair, and screenshot vision.
The runner prints the interpreted search, chosen URL, table, and evidence quotes, then
keeps the window open. It refuses to start if
`SCOUT_PROVIDER` selects Nebius or NVIDIA. To run non-interactively from PowerShell:

```powershell
run_scout.bat --request "Search Wikipedia for Minecraft and extract its release date."
```

Use `run_scout.bat --refresh` (or add `--refresh` to the PowerShell command) to
fetch and extract again instead of using the cached result. `--url URL --instruction
"..."` keeps direct-page mode; `--request "..."` (or `--instruction` without a URL)
uses request interpretation and URL discovery. No schema is hard-coded: Nemotron
plans fields from the request and selected page. This is terminal output, not the
Streamlit UI, which is a later stage.
The runner makes a small authenticated OpenRouter key-status request before
scraping, then prints total elapsed time, per-call token counts/model latency, and
the quota fields OpenRouter exposes. It can report free-model daily requests
(RPD) and configured credit budget, but OpenRouter does not expose current RPM or
TPM/TPD remaining counters through that endpoint. 429 responses may provide the
specific rate-limit headers and Retry-After value when a limit is actually hit.

Regular text extraction uses Gemini and requires `GEMINI_KEY` in the project `.env`.
Screenshot vision uses Nemotron and can run without a Gemini key. The runner reports
each call's model, tokens, latency, warnings, and recorded cost. Google
free-tier quotas and billing depend on the AI Studio project; verify them there.
Google may use free-tier prompts to improve its products.

General URL discovery defaults to TinyFish and needs `TINYFISH_API_KEY` in `.env`.
Direct `--url` skips search, but can still use TinyFish Fetch when the key is present.
`--search-provider tavily` explicitly selects the existing Tavily search adapter and
requires `TAVILY_API_KEY`; it is never used as an automatic fallback. Selecting a
search provider does not change the page reader: use `--fetch-backend http` for local
HTTP/Playwright or `--fetch-backend tinyfish` for TinyFish.

TinyFish Search/Fetch have published free allowances, but its pricing page and API
reference differ about over-allowance billing. Scout conservatively stops at **100
search attempts and 100 fetch attempts per key per UTC day**, persisted atomically in
`data/cache.db`. Failed calls count too; clearing cached pages/results does not reset
the counters. This is a local guard, not an account-wide billing guarantee: other
apps, copies of this database, deleted databases, or keys can consume allowances
outside it. Check your TinyFish account's free limits and disable paid overage/automatic
reload where available. Scout never calls Agent, Browser, Research, wallet funding,
or payment endpoints and does not retry allowance/rate-limit denials.

The request and search snippets go to the configured OpenRouter Nemotron model.
Only a returned candidate may be selected; its URL is checked before fetching.
**This first version still scrapes one page only**: even a broad topic returns one
selected page, not a comparison or multi-source research report. Multi-page research
is a later feature.

On the local reader, browser rendering is automatic for detected JavaScript shells.
Add `--browser` to force local Chromium, or `--fetch-backend http --no-browser` for
static-only local fetching. `--no-browser` disables local Chromium, not TinyFish's
server-side rendering. For example:

```powershell
run_scout.bat --browser --url "https://quotes.toscrape.com/js/" --instruction "Extract each quote and its author."
```

Canvas vision requires the configured Nemotron Super VL model to advertise image
input support. A missing Chromium installation produces an actionable warning; use
`install_browser.bat` and retry. Browser navigation does not bypass 403/429 responses.

### YouTube search and subtitles

```powershell
run_scout.bat --request "Find a YouTube video about Python decorators and extract its title and channel."
run_scout.bat --url "https://www.youtube.com/watch?v=jNQXAC9IVRw" --instruction "Extract the title, channel, upload date, and subtitle text."
```

A request that names YouTube uses yt-dlp search (up to five video candidates), then
reads one selected video. A direct YouTube video URL skips search. No TinyFish/Tavily
key, Node/npm, FFmpeg, browser login, or Whisper key is needed for this path. yt-dlp
runs in a subprocess with a 60-second metadata deadline, user config/plugins disabled,
and a guarded proxy. It never downloads video/audio or imports browser cookies.
JavaScript runtimes are disabled; some YouTube extraction paths may therefore fail.

Manual captions are preferred over automatic captions, with English preferred within
each kind, then another available language. One JSON3 subtitle track is read, never
transcribed from audio. The output warns whether subtitles are manual, automatic, or
unavailable. Missing/failed captions leave metadata only; Scout does not claim to have
watched the video. Captions are size-bounded and target-validated, and access denials
or rate limits are not retried through another service. Returned metadata and subtitle
text are escaped into a source document for the normal schema/extraction/evidence
flow and its cache. Exa/Jina and GitHub/Reddit connectors are not implemented.

## Development checks

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
# Also lint local tests when tests/ is gitignored:
uv run ruff check --no-respect-gitignore tests
```

Tests run offline without API keys. Tests marked `live` are excluded by default.
`.env`, runtime `data/`, virtual environments, and tool caches are gitignored.
The original starter source under `src/nebius_nvidia_web_scraper/` is preserved,
but the installed project package is now `scout`.

## Intended limits

Public pages only; no login, paywall, CAPTCHA bypass, proxies, or personal-data
scraping. Robots rules will be respected by default. Page content must always be
treated as data, never as instructions.
