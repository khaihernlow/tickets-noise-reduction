# NRC AI — Noise Reduction Committee Analysis Tool

An AI-assisted ticket analysis tool built for **TAG Solutions** (Albany, NY) to support the bi-weekly Noise Reduction Committee (NRC) process. It ingests IT support ticket data exported from Autotask, detects recurring patterns, and uses the HatzAI LLM API to generate actionable recommendations — identifying what proactive steps (automation, training, process changes, or sales opportunities) would prevent tickets from recurring.

---

## Background

TAG Solutions runs a bi-weekly **Noise Reduction Committee** meeting where support staff manually review tickets from the top accounts and try to identify patterns — repeated issues that could be eliminated through proactive action. That manual process involves:

1. Reviewing tagged tickets from the last 30 days
2. Reviewing the three accounts with the most tickets from the prior week
3. Reviewing Root Cause Analysis forms for major incidents
4. Generating recommendation forms submitted to the NRC chair

This tool automates steps 1, 2, and 4. It replaces manual ticket reading with a pipeline that:
- Statistically groups tickets into meaningful clusters using traditional code (no AI)
- Sends each cluster to an LLM with rich context about recurrence, affected users, and historical trend
- Outputs structured recommendations in the same format the NRC uses

---

## How It Works

### The Two-Stage Design

The system deliberately separates what AI does from what traditional code does.

**Stage 1 — Pattern detection (no AI)**

Pure Python/pandas grouping on the ticket DataFrame. This finds clusters by:

| Detector | What it finds |
|---|---|
| `recurring_issue` | Same account + issue type appearing 2+ times |
| `repeat_contact` | Same technician handling multiple tickets for the same account |
| `same_day_burst` | 3+ tickets from the same account on the same calendar day |

These detectors produce reliable, countable groupings. No LLM is needed here — this is just `groupby` and counting.

**Stage 2 — Recommendation generation (LLM)**

For each significant cluster, the system builds a structured prompt containing:
- Ticket count, recurrence rate (tickets/week), % of the account's total tickets
- How many unique contacts/users are affected
- A representative sample of up to 15 tickets selected for sub-issue variety and description detail
- Historical context: when this pattern first appeared, all-time count, and trend vs. the prior equivalent period (increasing / stable / decreasing)

The LLM's job is to read the free-text descriptions and answer: *what is actually causing this, and what specific action would prevent it?* This is what LLMs are good at. Counting and grouping is left entirely to code.

### Why Not Send the Whole CSV to the LLM?

A 30-day Autotask export can be 500–1,000 tickets. Sending everything to the LLM:
- Is slow and expensive
- Produces low-quality output — LLMs are unreliable at arithmetic, grouping, and counting
- Wastes tokens describing obvious summaries instead of the nuanced insights that matter

By doing the grouping in code first, the LLM receives a pre-digested cluster of related tickets and focuses entirely on interpretation and recommendation.

---

## Architecture

```
CSV file(s)  ──►  csv_importer.py  ──►  SQLite store (data/nrc.db)
                                               │
                                    load_tickets(since_date)
                                               │
                                          patterns.py
                                     (pure pandas groupby)
                                               │
                                      Pattern clusters
                                               │
                              ┌────────────────┴────────────────┐
                              │  cache lookup (SQLite)          │
                              │  historical context (SQLite)    │
                              └────────────────┬────────────────┘
                                               │
                                  ThreadPoolExecutor (5 workers)
                                               │
                                    HatzAI API (LLM)
                                               │
                              cache_set ◄──────┘
                                               │
                                     Recommendations
                                               │
                                       CLI output
```

### Project Structure

```
nrc-ai/
├── main.py                         CLI entry point
├── requirements.txt
├── .env                            API key + model config (not committed)
├── .env.example                    Template for .env
├── data/
│   └── nrc.db                      SQLite store (not committed)
└── src/
    ├── models/
    │   └── ticket.py               Ticket, Pattern, Recommendation dataclasses
    ├── ingest/
    │   ├── csv_importer.py         Autotask CSV loader and cleaner
    │   └── autotask_client.py      Stub for future Autotask REST API integration
    ├── store/
    │   └── db.py                   SQLite store: tickets, recommendation cache, historical context
    ├── hatzai/
    │   └── client.py               HatzAI REST API wrapper with retry logic
    └── analysis/
        ├── patterns.py             Pattern detectors (recurring, repeat contact, burst)
        └── recommender.py          LLM prompt builder, concurrent execution, cache integration
```

---

## Setup

**Requirements:** Python 3.12+

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:

```
HATZAI_API_KEY=your-api-key-here
HATZAI_MODEL=anthropic.claude-sonnet-4-6
```

To see all available models in your HatzAI account:

```bash
python main.py models
```

The `HATZAI_MODEL` value should be the **Model ID** column from that output (e.g. `anthropic.claude-sonnet-4-6`, `anthropic.claude-opus-4-7`).

---

## Usage

### Importing Tickets

Export completed tickets from Autotask as a CSV. The CSV must include these columns:

```
Ticket Number, Title, Description, Account, Resources, Status,
Created, Total Hours Worked, Billed Hours, Sub-Issue Type, Issue Type
```

Import one or more files:

```bash
python main.py import data/tickets_april.csv
python main.py import data/q1.csv data/q2.csv data/q3.csv
```

The importer deduplicates by ticket number — re-importing the same file is safe. Each run reports how many tickets were new vs. skipped as duplicates.

Spam/phishing tickets (issue type `Email`, sub-issue type `PHISH`) are automatically excluded on import.

### Running Analysis

**Analyze all stored tickets:**
```bash
python main.py analyze
```

**Analyze only the last N days (recommended for NRC meetings):**
```bash
python main.py analyze --window 30
```

**Analyze since a specific date:**
```bash
python main.py analyze --since 2026-01-01
```

**Import and analyze in one step:**
```bash
python main.py analyze data/new_tickets.csv --window 30
```

**Pattern stats only — no LLM calls:**
```bash
python main.py analyze --window 30 --no-llm
```

**Force re-analysis of all patterns (ignore cache):**
```bash
python main.py analyze --force-refresh
```

**Limit to top N patterns:**
```bash
python main.py analyze --window 30 --top 15
```

### Other Commands

```bash
python main.py status         # tickets in store, cached recommendations
python main.py cache-clear    # wipe cached recommendations (tickets are kept)
python main.py models         # list available HatzAI models
```

---

## Output

The analysis produces three sections:

**1. Ticket Summary** — ticket counts by account and issue type for the analysis window.

**2. Detected Patterns** — table of all clusters found, with:
- Pattern type
- Account
- Ticket count
- Unique contacts/resources involved
- Recurrence rate (tickets/week)
- Percentage of the account's total tickets this cluster represents

**3. NRC Recommendations** — one recommendation per significant pattern, each containing:
- Priority (high / medium / low)
- Recommendation type (Automation / User Training / Process Change / Sales Opportunity)
- Which tickets are involved
- Pattern summary
- Root cause
- Specific recommended action
- Estimated monthly tickets prevented

Example output:

```
------------------------------------------------------------------------------
  #1  [  MEDIUM   ]  Process Change
  Account : Manlius Pebble Hill School
  Tickets : T20260424.0084, T20260424.0085, T20260424.0086 ...
  Impact  : ~6 tickets/mo prevented

  PATTERN
  A Toshiba ESPM print system rollout generated 8 same-day reactive tickets
  due to incomplete deployment, missing pre-configuration, and absent training.

  ROOT CAUSE
  No deployment runbook was followed. App deployment was incomplete, ESPM
  sessions were not pre-configured, user accounts were not finalized before
  launch, and end-user training was reactive rather than proactive.

  RECOMMENDED ACTION
  Create a mandatory Print System Deployment Runbook requiring sign-off on:
  MDM-verified app deployment to 100% of endpoints, scripted ESPM persistent
  login via GPO/MDM, user accounts provisioned 48h before launch, and a
  scheduled training session prior to cutover.
```

---

## Data & Caching

### Ticket Store (SQLite)

All imported tickets are stored permanently in `data/nrc.db`. The store:
- Deduplicates on `ticket_number` — safe to re-import old exports
- Uses WAL journal mode for concurrent read access during analysis
- Has indexes on `account`, `issue_type`, `created`, and `(account, issue_type)` for efficient filtering at scale

### Detection Window vs. Historical Context

These are treated separately:

- **Detection window** (`--window` / `--since`): the scope of tickets used for pattern detection. Default is all stored tickets; use `--window 30` for the standard NRC bi-weekly review.
- **Historical context**: before each LLM call, the full ticket store is queried to provide trend data — how long the pattern has existed, all-time ticket count, and whether it's increasing or decreasing vs. the prior equivalent period. This means even with `--window 30`, the LLM knows if a pattern started in 2025 or just this week.

This separation prevents historical data from inflating current pattern scores while still giving the LLM the context it needs to assess urgency and priority accurately.

### Recommendation Cache

LLM recommendations are cached in `data/nrc.db` keyed by a SHA-256 hash of:

```
pattern_type + account + issue_type + sorted(ticket_numbers) + model_name
```

On each analysis run:
1. Each pattern's fingerprint is computed
2. If a cache entry exists, the stored recommendation is reused — no LLM call
3. If even one ticket is added to a cluster, the fingerprint changes and the pattern is re-analyzed automatically

The cache never silently reuses stale data for a changed cluster. Use `cache-clear` to force full re-analysis, or `--force-refresh` to bypass cache for a single run without deleting entries.

---

## Scalability Design

The system is designed to handle the full TAG Solutions ticket history (estimated 22,000+ tickets across 2025–2026) without degrading analysis quality.

### What Prevents Quality Degradation at Scale

**Significance filtering:** Patterns with fewer than 3 tickets are excluded before any LLM call. With 22k tickets, pattern detection produces many small clusters — this filter keeps LLM analysis focused on patterns that actually matter.

**Representative ticket sampling:** When a cluster has more than 15 tickets, the system selects a representative sample rather than taking the first 15. Selection prioritizes:
1. Sub-issue type variety — the LLM sees the full shape of the problem, not 15 copies of the same thing
2. Description length — longer descriptions contain more diagnostic detail

The LLM prompt always includes a statistical header with the full cluster size, so it understands the scale even when only reading a sample.

**Two-scope architecture:** Pattern detection runs on the recent window (e.g. last 30 days). Historical context is a separate SQL query against the full store that enriches each prompt with trend data. The LLM gets: "this pattern has 15 tickets in the last 30 days (+87% vs. the prior 30 days, first seen March 2025)." That is more useful than seeing all 47 historical tickets.

**Temporal drift awareness:** Patterns are analyzed in the context of their trend. A cluster that is decreasing over time is likely being resolved and the LLM will reflect that in its recommendation. A cluster that is escalating gets higher priority. This prevents the system from surfacing already-resolved patterns from 2025 as urgent current recommendations.

### Performance

| Operation | Mechanism |
|---|---|
| CSV import | `executemany()` bulk insert — ~7k rows in ~1-2s |
| Ticket loading | SQL `WHERE created >= ?` — only loads the detection window into memory |
| Pattern detection | In-memory pandas groupby — fast at any realistic scale |
| LLM calls | `ThreadPoolExecutor` with 5 concurrent workers — 20 patterns takes ~20s instead of ~60s |
| Repeated runs | Cache hits return instantly — 0 LLM calls if no new tickets joined any cluster |
| API failures | Exponential backoff retry (1s, 2s, 4s) for rate limits and transient errors |

---

## Design Decisions

**Why total hours are not used:** Technicians at TAG Solutions do not consistently log time against tickets, making `Total Hours Worked` unreliable as an analytical signal. The system uses ticket count, unique contacts affected, recurrence rate, and account noise ratio instead. The `total_hours` field is stored in the DB but not used for pattern scoring, significance filtering, or LLM prompts.

**Why SQLite instead of PostgreSQL:** This is a single-user CLI tool. SQLite with WAL mode handles concurrent reads safely and performs well up to millions of rows. A migration to PostgreSQL would be straightforward if the tool is later wrapped in a web interface with multiple concurrent users.

**Why HatzAI instead of direct Anthropic API:** TAG Solutions has an existing HatzAI subscription. The tool is built against HatzAI's `/v1/chat/completions` endpoint (base URL: `https://ai.hatz.ai/v1`, auth via `X-API-Key` header). The model is configurable via `HATZAI_MODEL` in `.env` — any model available in the HatzAI account can be used.

---

## Future Work

- **Autotask API integration:** `src/ingest/autotask_client.py` is a stub. Once API credentials are provisioned, `fetch_completed_tickets(days_back=30)` can be wired up to replace manual CSV exports with scheduled automated ingestion.
- **Pattern lifecycle tracking:** A `pattern_history` table would record each time a pattern was detected and its ticket count, enabling the system to automatically mark patterns as "likely resolved" if they stop recurring after a recommendation was made.
- **Report export:** Output recommendations as a formatted PDF or email digest for the NRC meeting, replacing the need to run the CLI.
- **`--top` auto-tuning:** Automatically set the pattern limit based on the number of significant clusters rather than a fixed default.
