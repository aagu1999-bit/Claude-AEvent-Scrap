# ADR 0004: Streamlit dashboard for review + browse

Date: 2026-05-07
Status: Accepted

## Context

The pipeline's primary UI has been the Google Sheet (for browsing events,
managing accounts) plus the Replit shell (for running scripts). After today's
work — auto-promotion candidates, typo suspects, anomaly summaries,
regression alerts — the volume of decisions and data to review per week has
grown past what's comfortable in those two surfaces:

- **Sheet UX:** scrolling 18k events to find this week's worth, copy-pasting
  account handles to add/remove, no per-account drill-down.
- **Shell UX:** `python recurring_accounts.py --dry-run`, eyeball 343 lines
  of console output, edit `promotion_blocklist.json` by hand, re-run, repeat.

The user does not want a second team to touch this; the audience is one
person who works in the browser most of the day.

## Decision

Single-file Streamlit app (`dashboard.py`) with three pages:

1. **Health** — last run stats, anomaly reasons grouped by `error` field,
   per-account scrape→extract counts.
2. **Events** — filterable view of the most recent `Events_*.csv`
   (account, region, event type, search, date range).
3. **Review** — promotion candidates and typo suspects with batched
   write-back. Promote/Block toggles for promotions; REPLACE/REMOVE/KEEP
   actions for typos. Nothing writes until the user clicks **Apply**.

All detection logic is shared with `recurring_accounts.py` via direct import
(`detect_recurring_accounts`, `_detect_typos`, `_load_promotion_blocklist`)
so the dashboard and the CLI never disagree about what counts as a
candidate.

## Why Streamlit

- 10× faster to build than Flask for this shape of app (page = pandas
  operations + a few widgets).
- Reads the existing CSVs and Sheet directly with no schema layer.
- Same Python environment as the pipeline — no cross-language plumbing,
  shared service-account credentials.
- Replit hosts it via a separate workflow entry; runs alongside the cron
  pipeline without coupling.
- Per-page caching with TTL handles the "freshness vs. speed" tradeoff
  cheaply.

Considered alternatives: Flask (too much boilerplate for the same MVP),
Apps Script sidebar inside the Sheet (can't read the local `outputs/`
directory), Streamlit Cloud / Vercel hosting (extra hop, extra auth surface,
unnecessary).

## Consequences

- **Two long-running processes on Replit** instead of one. The cron
  pipeline (Thursday weekly) and the Streamlit server (continuous). They
  share files but neither requires the other to be running.
- **Replit hosting tier may need an upgrade** if the user wants the
  dashboard always-on. The cron alone fits the free tier; an always-on
  Streamlit process typically does not. Not a blocker for development.
- **Detection logic is now imported in two places** (the CLI script and
  the dashboard). Refactoring `_load_all_csvs` / `detect_recurring_accounts`
  signatures will break the dashboard. The shared-import pattern is
  intentional — it's better than copy-paste — but worth knowing.
- **Write actions go straight to the Sheet** via the same service account.
  No staging step. The "Apply" button is the only confirmation. If this
  proves too risky in practice (accidental clicks), wrap with a confirm
  dialog or a staging tab — both straightforward additions.

## What this does NOT include in v1

- A Re-extract page (paste IG URL → run pipeline against just that post).
  Wants real Apify-by-URL integration; deferred.
- Authentication. The dashboard is single-user; Replit's URL is
  unguessable but not secret. If sharing is ever needed, add `streamlit-authenticator`.
- Mobile-first design. Streamlit is responsive enough; not optimized.
- Charts / graphs. Trends over time would be useful (events/week, accounts
  with regression history) but are deferred to keep v1 small.
