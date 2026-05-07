#!/usr/bin/env python3
"""
Streamlit dashboard for the Instagram event extraction pipeline.

Read-only screens (Health, Events) browse the most recent run output.
Write actions on the Review page push to Google Sheets via the same
service-account credentials the pipeline uses.

The dashboard is a separate process from the cron pipeline; either can
run without the other. Shares input data (outputs/, the Sheet) but
nothing else. See docs/decisions/0004-streamlit-dashboard.md.

Usage on Replit:
    streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
"""

import glob
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# Reuse detection logic from the CLI tool so dashboard + CLI never diverge.
import recurring_accounts as ra

OUTPUTS_DIR = Path("outputs")
SERVICE_ACCOUNT_FILE = "apt-mark-468506-u9-ec44cabc7335 copy.json"
SHEET_NAME = "Instagram_Events_Master"
BLOCKLIST_PATH = Path("promotion_blocklist.json")

st.set_page_config(
    page_title="Event Scout Dashboard",
    page_icon="📅",
    layout="wide",
)


# ─────────────────────────── data loaders (cached) ─────────────────────────


def _latest(pattern: str) -> Path | None:
    """Return the most recently modified file matching pattern, or None."""
    files = list(OUTPUTS_DIR.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


@st.cache_data(ttl=60)
def load_latest_stats() -> dict:
    p = _latest("stats_*.json")
    if not p:
        return {}
    with open(p) as f:
        data = json.load(f)
    data["_source_file"] = p.name
    data["_modified_at"] = datetime.fromtimestamp(p.stat().st_mtime).isoformat(sep=" ", timespec="seconds")
    return data


@st.cache_data(ttl=60)
def load_latest_anomalies() -> dict:
    p = _latest("anomalies_*.json")
    if not p:
        return {}
    with open(p) as f:
        data = json.load(f)
    data["_source_file"] = p.name
    data["_modified_at"] = datetime.fromtimestamp(p.stat().st_mtime).isoformat(sep=" ", timespec="seconds")
    return data


@st.cache_data(ttl=60)
def load_latest_events() -> pd.DataFrame:
    p = _latest("Events_*.csv")
    if not p:
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    df["_source_file"] = p.name
    return df


@st.cache_data(ttl=300)
def load_recurring_accounts() -> pd.DataFrame:
    """Run the recurring-events detector against the historical CSVs."""
    df = ra._load_all_csvs()
    if df.empty:
        return pd.DataFrame()
    return ra.detect_recurring_accounts(df)


@st.cache_data(ttl=60)
def load_blocklist() -> set[str]:
    return ra._load_promotion_blocklist()


@st.cache_data(ttl=120)
def load_active_accounts_from_sheet() -> list[str]:
    """Pull the live Accounts tab. Cached so we don't hammer the Sheets API."""
    sheet = _connect_sheet()
    if sheet is None:
        return []
    try:
        ws = sheet.worksheet("Accounts")
        col = ws.col_values(1)
        if col and col[0].strip().lower() == "username":
            col = col[1:]
        return [u.strip() for u in col if u.strip()]
    except Exception as e:
        st.warning(f"Could not read Accounts tab: {e}")
        return []


def _connect_sheet():
    """Return the gspread Spreadsheet, or None if creds are missing."""
    if not Path(SERVICE_ACCOUNT_FILE).exists():
        return None
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        scope = ["https://spreadsheets.google.com/feeds",
                 "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
        return gspread.authorize(creds).open(SHEET_NAME)
    except Exception as e:
        st.error(f"Sheet connection failed: {e}")
        return None


# ─────────────────────────── pages ─────────────────────────


def page_health():
    st.title("📊 Health")
    st.caption("Most recent run status, anomaly counts, and regression alerts.")

    stats = load_latest_stats()
    anomalies = load_latest_anomalies()

    if not stats and not anomalies:
        st.info("No recent run data found in `outputs/`. Run the pipeline first.")
        return

    cols = st.columns(4)
    cols[0].metric("Posts processed", stats.get("processed", 0))
    cols[1].metric("Events found", stats.get("events_found", 0))
    cols[2].metric("Posts with events", stats.get("posts_with_events", 0))
    cols[3].metric("Posts no events", stats.get("posts_no_events", 0))

    cols2 = st.columns(4)
    cols2[0].metric("Gemini errors", stats.get("gemini_errors", 0))
    cols2[1].metric("OCR failed", stats.get("ocr_failed", 0))
    cols2[2].metric("OCR success", stats.get("ocr_success", 0))
    cols2[3].metric("Calendar posts", stats.get("calendar_posts", 0))

    with st.expander("Run metadata", expanded=False):
        st.write(f"**Latest stats file:** `{stats.get('_source_file', '?')}`")
        st.write(f"**Modified:** {stats.get('_modified_at', '?')}")
        if anomalies:
            st.write(f"**Latest anomalies file:** `{anomalies.get('_source_file', '?')}`")

    # Reasons breakdown
    if anomalies and "anomalies" in anomalies:
        st.subheader("Why posts produced no events")
        reasons = {}
        for entry in anomalies["anomalies"].values():
            r = entry.get("error", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        if reasons:
            reasons_df = pd.DataFrame(
                sorted(reasons.items(), key=lambda x: -x[1]),
                columns=["Reason", "Count"],
            )
            st.dataframe(reasons_df, hide_index=True, use_container_width=True)
        else:
            st.success("No anomalies recorded in the last run.")

    # Per-account scrape→extract
    if anomalies and "per_account" in anomalies:
        st.subheader("Per-account outcomes (last run)")
        rows = []
        for acct, b in anomalies["per_account"].items():
            rows.append({
                "Account": acct,
                "Scraped": b.get("scraped", 0),
                "Events found": b.get("events_found", 0),
                "No events": b.get("no_events_found", 0),
                "Gemini errors": b.get("gemini_error", 0),
                "OCR failed": b.get("ocr_failed", 0),
            })
        per_acct_df = pd.DataFrame(rows).sort_values("Scraped", ascending=False)
        st.dataframe(per_acct_df, hide_index=True, use_container_width=True)


def page_events():
    st.title("🎟  Events")
    st.caption("Most recent extraction snapshot. Filter, search, click through to the IG post.")

    df = load_latest_events()
    if df.empty:
        st.info("No `Events_*.csv` found in `outputs/`. Run the pipeline first.")
        return

    st.write(f"**Source:** `{df['_source_file'].iloc[0]}` — {len(df):,} rows")
    df = df.drop(columns=["_source_file"])

    # Normalise column names — uppercase variant from new format
    col_map = {c: c for c in df.columns}
    handle_col = next((c for c in df.columns if c.lower().replace(" ", "_") == "instagram_handle"), None)
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    region_col = next((c for c in df.columns if c.lower().replace(" ", "_") == "section_of_nj"), None)
    type_col = next((c for c in df.columns if c.lower().replace(" ", "_") == "event_type"), None)
    name_col = next((c for c in df.columns if c.lower().replace(" ", "_") == "event_name"), None)

    # Filters
    with st.sidebar:
        st.header("Filters")
        if handle_col:
            handles = sorted({str(h).lower() for h in df[handle_col].dropna() if str(h).strip()})
            selected_handles = st.multiselect("Account", handles, default=[])
        else:
            selected_handles = []
        if region_col:
            regions = sorted({str(r) for r in df[region_col].dropna() if str(r).strip()})
            selected_regions = st.multiselect("Region", regions, default=[])
        else:
            selected_regions = []
        if type_col:
            types = sorted({str(t) for t in df[type_col].dropna() if str(t).strip()})
            selected_types = st.multiselect("Event type", types, default=[])
        else:
            selected_types = []
        search_text = st.text_input("Search event name", "")
        if date_col:
            df_dates = pd.to_datetime(df[date_col], errors="coerce")
            min_d = df_dates.min()
            max_d = df_dates.max()
            if pd.notna(min_d) and pd.notna(max_d):
                date_range = st.date_input(
                    "Date range",
                    value=(min_d.date(), max_d.date()),
                    min_value=min_d.date(),
                    max_value=max_d.date(),
                )
            else:
                date_range = None
        else:
            date_range = None

    filtered = df.copy()
    if selected_handles and handle_col:
        filtered = filtered[filtered[handle_col].astype(str).str.lower().isin(selected_handles)]
    if selected_regions and region_col:
        filtered = filtered[filtered[region_col].astype(str).isin(selected_regions)]
    if selected_types and type_col:
        filtered = filtered[filtered[type_col].astype(str).isin(selected_types)]
    if search_text and name_col:
        filtered = filtered[filtered[name_col].astype(str).str.contains(search_text, case=False, na=False)]
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2 and date_col:
        d_series = pd.to_datetime(filtered[date_col], errors="coerce")
        start, end = date_range
        filtered = filtered[(d_series.dt.date >= start) & (d_series.dt.date <= end)]

    st.write(f"**Showing {len(filtered):,} of {len(df):,} events**")
    st.dataframe(filtered, use_container_width=True, hide_index=True)


def page_review():
    st.title("✅ Review")
    st.caption("Approve promotions, resolve typos, and update your active accounts list. "
               "Changes are batched — nothing is written to the Sheet until you click "
               "**Apply selected**.")

    result = load_recurring_accounts()
    if result.empty:
        st.info("No recurring accounts detected. Run the pipeline first.")
        return

    active = [h.lower() for h in load_active_accounts_from_sheet()]
    if not active:
        st.error("Could not read the live Accounts tab. Promotions disabled until that's fixed.")
        return

    blocklist = load_blocklist()
    active_set = set(active)
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=180)

    # ─────────── Promotions ───────────
    candidates = []
    for _, row in result.iterrows():
        h = str(row["Account"]).strip().lower()
        if not h or h in active_set or h in blocklist:
            continue
        dp = int(row.get("Distinct Posts", 0) or 0)
        if dp < 3:
            continue
        ls = row.get("_last_seen")
        if ls and ls < cutoff:
            continue
        candidates.append({
            "Handle": h,
            "Distinct Posts": dp,
            "Occurrences": int(row.get("Occurrences", 0) or 0),
            "Last Seen": ls.isoformat() if ls else "unknown",
            "Sample": str(row.get("Example Series Names", ""))[:80],
        })
    candidates.sort(key=lambda c: -c["Distinct Posts"])

    st.subheader(f"Promotion candidates ({len(candidates)})")
    if candidates:
        promote_df = pd.DataFrame(candidates)
        promote_df.insert(0, "Promote", False)
        promote_df.insert(1, "Block", False)
        promote_edited = st.data_editor(
            promote_df,
            column_config={
                "Promote": st.column_config.CheckboxColumn("✓ Promote", default=False),
                "Block": st.column_config.CheckboxColumn("✗ Block", default=False),
                "Handle": st.column_config.TextColumn(disabled=True),
                "Distinct Posts": st.column_config.NumberColumn(disabled=True),
                "Occurrences": st.column_config.NumberColumn(disabled=True),
                "Last Seen": st.column_config.TextColumn(disabled=True),
                "Sample": st.column_config.TextColumn(disabled=True),
            },
            hide_index=True,
            use_container_width=True,
            key="promotion_editor",
        )
    else:
        promote_edited = pd.DataFrame()
        st.success("No new accounts qualify for promotion right now.")

    # ─────────── Typo suspects ───────────
    suspects = ra._detect_typos(active, result["Account"].tolist())
    typo_action_choices = ["", "REPLACE (delete + add suggested)", "REMOVE (delete only)", "KEEP (suppress future flags)"]
    st.subheader(f"Typo suspects ({len(suspects)})")
    if suspects:
        typo_df = pd.DataFrame(
            [{"Active handle": a, "Suggested": s, "Distance": d}
             for a, s, d in suspects]
        )
        typo_df.insert(0, "Action", "")
        typo_edited = st.data_editor(
            typo_df,
            column_config={
                "Action": st.column_config.SelectboxColumn(
                    "Action",
                    options=typo_action_choices,
                    default="",
                    required=False,
                ),
                "Active handle": st.column_config.TextColumn(disabled=True),
                "Suggested": st.column_config.TextColumn(disabled=True),
                "Distance": st.column_config.NumberColumn(disabled=True),
            },
            hide_index=True,
            use_container_width=True,
            key="typo_editor",
        )
    else:
        typo_edited = pd.DataFrame()
        st.success("No typo suspects flagged.")

    # ─────────── Apply ───────────
    st.divider()
    if st.button("Apply selected actions", type="primary"):
        _apply_review_actions(promote_edited, typo_edited)


def _apply_review_actions(promote_df, typo_df):
    """Execute the user's selections from the Review page against the live Sheet."""
    sheet = _connect_sheet()
    if sheet is None:
        st.error("Could not connect to the Sheet. Aborting.")
        return

    # Re-read accounts so we have row indices for deletes
    try:
        ws = sheet.worksheet("Accounts")
        all_values = ws.col_values(1)
    except Exception as e:
        st.error(f"Could not read Accounts tab: {e}")
        return

    promote_actions = []
    block_actions = []
    if not promote_df.empty:
        for _, row in promote_df.iterrows():
            if bool(row.get("Promote")):
                promote_actions.append(row["Handle"])
            elif bool(row.get("Block")):
                block_actions.append(row["Handle"])

    typo_replace = []
    typo_remove = []
    typo_keep = []
    if not typo_df.empty:
        for _, row in typo_df.iterrows():
            action = str(row.get("Action", "") or "")
            if action.startswith("REPLACE"):
                typo_replace.append((row["Active handle"], row["Suggested"]))
            elif action.startswith("REMOVE"):
                typo_remove.append(row["Active handle"])
            elif action.startswith("KEEP"):
                typo_keep.append(row["Active handle"])

    if not (promote_actions or block_actions or typo_replace or typo_remove or typo_keep):
        st.warning("Nothing selected. Tick a Promote/Block box or pick a typo Action first.")
        return

    progress = st.empty()
    log_lines = []

    # Apply promotions (append to Accounts tab)
    if promote_actions:
        try:
            ws.append_rows([[h] for h in promote_actions], value_input_option="RAW")
            log_lines.append(f"✓ Promoted {len(promote_actions)} accounts to Accounts tab")
        except Exception as e:
            log_lines.append(f"✗ Promotion failed: {e}")

    # Apply blocklist additions (write to local JSON)
    if block_actions:
        try:
            data = {"_comment": "", "handles": []}
            if BLOCKLIST_PATH.exists():
                with open(BLOCKLIST_PATH) as f:
                    data = json.load(f)
            existing = set(data.get("handles", []))
            data["handles"] = sorted(existing | {h.lower() for h in block_actions})
            with open(BLOCKLIST_PATH, "w") as f:
                json.dump(data, f, indent=2)
            log_lines.append(
                f"✓ Added {len(block_actions)} to {BLOCKLIST_PATH.name} "
                f"(now {len(data['handles'])} total)"
            )
        except Exception as e:
            log_lines.append(f"✗ Blocklist update failed: {e}")

    # Apply typo REPLACEs and REMOVEs
    if typo_replace or typo_remove:
        try:
            # Build {handle: row_index} for current Accounts tab (1-based)
            handle_to_row = {}
            for i, v in enumerate(all_values, start=1):
                key = v.strip().lower()
                if key and key != "username":
                    handle_to_row[key] = i

            rows_to_delete = []
            handles_to_add = []
            for typo_handle, suggested in typo_replace:
                idx = handle_to_row.get(typo_handle.lower())
                if idx:
                    rows_to_delete.append(idx)
                if suggested.lower() not in {v.strip().lower() for v in all_values}:
                    handles_to_add.append(suggested)
            for typo_handle in typo_remove:
                idx = handle_to_row.get(typo_handle.lower())
                if idx:
                    rows_to_delete.append(idx)

            # Delete in reverse so earlier indices don't shift
            for idx in sorted(rows_to_delete, reverse=True):
                try:
                    ws.delete_rows(idx)
                except Exception as e:
                    log_lines.append(f"✗ Could not delete row {idx}: {e}")

            if handles_to_add:
                ws.append_rows([[h] for h in handles_to_add], value_input_option="RAW")

            log_lines.append(
                f"✓ Typo actions applied: {len(typo_replace)} replaced, "
                f"{len(typo_remove)} removed"
            )
        except Exception as e:
            log_lines.append(f"✗ Typo updates failed: {e}")

    # Persist KEEP decisions to a local allowlist (don't flag again)
    if typo_keep:
        try:
            allow_path = Path("typo_allowlist.json")
            data = {"handles": []}
            if allow_path.exists():
                with open(allow_path) as f:
                    data = json.load(f)
            existing = set(data.get("handles", []))
            data["handles"] = sorted(existing | {h.lower() for h in typo_keep})
            with open(allow_path, "w") as f:
                json.dump(data, f, indent=2)
            log_lines.append(f"✓ Added {len(typo_keep)} to typo_allowlist.json")
        except Exception as e:
            log_lines.append(f"✗ Allowlist update failed: {e}")

    progress.success("\n".join(log_lines))
    # Bust caches so the page reflects the new state on rerun
    load_active_accounts_from_sheet.clear()
    load_blocklist.clear()
    st.info("Refresh the page (or rerun) to see updated state.")


# ─────────────────────────── nav ─────────────────────────

PAGES = {
    "Health": page_health,
    "Events": page_events,
    "Review": page_review,
}


def main():
    st.sidebar.title("Event Scout")
    page = st.sidebar.radio("Page", list(PAGES.keys()))
    st.sidebar.divider()
    st.sidebar.caption(f"Working directory: `{Path.cwd()}`")
    st.sidebar.caption(f"Outputs: `{OUTPUTS_DIR.resolve()}`")
    if st.sidebar.button("Refresh data"):
        load_latest_stats.clear()
        load_latest_anomalies.clear()
        load_latest_events.clear()
        load_recurring_accounts.clear()
        load_active_accounts_from_sheet.clear()
        load_blocklist.clear()
        st.rerun()

    PAGES[page]()


if __name__ == "__main__":
    main()
