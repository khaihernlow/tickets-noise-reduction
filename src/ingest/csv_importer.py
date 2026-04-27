import pandas as pd
from pathlib import Path
from typing import Optional

COLUMN_MAP = {
    "Ticket Number": "ticket_number",
    "Title": "title",
    "Description": "description",
    "Account": "account",
    "Resources": "resources",
    "Status": "status",
    "Created": "created",
    "Total Hours Worked": "total_hours",
    "Billed Hours": "billed_hours",
    "Sub-Issue Type": "sub_issue_type",
    "Issue Type": "issue_type",
}

# Ticket types that are not useful for noise analysis
EXCLUDED_ISSUE_TYPES = {"Email"}
EXCLUDED_SUB_ISSUE_TYPES = {"PHISH"}


def load_csv(path: str, exclude_noise_meta: bool = True) -> pd.DataFrame:
    """Load an Autotask ticket export CSV and return a cleaned DataFrame."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    df = df.rename(columns=COLUMN_MAP)
    df["created"] = pd.to_datetime(df["created"], errors="coerce")
    df["total_hours"] = pd.to_numeric(df["total_hours"], errors="coerce").fillna(0.0)
    df["billed_hours"] = pd.to_numeric(df["billed_hours"], errors="coerce").fillna(0.0)
    df["issue_type"] = df["issue_type"].fillna("").str.strip()
    df["sub_issue_type"] = df["sub_issue_type"].fillna("").str.strip()
    df["account"] = df["account"].fillna("Unknown").str.strip()
    df["resources"] = df["resources"].fillna("").str.strip()
    df["description"] = df["description"].fillna("").str.strip()
    df["title"] = df["title"].fillna("").str.strip()

    if exclude_noise_meta:
        df = df[~df["issue_type"].isin(EXCLUDED_ISSUE_TYPES)]
        df = df[~df["sub_issue_type"].isin(EXCLUDED_SUB_ISSUE_TYPES)]

    df = df.reset_index(drop=True)
    return df


def merge_csvs(paths: list[str], exclude_noise_meta: bool = True) -> pd.DataFrame:
    """Merge multiple CSV exports, deduplicate by ticket number."""
    frames = [load_csv(p, exclude_noise_meta) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticket_number"], keep="last")
    return combined.reset_index(drop=True)
