import pandas as pd
from typing import List
from ..models.ticket import Pattern

# Minimum tickets in a group before it's considered a pattern worth reviewing
MIN_CLUSTER_SIZE = 2
HIGH_EFFORT_THRESHOLD = 0.5  # hours


def find_patterns(df: pd.DataFrame) -> List[Pattern]:
    patterns: List[Pattern] = []

    patterns.extend(_recurring_issue_patterns(df))
    patterns.extend(_repeat_contact_patterns(df))
    patterns.extend(_high_effort_patterns(df))
    patterns.extend(_same_day_burst_patterns(df))

    # deduplicate: a ticket can appear in multiple pattern types but we want
    # the most specific grouping first — sort by ticket_count desc, then hours desc
    patterns.sort(key=lambda p: (p.ticket_count, p.total_hours), reverse=True)
    return patterns


def _to_records(group: pd.DataFrame) -> list:
    cols = ["ticket_number", "title", "description", "resources", "created", "total_hours", "issue_type", "sub_issue_type"]
    return group[cols].to_dict("records")


def _recurring_issue_patterns(df: pd.DataFrame) -> List[Pattern]:
    """Same account + issue_type appearing multiple times."""
    results = []
    valid = df[df["issue_type"] != ""]
    for (account, issue_type), group in valid.groupby(["account", "issue_type"]):
        if len(group) < MIN_CLUSTER_SIZE:
            continue
        results.append(Pattern(
            pattern_type="recurring_issue",
            account=account,
            issue_type=issue_type,
            ticket_count=len(group),
            total_hours=round(group["total_hours"].sum(), 2),
            tickets=_to_records(group),
            sub_issue_breakdown=group["sub_issue_type"].value_counts().to_dict(),
        ))
    return results


def _repeat_contact_patterns(df: pd.DataFrame) -> List[Pattern]:
    """Single end-user generating multiple tickets — training opportunity."""
    results = []

    # Extract primary contact from Resources column (format: "LastName, FirstName (primary)")
    def extract_primary(resources_str: str) -> str:
        if not resources_str:
            return ""
        for part in resources_str.split(";"):
            if "(primary)" in part.lower():
                return part.replace("(primary)", "").strip()
        return resources_str.split(";")[0].strip()

    df = df.copy()
    df["_primary_resource"] = df["resources"].apply(extract_primary)

    valid = df[df["_primary_resource"] != ""]
    for (account, contact), group in valid.groupby(["account", "_primary_resource"]):
        if len(group) < MIN_CLUSTER_SIZE:
            continue
        results.append(Pattern(
            pattern_type="repeat_contact",
            account=account,
            issue_type="(multiple)",
            ticket_count=len(group),
            total_hours=round(group["total_hours"].sum(), 2),
            tickets=_to_records(group),
            contact=contact,
            sub_issue_breakdown=group["issue_type"].value_counts().to_dict(),
        ))
    return results


def _high_effort_patterns(df: pd.DataFrame) -> List[Pattern]:
    """Tickets consuming > HIGH_EFFORT_THRESHOLD hours for an issue type that recurs."""
    high = df[df["total_hours"] > HIGH_EFFORT_THRESHOLD]
    results = []
    valid = high[high["issue_type"] != ""]
    for (account, issue_type), group in valid.groupby(["account", "issue_type"]):
        if len(group) < MIN_CLUSTER_SIZE:
            continue
        results.append(Pattern(
            pattern_type="high_effort",
            account=account,
            issue_type=issue_type,
            ticket_count=len(group),
            total_hours=round(group["total_hours"].sum(), 2),
            tickets=_to_records(group),
            sub_issue_breakdown=group["sub_issue_type"].value_counts().to_dict(),
            extra={"avg_hours": round(group["total_hours"].mean(), 2)},
        ))
    return results


def _same_day_burst_patterns(df: pd.DataFrame) -> List[Pattern]:
    """Multiple tickets from the same account on the same day — suggests an incident or deployment issue."""
    df = df.copy()
    df["_date"] = df["created"].dt.date
    results = []
    for (account, date), group in df.groupby(["account", "_date"]):
        if len(group) < 3:  # higher threshold — 2 tickets in a day is normal
            continue
        results.append(Pattern(
            pattern_type="same_day_burst",
            account=account,
            issue_type="(multiple)",
            ticket_count=len(group),
            total_hours=round(group["total_hours"].sum(), 2),
            tickets=_to_records(group),
            sub_issue_breakdown=group["issue_type"].value_counts().to_dict(),
            extra={"date": str(date)},
        ))
    return results
