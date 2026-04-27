import json
import sqlite3
from typing import List, Optional
from ..models.ticket import Pattern, Recommendation
from ..hatzai.client import HatzAIClient
from ..store.db import cache_get, cache_set

SYSTEM_PROMPT = """\
You are a Noise Reduction analyst for TAG Solutions, a Managed Service Provider (MSP) in Albany, NY.

Your job is to analyze clusters of related IT support tickets and produce a structured Noise Reduction Committee (NRC) recommendation.

Noise = preventable reactive work. Your recommendations should identify what proactive action (automation, user training, process change, or new sales opportunity) would prevent these tickets from recurring.

Be specific and actionable. Reference ticket numbers and actual issue patterns. Keep language concise.
"""


def _build_prompt(pattern: Pattern) -> str:
    ticket_lines = []
    for t in pattern.tickets[:15]:
        desc = str(t.get("description", ""))[:250].replace("\n", " ")
        ticket_lines.append(
            f"  [{t['ticket_number']}] {t['title']} | {t['issue_type']} / {t['sub_issue_type']} | "
            f"{t['total_hours']}h | {desc}"
        )

    tickets_block = "\n".join(ticket_lines)
    contact_line = f"\nPrimary technician generating these tickets: {pattern.contact}" if pattern.contact else ""
    extra_line = f"\nAdditional context: {json.dumps(pattern.extra)}" if pattern.extra else ""

    return f"""\
Analyze the following cluster of support tickets and generate a Noise Reduction Committee recommendation.

Pattern type: {pattern.pattern_type}
Account: {pattern.account}
Issue type: {pattern.issue_type}
Ticket count: {pattern.ticket_count}
Total hours spent: {pattern.total_hours}h
Sub-issue / issue breakdown: {json.dumps(pattern.sub_issue_breakdown)}{contact_line}{extra_line}

Tickets in this cluster:
{tickets_block}

Respond with ONLY a JSON object in this exact schema (no markdown, no explanation):
{{
  "pattern_summary": "<1-2 sentence description of the recurring pattern>",
  "root_cause": "<what is actually driving these tickets>",
  "recommendation_type": "<one of: automation | training | process_change | sales_opportunity>",
  "recommended_action": "<specific, actionable step TAG Solutions should take>",
  "estimated_monthly_tickets_prevented": <integer>,
  "estimated_monthly_hours_saved": <number>,
  "priority": "<one of: high | medium | low>"
}}
"""


def _ticket_numbers(pattern: Pattern) -> list[str]:
    return [t["ticket_number"] for t in pattern.tickets]


def _result_to_recommendation(result: dict, pattern: Pattern) -> Recommendation:
    return Recommendation(
        pattern=pattern,
        pattern_summary=result.get("pattern_summary", ""),
        root_cause=result.get("root_cause", ""),
        recommendation_type=result.get("recommendation_type", "process_change"),
        recommended_action=result.get("recommended_action", ""),
        estimated_monthly_tickets_prevented=int(result.get("estimated_monthly_tickets_prevented", 0)),
        estimated_monthly_hours_saved=float(result.get("estimated_monthly_hours_saved", 0)),
        priority=result.get("priority", "medium"),
        source_ticket_numbers=_ticket_numbers(pattern),
    )


def generate_recommendations(
    patterns: List[Pattern],
    client: HatzAIClient,
    conn: Optional[sqlite3.Connection] = None,
    force_refresh: bool = False,
    max_patterns: int = 20,
) -> List[Recommendation]:
    """Generate NRC recommendations for the given patterns.

    If `conn` is provided, results are cached in SQLite by pattern fingerprint.
    Cache hits skip the LLM call entirely.
    `force_refresh=True` ignores the cache and always calls the LLM.
    """
    recommendations = []
    cache_hits = 0
    llm_calls = 0

    for pattern in patterns[:max_patterns]:
        ticket_nums = _ticket_numbers(pattern)

        # ── cache lookup ───────────────────────────────────────────────────────
        cached = None
        if conn and not force_refresh:
            cached = cache_get(
                pattern.pattern_type, pattern.account, pattern.issue_type,
                ticket_nums, client.model, conn,
            )

        if cached:
            cache_hits += 1
            print(f"  [cached]  {pattern.pattern_type} | {pattern.account} | {pattern.ticket_count} tickets")
            recommendations.append(_result_to_recommendation(cached, pattern))
            continue

        # ── LLM call ───────────────────────────────────────────────────────────
        llm_calls += 1
        print(f"  [llm]     {pattern.pattern_type} | {pattern.account} | {pattern.ticket_count} tickets...")
        try:
            result = client.chat_json(_build_prompt(pattern), system=SYSTEM_PROMPT)

            if conn:
                cache_set(
                    pattern.pattern_type, pattern.account, pattern.issue_type,
                    ticket_nums, client.model, result, conn,
                )

            recommendations.append(_result_to_recommendation(result, pattern))

        except Exception as e:
            print(f"  WARNING: failed for ({pattern.account} / {pattern.issue_type}): {e}")

    if conn:
        print(f"\n  Cache: {cache_hits} hits, {llm_calls} LLM calls made.")

    recommendations.sort(
        key=lambda r: ({"high": 0, "medium": 1, "low": 2}.get(r.priority, 1), -r.estimated_monthly_hours_saved)
    )
    return recommendations
