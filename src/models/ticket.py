from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Ticket:
    ticket_number: str
    title: str
    description: str
    account: str
    resources: str
    status: str
    created: datetime
    total_hours: float
    billed_hours: float
    sub_issue_type: str
    issue_type: str

    @property
    def short_description(self) -> str:
        return self.description[:300].strip() if self.description else ""


@dataclass
class Pattern:
    pattern_type: str           # recurring_issue | repeat_contact | high_effort | same_day_burst
    account: str
    issue_type: str
    ticket_count: int
    total_hours: float
    tickets: list = field(default_factory=list)
    contact: Optional[str] = None
    sub_issue_breakdown: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


@dataclass
class Recommendation:
    pattern: Pattern
    pattern_summary: str
    root_cause: str
    recommendation_type: str    # automation | training | process_change | sales_opportunity
    recommended_action: str
    estimated_monthly_tickets_prevented: int
    estimated_monthly_hours_saved: float
    priority: str               # high | medium | low
    source_ticket_numbers: list = field(default_factory=list)
