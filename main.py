"""
NRC AI — Noise Reduction Committee Analysis Tool
Tag Solutions, Albany NY

Usage:
  python main.py models                        List available HatzAI models
  python main.py analyze <csv_file> [...]      Analyze one or more ticket CSV exports
  python main.py analyze <csv_file> --top 10   Limit to top N patterns
  python main.py analyze <csv_file> --no-llm   Pattern stats only, skip LLM calls
"""

import argparse
import sys
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()

from src.hatzai.client import HatzAIClient, HatzAIError
from src.ingest.csv_importer import load_csv, merge_csvs
from src.analysis.patterns import find_patterns
from src.analysis.recommender import generate_recommendations


# ── helpers ────────────────────────────────────────────────────────────────────

PRIORITY_COLOR = {"high": "*** HIGH ***", "medium": "  MEDIUM   ", "low": "   low    "}
REC_TYPE_LABEL = {
    "automation": "Automation",
    "training": "User Training",
    "process_change": "Process Change",
    "sales_opportunity": "Sales Opportunity",
}


def print_header(text: str) -> None:
    width = 78
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_section(text: str) -> None:
    print(f"\n--- {text} ---")


def cmd_models(args) -> None:
    client = HatzAIClient()
    print("Fetching available models from HatzAI...")
    models = client.list_models()
    if not models:
        print("No models returned.")
        return
    rows = []
    for m in models:
        if isinstance(m, dict):
            rows.append([m.get("name", "?"), m.get("display_name", ""), m.get("developer", "")])
        else:
            rows.append([str(m), "", ""])
    print(tabulate(rows, headers=["Model ID (use in .env)", "Display Name", "Developer"], tablefmt="simple"))


def cmd_analyze(args) -> None:
    # ── load data ──────────────────────────────────────────────────────────────
    csv_files = args.csv_files
    print(f"\nLoading {len(csv_files)} CSV file(s)...")
    try:
        df = merge_csvs(csv_files) if len(csv_files) > 1 else load_csv(csv_files[0])
    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        sys.exit(1)

    print(f"Loaded {len(df)} tickets across {df['account'].nunique()} accounts.")

    # ── stats summary ──────────────────────────────────────────────────────────
    print_header("TICKET SUMMARY")
    account_counts = df.groupby("account").size().sort_values(ascending=False).head(10)
    rows = [[acct, cnt] for acct, cnt in account_counts.items()]
    print(tabulate(rows, headers=["Account", "Ticket Count"], tablefmt="simple"))

    issue_counts = df.groupby("issue_type").size().sort_values(ascending=False).head(10)
    rows = [[it, cnt] for it, cnt in issue_counts.items()]
    print(tabulate(rows, headers=["Issue Type", "Count"], tablefmt="simple"))

    # ── pattern detection ──────────────────────────────────────────────────────
    print_header("DETECTED PATTERNS")
    patterns = find_patterns(df)

    if not patterns:
        print("No patterns detected. More ticket data may be needed.")
        return

    top_n = args.top if hasattr(args, "top") and args.top else 20
    patterns = patterns[:top_n]

    pattern_rows = []
    for i, p in enumerate(patterns, 1):
        pattern_rows.append([
            i,
            p.pattern_type,
            p.account[:35],
            p.issue_type[:25],
            p.ticket_count,
            f"{p.total_hours}h",
        ])
    print(tabulate(
        pattern_rows,
        headers=["#", "Pattern Type", "Account", "Issue Type", "Tickets", "Hours"],
        tablefmt="simple",
    ))

    if getattr(args, "no_llm", False):
        print("\n(LLM analysis skipped — use without --no-llm to generate recommendations)")
        return

    # ── LLM recommendations ────────────────────────────────────────────────────
    print_header("GENERATING NRC RECOMMENDATIONS")
    print(f"Sending {len(patterns)} patterns to HatzAI ({HatzAIClient().model})...\n")

    try:
        client = HatzAIClient()
        recommendations = generate_recommendations(patterns, client)
    except HatzAIError as e:
        print(f"ERROR calling HatzAI: {e}")
        sys.exit(1)

    if not recommendations:
        print("No recommendations generated.")
        return

    # ── print recommendations ──────────────────────────────────────────────────
    print_header(f"NRC RECOMMENDATIONS ({len(recommendations)} total)")

    for i, rec in enumerate(recommendations, 1):
        priority_label = PRIORITY_COLOR.get(rec.priority, rec.priority).upper()
        type_label = REC_TYPE_LABEL.get(rec.recommendation_type, rec.recommendation_type)

        print(f"\n{'-'*78}")
        print(f"  #{i}  [{priority_label}]  {type_label}")
        print(f"  Account : {rec.pattern.account}")
        print(f"  Tickets : {', '.join(rec.source_ticket_numbers[:8])}"
              + (" ..." if len(rec.source_ticket_numbers) > 8 else ""))
        print(f"  Impact  : ~{rec.estimated_monthly_tickets_prevented} tickets/mo  |  "
              f"~{rec.estimated_monthly_hours_saved}h saved/mo")
        print()
        print(f"  PATTERN\n  {rec.pattern_summary}")
        print()
        print(f"  ROOT CAUSE\n  {rec.root_cause}")
        print()
        print(f"  RECOMMENDED ACTION\n  {rec.recommended_action}")

    print(f"\n{'-'*78}")
    total_tickets = sum(r.estimated_monthly_tickets_prevented for r in recommendations)
    total_hours = sum(r.estimated_monthly_hours_saved for r in recommendations)
    print(f"\n  TOTAL ESTIMATED IMPACT (monthly)")
    print(f"  Tickets prevented : ~{total_tickets}")
    print(f"  Hours saved       : ~{total_hours:.1f}h")
    print()


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NRC AI — Noise Reduction Committee Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("models", help="List available HatzAI models")

    analyze_p = sub.add_parser("analyze", help="Analyze ticket CSV exports")
    analyze_p.add_argument("csv_files", nargs="+", metavar="CSV_FILE")
    analyze_p.add_argument("--top", type=int, default=20, metavar="N", help="Max patterns to analyze (default 20)")
    analyze_p.add_argument("--no-llm", action="store_true", help="Skip LLM calls, show pattern stats only")

    args = parser.parse_args()

    if args.command == "models":
        cmd_models(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
