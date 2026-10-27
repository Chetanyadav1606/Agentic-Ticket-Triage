"""Data loaders. Loading only - no joins, matching or business logic.

Resolving accounts, interpreting usage and finding related tickets is part of
the task. Note the realistic noise described in data/README.md.

Uses the stdlib csv module (rows as list[dict]) instead of pandas so the agent
has no compiled dependencies (pandas DLLs are blocked on some locked-down hosts).
"""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def _read(name: str) -> list[dict]:
    with open(DATA / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_accounts() -> list[dict]:
    return _read("accounts.csv")


def load_tickets() -> list[dict]:
    return _read("tickets.csv")


def load_usage() -> list[dict]:
    rows = _read("usage_metrics.csv")
    for r in rows:
        for k in ("active_users", "documents_processed", "tickets_opened"):
            r[k] = int(r[k])
        r["login_rate_pct"] = float(r["login_rate_pct"])
    return rows


def load_review_set() -> list[str]:
    return [l.strip() for l in (DATA / "review_set.txt").read_text().splitlines() if l.strip()]


AS_OF_DATE = "2026-10-20"  # treat this as "today" for all date calculations
