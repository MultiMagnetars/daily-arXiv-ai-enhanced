#!/usr/bin/env python3
"""Safe, read-only SciX API smoke-test entry point.

This module deliberately stops at inspecting one API response.  It does not
write repository data, invoke source merging, call AI services, or perform
any git operation.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Sequence

try:  # Package execution: python -m daily_arxiv.daily_arxiv.scix_smoke
    from .scix_client import ScixClient, build_scix_query
    from .source_merge import normalize_arxiv_id, normalize_doi
except ImportError:  # Direct script execution from the package directory
    from scix_client import ScixClient, build_scix_query
    from source_merge import normalize_arxiv_id, normalize_doi


SMOKE_ROWS = 1
SMOKE_MAX_PAGES = 1
SMOKE_RETRIES = 0
SMOKE_LOOKBACK_DAYS = 3
ABSTRACT_PREVIEW_LIMIT = 240

PASS_STATUSES = {"PASS_API_CONNECTION", "PASS_EMPTY"}
INSPECTED_FIELDS: tuple[str, ...] = (
    "bibcode",
    "title",
    "abstract",
    "author",
    "doi",
    "identifier",
    "year",
    "pub",
    "pubdate",
    "entdate",
    "database",
    "doctype",
    "property",
    "esources",
)


@dataclass
class SmokeRun:
    """A rendered smoke result and its safe process status."""

    status: str
    report: str
    start_date: str
    end_date: str
    client_result: Any


def resolve_utc_date(value: date | datetime | None = None) -> date:
    """Resolve a supplied date or the current UTC calendar date."""

    if value is None:
        return datetime.now(timezone.utc).date()
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).date()
        return value.date()
    return value


def smoke_window(run_date: date | datetime | None = None) -> tuple[str, str]:
    """Return the same UTC lookback convention used by the P5A pipeline."""

    end = resolve_utc_date(run_date)
    start = end - timedelta(days=SMOKE_LOOKBACK_DAYS)
    return start.isoformat(), end.isoformat()


def _redact(text: str, secret: str) -> str:
    """Remove credentials and header-like wording from rendered output."""

    if secret:
        text = text.replace(secret, "[REDACTED]")
    for marker in ("Authorization", "authorization", "Bearer"):
        text = text.replace(marker, "[REDACTED]")
    return text


def _safe_json(value: Any, secret: str) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = repr(value)
    return _redact(rendered, secret)


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(part for part in (_text_content(item) for item in value) if part)
    if value is None:
        return ""
    return str(value).strip()


def _abstract_summary(value: Any, secret: str) -> tuple[bool, int, str]:
    text = _text_content(value)
    preview = " ".join(text.split())[:ABSTRACT_PREVIEW_LIMIT]
    return bool(text), len(text), _redact(preview, secret)


def classify_client_result(result: Any) -> str:
    """Map the existing client status to the smoke-test status contract."""

    client_status = getattr(result, "status", "")
    docs = getattr(result, "docs", []) or []
    error = str(getattr(result, "error", "") or "")

    if client_status == "ok":
        # The request asks for one row.  Treat a server response with more
        # than one document as a response-shape violation for this smoke test.
        return "PASS_API_CONNECTION" if len(docs) <= 1 else "FAIL_RESPONSE_SHAPE"
    if client_status == "success_empty":
        return "PASS_EMPTY"
    if client_status == "auth_error":
        return "FAIL_AUTH"
    if client_status == "rate_limited":
        return "FAIL_RATE_LIMIT"
    if client_status == "unavailable":
        if error.startswith("Invalid SciX response") or "docs is not a list" in error:
            return "FAIL_RESPONSE_SHAPE"
        return "FAIL_NETWORK"
    if client_status == "truncated":
        return "FAIL_RESPONSE_SHAPE"
    return "FAIL_RESPONSE_SHAPE"


def render_report(
    result: Any,
    *,
    start_date: str,
    end_date: str,
    secret: str = "",
) -> str:
    """Render only bounded, non-credential response inspection output."""

    status = classify_client_result(result)
    docs = getattr(result, "docs", []) or []
    num_found = getattr(result, "num_found", 0)
    client_status = getattr(result, "status", "")
    error = _redact(str(getattr(result, "error", "") or ""), secret)

    lines = [
        "SciX smoke test report",
        f"status: {status}",
        f"client_status: {client_status}",
        f"numFound: {num_found}",
        f"docs_received: {len(docs)}",
        "rows: 1",
        "max_pages: 1",
        f"query_start_utc: {start_date}",
        f"query_end_utc: {end_date}",
        f"query: {_redact(build_scix_query(start_date, end_date), secret)}",
    ]
    if error:
        lines.append(f"error: {error}")

    if not docs:
        return "\n".join(lines) + "\n"

    record = docs[0]
    lines.append("record_inspected: 1")
    for field_name in INSPECTED_FIELDS:
        value = record.get(field_name) if isinstance(record, Mapping) else None
        lines.append(f"{field_name}_type: {type(value).__name__}")
        if field_name == "abstract":
            present, length, preview = _abstract_summary(value, secret)
            lines.append(f"abstract_present: {str(present).lower()}")
            lines.append(f"abstract_length: {length}")
            if preview:
                lines.append(f"abstract_preview: {json.dumps(preview, ensure_ascii=False)}")
        else:
            lines.append(f"{field_name}: {_safe_json(value, secret)}")

    raw_doi = record.get("doi") if isinstance(record, Mapping) else None
    lines.append(f"doi_raw: {_safe_json(raw_doi, secret)}")
    lines.append(f"normalized_doi: {_safe_json(normalize_doi(raw_doi), secret)}")

    raw_identifier = record.get("identifier") if isinstance(record, Mapping) else None
    normalized_arxiv_id = normalize_arxiv_id(raw_identifier)
    lines.append(
        f"normalized_arxiv_id: {_safe_json(normalized_arxiv_id, secret)}"
    )
    if normalized_arxiv_id:
        # These are deterministic arXiv URLs only; the smoke test never opens
        # them or performs any follow-up request.
        lines.append(
            f"arxiv_abs_url: https://arxiv.org/abs/{normalized_arxiv_id}"
        )
        lines.append(
            f"arxiv_pdf_url: https://arxiv.org/pdf/{normalized_arxiv_id}"
        )
    return "\n".join(lines) + "\n"


def run_smoke(
    *,
    client_factory: Callable[..., Any] = ScixClient,
    run_date: date | datetime | None = None,
    token: str | None = None,
) -> SmokeRun:
    """Run one bounded fetch through the existing :class:`ScixClient`."""

    start_date, end_date = smoke_window(run_date)
    secret = os.environ.get("SCIX_API_TOKEN", "") if token is None else token
    client = client_factory(
        token=secret,
        rows=SMOKE_ROWS,
        max_pages=SMOKE_MAX_PAGES,
        retries=SMOKE_RETRIES,
    )
    result = client.fetch(start_date, end_date)
    report = render_report(
        result,
        start_date=start_date,
        end_date=end_date,
        secret=secret,
    )
    return SmokeRun(
        status=classify_client_result(result),
        report=report,
        start_date=start_date,
        end_date=end_date,
        client_result=result,
    )


def main() -> int:
    smoke = run_smoke()
    sys.stdout.write(smoke.report)
    return 0 if smoke.status in PASS_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())
