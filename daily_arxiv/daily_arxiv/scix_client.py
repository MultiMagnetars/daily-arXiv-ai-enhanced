#!/usr/bin/env python3
"""Small, injectable SciX/ADS search client.

The module deliberately keeps HTTP transport separate from normalization.  The
production entry point uses requests, while tests can inject a fake session so
no network is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    import requests
except ImportError:  # pragma: no cover - the project already uses requests
    requests = None  # type: ignore[assignment]


SCIX_API_URL = "https://api.adsabs.harvard.edu/v1/search/query"
DEFAULT_ROWS = 100
DEFAULT_MAX_PAGES = 50
DEFAULT_TIMEOUT = 20.0
DEFAULT_RETRIES = 2

SCIX_FIELDS: tuple[str, ...] = (
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

TOPICAL_TERMS: tuple[str, ...] = (
    "pulsar",
    "magnetar",
    "neutron star",
    "fast radio burst",
    "FRB",
    "radio transient",
)


@dataclass
class ScixFetchResult:
    """Serializable result of one candidate-retrieval operation."""

    docs: list[dict[str, Any]] = field(default_factory=list)
    status: str = "unavailable"
    num_found: int = 0
    pages: int = 0
    truncated: bool = False
    error: str = ""

    def status_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "num_found": self.num_found,
            "pages": self.pages,
            "returned": len(self.docs),
            "truncated": self.truncated,
            "error": self.error,
        }


def build_scix_query(start_date: str, end_date: str) -> str:
    """Build a broad topical candidate query.

    The topical clause intentionally stays smaller than FILTER_KEYWORDS.  It
    only limits candidate retrieval; the existing local filter remains the
    final business rule.
    """

    topical_clause = " OR ".join(
        f'abs:"{term}"' if " " in term else f"abs:{term}"
        for term in TOPICAL_TERMS
    )
    return (
        "database:astronomy "
        f"AND entdate:[{start_date} TO {end_date}] "
        f"AND ({topical_clause})"
    )


def _validate_date(value: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date: {value}") from exc
    return value


class ScixClient:
    """Fetch SciX/ADS candidate records with injectable HTTP and sleeping."""

    def __init__(
        self,
        token: str | None = None,
        *,
        session: Any | None = None,
        endpoint: str = SCIX_API_URL,
        rows: int = DEFAULT_ROWS,
        max_pages: int = DEFAULT_MAX_PAGES,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if rows <= 0:
            raise ValueError("rows must be positive")
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        if retries < 0:
            raise ValueError("retries cannot be negative")

        self.token = token if token is not None else os.environ.get("SCIX_API_TOKEN", "")
        self.endpoint = endpoint
        self.rows = rows
        self.max_pages = max_pages
        self.timeout = timeout
        self.retries = retries
        self.sleep_fn = sleep_fn

        if session is not None:
            self.session = session
        elif requests is not None:
            self.session = requests.Session()
        else:  # pragma: no cover - only relevant to an incomplete environment
            self.session = None

    def fetch(self, start_date: str, end_date: str) -> ScixFetchResult:
        """Fetch all pages within the configured safety limit."""

        _validate_date(start_date)
        _validate_date(end_date)
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date")

        if not self.token:
            return ScixFetchResult(
                status="unavailable",
                error="SCIX_API_TOKEN is not set",
            )
        if self.session is None:
            return ScixFetchResult(
                status="unavailable",
                error="requests is not available",
            )

        query = build_scix_query(start_date, end_date)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        params_base = {
            "q": query,
            "fl": ",".join(SCIX_FIELDS),
            "rows": self.rows,
            "sort": "entdate desc",
        }

        docs: list[dict[str, Any]] = []
        num_found = 0
        pages = 0
        start = 0

        while pages < self.max_pages:
            params = {**params_base, "start": start}
            response_or_result = self._request_with_retries(
                headers=headers,
                params=params,
            )
            if isinstance(response_or_result, ScixFetchResult):
                response_or_result.docs = docs
                response_or_result.num_found = num_found
                response_or_result.pages = pages
                return response_or_result

            response = response_or_result
            pages += 1
            try:
                payload = response.json()
                response_data = payload.get("response", {}) or {}
                page_docs = response_data.get("docs", []) or []
                page_num_found = int(response_data.get("numFound", 0) or 0)
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return ScixFetchResult(
                    docs=docs,
                    status="unavailable",
                    num_found=num_found,
                    pages=pages,
                    error=f"Invalid SciX response: {type(exc).__name__}",
                )

            if not isinstance(page_docs, list):
                return ScixFetchResult(
                    docs=docs,
                    status="unavailable",
                    num_found=num_found,
                    pages=pages,
                    error="SciX response docs is not a list",
                )

            num_found = max(num_found, page_num_found)
            docs.extend(doc for doc in page_docs if isinstance(doc, dict))

            if not page_docs or len(docs) >= num_found:
                return ScixFetchResult(
                    docs=docs,
                    status="success_empty" if not docs else "ok",
                    num_found=num_found,
                    pages=pages,
                )

            start += self.rows

        return ScixFetchResult(
            docs=docs,
            status="truncated",
            num_found=num_found,
            pages=pages,
            truncated=True,
            error=f"Reached MAX_PAGES={self.max_pages}",
        )

    def _request_with_retries(
        self,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, Any],
    ) -> Any | ScixFetchResult:
        attempts = 0
        while True:
            try:
                response = self.session.get(  # type: ignore[union-attr]
                    self.endpoint,
                    params=dict(params),
                    headers=dict(headers),
                    timeout=self.timeout,
                )
            except Exception as exc:  # transport implementations vary in tests
                if self._is_timeout(exc) and attempts < self.retries:
                    self._sleep_for_retry(attempts)
                    attempts += 1
                    continue
                if attempts < self.retries and self._is_retryable_exception(exc):
                    self._sleep_for_retry(attempts)
                    attempts += 1
                    continue
                return ScixFetchResult(
                    status="unavailable",
                    error=f"HTTP transport error: {type(exc).__name__}",
                )

            status_code = int(getattr(response, "status_code", 0) or 0)
            if status_code in (401, 403):
                return ScixFetchResult(
                    status="auth_error",
                    error=f"HTTP {status_code}",
                )
            if status_code == 429:
                if attempts < self.retries:
                    self._sleep_for_retry(attempts, response)
                    attempts += 1
                    continue
                return ScixFetchResult(
                    status="rate_limited",
                    error="HTTP 429",
                )
            if 500 <= status_code <= 599:
                if attempts < self.retries:
                    self._sleep_for_retry(attempts)
                    attempts += 1
                    continue
                return ScixFetchResult(
                    status="unavailable",
                    error=f"HTTP {status_code}",
                )
            if status_code < 200 or status_code >= 300:
                return ScixFetchResult(
                    status="unavailable",
                    error=f"HTTP {status_code}",
                )
            return response

    def _sleep_for_retry(self, attempt: int, response: Any | None = None) -> None:
        delay = 2.0**attempt
        if response is not None:
            raw_retry_after = getattr(response, "headers", {}).get("Retry-After")
            try:
                if raw_retry_after is not None:
                    delay = max(0.0, min(float(raw_retry_after), 60.0))
            except (TypeError, ValueError):
                pass
        self.sleep_fn(delay)

    @staticmethod
    def _is_timeout(exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        return requests is not None and isinstance(exc, requests.exceptions.Timeout)

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        if isinstance(exc, (OSError, ConnectionError)):
            return True
        return requests is not None and isinstance(exc, requests.exceptions.RequestException)


def write_raw_jsonl(docs: Sequence[Mapping[str, Any]], output: str | Path) -> None:
    """Write only API docs; never write headers, token, or exceptions."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(dict(doc), ensure_ascii=False) + "\n")


def write_status(result: ScixFetchResult, output: str | Path | None) -> None:
    if output is None:
        return
    status_path = Path(output)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(result.status_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch SciX/ADS candidate records")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True, help="SciX raw JSONL path")
    parser.add_argument("--status-file", help="Optional safe status JSON path")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)

    client = ScixClient(
        rows=args.rows,
        max_pages=args.max_pages,
        timeout=args.timeout,
    )
    result = client.fetch(args.start_date, args.end_date)
    write_raw_jsonl(result.docs, args.output)
    write_status(result, args.status_file)
    print(
        f"SciX status={result.status} docs={len(result.docs)} "
        f"numFound={result.num_found} pages={result.pages} "
        f"truncated={result.truncated}",
        file=sys.stderr,
    )
    if result.error:
        print(f"SciX warning: {result.error}", file=sys.stderr)

    # Expected remote failures are handled by the arXiv fallback.  The CLI
    # therefore stays successful so workflow orchestration can continue.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
