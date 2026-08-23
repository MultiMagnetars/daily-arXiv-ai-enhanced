#!/usr/bin/env python3
"""Controlled SciX -> FILTER_KEYWORDS -> AI validation harness.

The harness reuses the production client, source merge, keyword filter,
prompt, Structure, and single-item AI processing. It never writes production
data files or performs git operations.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as datetime_module
import importlib.util
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .scix_client import ScixClient
    from .source_merge import (
        history_keys,
        load_jsonl,
        merge_sources,
        normalize_arxiv_id,
        normalize_doi,
    )
except ImportError:
    from scix_client import ScixClient
    from source_merge import (
        history_keys,
        load_jsonl,
        merge_sources,
        normalize_arxiv_id,
        normalize_doi,
    )


MAX_AI_ITEMS = 2
SCIX_ROWS = 10
SCIX_MAX_PAGES = 1
HISTORY_DAYS = 7
ABSTRACT_TRANSLATION_PREVIEW_LIMIT = 200
REQUIRED_AI_FIELDS: tuple[str, ...] = (
    "abstract_translation",
    "tldr",
    "motivation",
    "method",
    "result",
    "conclusion",
)
PASS_STATUSES = {
    "PASS_E2E",
    "PASS_E2E_LIMITED",
    "PASS_NO_AI_CANDIDATES",
}
_RAW_ARXIV_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_raw\.jsonl$")


@dataclass
class ValidationResult:
    status: str
    report: dict[str, Any]
    exit_code: int


class OutputSchemaValidationError(Exception):
    """Raised when a validation AI result is not the exact six-field schema."""


def resolve_utc_date(value: datetime_module.date | datetime_module.datetime | None = None) -> datetime_module.date:
    if value is None:
        return datetime_module.datetime.now(datetime_module.timezone.utc).date()
    if isinstance(value, datetime_module.datetime):
        if value.tzinfo is not None:
            return value.astimezone(datetime_module.timezone.utc).date()
        return value.date()
    return value


def _secret_values(secret: str | Sequence[str] | None) -> tuple[str, ...]:
    if secret is None:
        return ()
    if isinstance(secret, str):
        return (secret,) if secret else ()
    return tuple(value for value in secret if value)


def _redact(text: str, secret: str | Sequence[str] | None = None) -> str:
    for value in _secret_values(secret):
        text = text.replace(value, "[REDACTED]")
    for marker in ("Authorization", "authorization", "Bearer"):
        text = text.replace(marker, "[REDACTED]")
    return text


def _safe_json(value: Any, secret: str | Sequence[str] | None = None) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = repr(value)
    return _redact(rendered, secret)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(part for part in (_text(item) for item in value) if part)
    if value is None:
        return ""
    return str(value).strip()


def matched_keywords(paper: Mapping[str, Any], keywords: Sequence[str]) -> list[str]:
    """Report matches using the same title+summary OR semantics as enhance.py."""
    searchable_text = (
        f"{str(paper.get('title') or '')} "
        f"{str(paper.get('summary') or '')}"
    ).casefold()
    return [
        keyword
        for keyword in keywords
        if keyword.casefold() in searchable_text
    ]


def load_latest_arxiv_raw(history_dir: str | Path | None) -> tuple[list[dict[str, Any]], str | None]:
    if history_dir is None:
        return [], None
    root = Path(history_dir)
    candidates: list[tuple[datetime_module.date, Path]] = []
    for path in root.glob("*_raw.jsonl"):
        match = _RAW_ARXIV_RE.match(path.name)
        if match:
            candidates.append((datetime_module.date.fromisoformat(match.group("date")), path))
    if not candidates:
        return [], None
    source_date, source_path = max(candidates, key=lambda item: item[0])
    return load_jsonl(source_path), source_date.isoformat()


def load_history_keys(
    history_dir: str | Path | None,
    run_date: datetime_module.date,
) -> set[str]:
    if history_dir is None:
        return set()
    root = Path(history_dir)
    keys: set[str] = set()
    for offset in range(1, HISTORY_DAYS + 1):
        history_date = run_date - datetime_module.timedelta(days=offset)
        history_file = root / f"{history_date.isoformat()}.jsonl"
        for paper in load_jsonl(history_file):
            keys.update(history_keys(paper))
    return keys


def _validate_ai_output(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise OutputSchemaValidationError("AI result is not a mapping")
    if set(value.keys()) != set(REQUIRED_AI_FIELDS):
        raise OutputSchemaValidationError("AI result does not have exactly six fields")
    if not all(isinstance(value[field], str) for field in REQUIRED_AI_FIELDS):
        raise OutputSchemaValidationError("AI result contains a non-string field")
    return {field: value[field] for field in REQUIRED_AI_FIELDS}


def _ai_summary(
    paper: Mapping[str, Any],
    ai_data: Mapping[str, str],
    keywords: Sequence[str],
    secret: str | Sequence[str] | None,
) -> dict[str, Any]:
    abstract_translation = ai_data["abstract_translation"]
    return {
        "title": _redact(_text(paper.get("title")), secret),
        "source": _redact(_text(paper.get("source")), secret),
        "bibcode": _redact(_text(paper.get("bibcode")), secret),
        "doi": _redact(_text(paper.get("doi")) or (_text(paper.get("identifiers")) if not paper.get("doi") else ""), secret),
        "arxiv_id": normalize_arxiv_id(
            paper.get("id") or paper.get("identifiers")
        ),
        "matched_keywords": list(matched_keywords(paper, keywords)),
        "abstract_translation_present": bool(abstract_translation),
        "abstract_translation_length": len(abstract_translation),
        "abstract_translation_preview": _redact(
            abstract_translation[:ABSTRACT_TRANSLATION_PREVIEW_LIMIT],
            secret,
        ),
        "tldr_present": bool(ai_data["tldr"]),
        "motivation_present": bool(ai_data["motivation"]),
        "method_present": bool(ai_data["method"]),
        "result_present": bool(ai_data["result"]),
        "conclusion_present": bool(ai_data["conclusion"]),
        "tldr": _redact(ai_data["tldr"], secret),
    }


def _base_report(
    *,
    run_date: datetime_module.date,
    scix_result: Any,
    arxiv_raw_source_date: str | None,
    secret: str | Sequence[str] | None,
) -> dict[str, Any]:
    docs = getattr(scix_result, "docs", []) or []
    client_status = str(getattr(scix_result, "status", "") or "")
    return {
        "status": "",
        "run_date_utc": run_date.isoformat(),
        "scix_status": client_status,
        "scix_error_type": (
            type(getattr(scix_result, "error", None)).__name__
            if getattr(scix_result, "error", "")
            else ""
        ),
        "scix_num_found": int(getattr(scix_result, "num_found", 0) or 0),
        "scix_docs_received": len(docs),
        "limited_fetch": client_status == "truncated",
        "scix_rows": SCIX_ROWS,
        "scix_max_pages": SCIX_MAX_PAGES,
        "arxiv_raw_source_date": arxiv_raw_source_date or "",
        "cross_source_validation": "performed" if arxiv_raw_source_date else "skipped",
        "real_cross_source_duplicate_found": False,
        "canonical_before_merge": 0,
        "canonical_after_merge": 0,
        "canonical_before_history_dedup": 0,
        "canonical_after_history_dedup": 0,
        "history_duplicates_removed": 0,
        "filter_keyword_count": 0,
        "eligible_before_cap": 0,
        "eligible_after_filter": 0,
        "processed_after_cap": 0,
        "ai_invocations": 0,
        "capped": False,
        "matched_keyword_counts": {},
        "papers": [],
    }


def _write_report(
    report: dict[str, Any],
    output_dir: str | Path | None,
) -> None:
    if output_dir is None:
        return
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "scix_e2e_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _finish(
    report: dict[str, Any],
    status: str,
    output_dir: str | Path | None,
) -> ValidationResult:
    report["status"] = status
    _write_report(report, output_dir)
    return ValidationResult(
        status=status,
        report=report,
        exit_code=0 if status in PASS_STATUSES else 1,
    )


def run_validation(
    *,
    scix_client: Any,
    arxiv_records: Sequence[Mapping[str, Any]],
    arxiv_raw_source_date: str | None,
    history_dir: str | Path | None,
    filter_keywords_raw: str | None,
    filter_parser: Callable[[str | None], list[str]],
    filter_function: Callable[
        [list[dict[str, Any]], list[str]],
        tuple[list[dict[str, Any]], dict[str, int]],
    ],
    ai_runner_factory: Callable[[], Callable[[Mapping[str, Any]], Mapping[str, Any]]] | None,
    run_date: datetime_module.date | datetime_module.datetime | None = None,
    output_dir: str | Path | None = None,
    secret: str | Sequence[str] | None = None,
) -> ValidationResult:
    """Run bounded validation with production modules and injected test seams."""
    effective_date = resolve_utc_date(run_date)
    try:
        scix_result = scix_client.fetch(
            (effective_date - datetime_module.timedelta(days=3)).isoformat(),
            effective_date.isoformat(),
        )
    except Exception as exc:
        report = {
            "status": "FAIL_SCIX",
            "scix_error_type": type(exc).__name__,
            "scix_docs_received": 0,
            "arxiv_raw_source_date": arxiv_raw_source_date or "",
        }
        _write_report(report, output_dir)
        return ValidationResult("FAIL_SCIX", report, 1)

    report = _base_report(
        run_date=effective_date,
        scix_result=scix_result,
        arxiv_raw_source_date=arxiv_raw_source_date,
        secret=secret,
    )
    client_status = str(getattr(scix_result, "status", "") or "")
    scix_docs = list(getattr(scix_result, "docs", []) or [])
    if client_status not in {"ok", "success_empty", "truncated"}:
        return _finish(report, "FAIL_SCIX", output_dir)
    if client_status == "truncated" and not scix_docs:
        return _finish(report, "FAIL_SCIX", output_dir)

    try:
        merged = merge_sources(list(arxiv_records), scix_docs)
    except Exception as exc:
        report["merge_error_type"] = type(exc).__name__
        return _finish(report, "FAIL_MERGE", output_dir)

    report["canonical_before_merge"] = len(merged.records)
    report["canonical_after_merge"] = len(merged.records)
    report["real_cross_source_duplicate_found"] = any(
        _text(record.get("source")) == "arxiv+scix"
        for record in merged.records
    )

    try:
        history_key_set = load_history_keys(history_dir, effective_date)
        after_history = [
            record
            for record in merged.records
            if not history_keys(record).intersection(history_key_set)
        ]
    except Exception as exc:
        report["history_error_type"] = type(exc).__name__
        return _finish(report, "FAIL_HISTORY", output_dir)

    report["canonical_before_history_dedup"] = len(merged.records)
    report["canonical_after_history_dedup"] = len(after_history)
    report["history_duplicates_removed"] = len(merged.records) - len(after_history)

    try:
        eligible_input = [
            dict(record)
            for record in after_history
            if _text(record.get("title")) and _text(record.get("summary"))
        ]
        keywords = filter_parser(filter_keywords_raw)
        filtered, hit_counts = filter_function(eligible_input, keywords)
    except Exception as exc:
        report["filter_error_type"] = type(exc).__name__
        return _finish(report, "FAIL_FILTER", output_dir)

    report["filter_keyword_count"] = len(keywords)
    report["eligible_after_filter"] = len(filtered)
    report["matched_keyword_counts"] = dict(hit_counts)
    report["eligible_before_cap"] = len(filtered)
    selected = filtered[:MAX_AI_ITEMS]
    report["processed_after_cap"] = len(selected)
    report["capped"] = len(filtered) > MAX_AI_ITEMS

    if not selected:
        return _finish(report, "PASS_NO_AI_CANDIDATES", output_dir)
    if ai_runner_factory is None:
        return _finish(report, "FAIL_AI", output_dir)

    try:
        ai_runner = ai_runner_factory()
    except Exception as exc:
        report["ai_error_type"] = type(exc).__name__
        return _finish(report, "FAIL_AI", output_dir)

    for paper in selected:
        report["ai_invocations"] += 1
        try:
            ai_data = _validate_ai_output(ai_runner(paper))
        except OutputSchemaValidationError as exc:
            report["output_schema_error_type"] = type(exc).__name__
            return _finish(report, "FAIL_OUTPUT_SCHEMA", output_dir)
        except Exception as exc:
            report["ai_error_type"] = type(exc).__name__
            return _finish(report, "FAIL_AI", output_dir)
        report["papers"].append(_ai_summary(paper, ai_data, keywords, secret))

    status = "PASS_E2E_LIMITED" if client_status == "truncated" else "PASS_E2E"
    return _finish(report, status, output_dir)


def _load_enhance_module() -> Any:
    ai_dir = Path(__file__).resolve().parents[2] / "ai"
    module_name = "p5a4_enhance_runtime"
    spec = importlib.util.spec_from_file_location(module_name, ai_dir / "enhance.py")
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load ai/enhance.py")
    previous_cwd = Path.cwd()
    sys.path.insert(0, str(ai_dir))
    try:
        os.chdir(ai_dir)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(previous_cwd)
        if sys.path and sys.path[0] == str(ai_dir):
            sys.path.pop(0)


def _build_ai_runner(
    enhance_module: Any,
    *,
    model_name: str,
    language: str,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    llm = enhance_module.ChatOpenAI(
        model=model_name,
        model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
    ).with_structured_output(
        enhance_module.Structure,
        method="function_calling",
    )
    prompt_template = enhance_module.ChatPromptTemplate.from_messages([
        enhance_module.SystemMessagePromptTemplate.from_template(
            enhance_module.system
        ),
        enhance_module.HumanMessagePromptTemplate.from_template(
            template=enhance_module.template
        ),
    ])
    chain = prompt_template | llm

    def run_one(item: Mapping[str, Any]) -> Mapping[str, Any]:
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            processed = enhance_module.process_single_item(
                chain,
                dict(item),
                language,
            )
        diagnostics = captured.getvalue()
        if "Unexpected error for" in diagnostics:
            raise RuntimeError("AI invocation failed")
        if "Using partial AI data" in diagnostics or "Failed to parse JSON" in diagnostics:
            raise OutputSchemaValidationError("AI parser fallback was used")
        return processed.get("AI", {})

    return run_one


def create_scix_client() -> ScixClient:
    """Create the validation-only bounded client without changing production defaults."""

    return ScixClient(
        rows=SCIX_ROWS,
        max_pages=SCIX_MAX_PAGES,
        retries=0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Controlled SciX E2E validation")
    parser.add_argument("--history-dir", help="Read-only origin/data snapshot directory")
    parser.add_argument("--output-dir", required=True, help="Runner temp output directory")
    args = parser.parse_args(argv)

    run_date = resolve_utc_date()
    try:
        arxiv_records, arxiv_raw_source_date = load_latest_arxiv_raw(args.history_dir)
        enhance_module = _load_enhance_module()
        scix_client = create_scix_client()
        secret_values = (
            os.environ.get("SCIX_API_TOKEN", ""),
            os.environ.get("OPENAI_API_KEY", ""),
            os.environ.get("OPENAI_BASE_URL", ""),
        )
        model_name = os.environ.get("MODEL_NAME", "deepseek-chat")
        language = os.environ.get("LANGUAGE", "Chinese")

        result = run_validation(
            scix_client=scix_client,
            arxiv_records=arxiv_records,
            arxiv_raw_source_date=arxiv_raw_source_date,
            history_dir=args.history_dir,
            filter_keywords_raw=os.environ.get("FILTER_KEYWORDS"),
            filter_parser=enhance_module.parse_filter_keywords,
            filter_function=enhance_module.filter_papers_by_keywords,
            ai_runner_factory=lambda: _build_ai_runner(
                enhance_module,
                model_name=model_name,
                language=language,
            ),
            run_date=run_date,
            output_dir=args.output_dir,
            secret=secret_values,
        )
    except Exception as exc:
        result = ValidationResult(
            "FAIL_HISTORY",
            {
                "status": "FAIL_HISTORY",
                "error_type": type(exc).__name__,
                "arxiv_raw_source_date": "",
            },
            1,
        )
        _write_report(result.report, args.output_dir)

    print(json.dumps(result.report, ensure_ascii=False, indent=2))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
