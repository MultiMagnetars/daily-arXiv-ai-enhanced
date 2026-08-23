#!/usr/bin/env python3
"""Prepare and merge same-day AI JSONL results without reprocessing old papers."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .source_merge import history_keys
except ImportError:  # Script execution from the daily_arxiv directory.
    from source_merge import history_keys


class SameDayMergeError(ValueError):
    """Raised when a same-day JSONL input cannot be trusted."""


def _read_jsonl(path: str | Path, *, required: bool) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        if required:
            raise SameDayMergeError(f"required JSONL file missing: {source}")
        return []
    if not source.is_file():
        raise SameDayMergeError(f"JSONL path is not a file: {source}")

    records: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SameDayMergeError(
                        f"malformed JSONL file: {source} at line {line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise SameDayMergeError(
                        f"JSONL record is not an object: {source} at line {line_number}"
                    )
                records.append(value)
    except OSError as exc:
        raise SameDayMergeError(f"unable to read JSONL file: {source}") from exc
    return records


def _read_existing_today_ai(path: str | Path) -> list[dict[str, Any]]:
    try:
        return _read_jsonl(path, required=False)
    except SameDayMergeError as exc:
        raise SameDayMergeError(
            f"existing same-day AI file malformed: {Path(path)}"
        ) from exc


def _read_new_ai_staging(path: str | Path) -> list[dict[str, Any]]:
    try:
        return _read_jsonl(path, required=True)
    except SameDayMergeError as exc:
        raise SameDayMergeError(
            f"new AI staging file malformed or missing: {Path(path)}"
        ) from exc


def _record_keys(record: Mapping[str, Any]) -> set[str]:
    try:
        return history_keys(record)
    except Exception as exc:  # Keep malformed identity failures fail-closed.
        raise SameDayMergeError("record identity normalization failed") from exc


def subtract_records(
    current_candidates: Sequence[Mapping[str, Any]],
    existing_today_ai: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return current records whose canonical keys are absent from existing AI."""

    existing_keys: set[str] = set()
    for record in existing_today_ai:
        existing_keys.update(_record_keys(record))

    new_candidates: list[dict[str, Any]] = []
    for record in current_candidates:
        if not _record_keys(record).intersection(existing_keys):
            new_candidates.append(dict(record))
    return new_candidates


def merge_records(
    existing_today_ai: Sequence[Mapping[str, Any]],
    new_ai_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge existing-first AI records using the shared canonical identity."""

    merged = [dict(record) for record in existing_today_ai]
    known_keys: set[str] = set()
    for record in existing_today_ai:
        known_keys.update(_record_keys(record))

    for record in new_ai_records:
        record_keys = _record_keys(record)
        if record_keys and record_keys.intersection(known_keys):
            continue
        merged.append(dict(record))
        known_keys.update(record_keys)
    return merged


def _write_jsonl_atomic(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            for record in records:
                handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        os.replace(temporary_path, output)
        temporary_path = None
    except OSError as exc:
        raise SameDayMergeError(f"unable to write same-day JSONL output: {output}") from exc
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def subtract_files(
    current_path: str | Path,
    existing_path: str | Path,
    output_path: str | Path,
) -> int:
    current = _read_jsonl(current_path, required=True)
    existing = _read_existing_today_ai(existing_path)
    new_candidates = subtract_records(current, existing)
    _write_jsonl_atomic(new_candidates, output_path)
    print(
        f"same-day subtract current={len(current)} existing={len(existing)} "
        f"new={len(new_candidates)}",
        file=sys.stderr,
    )
    return len(new_candidates)


def merge_files(
    existing_path: str | Path,
    new_ai_path: str | Path,
    output_path: str | Path,
) -> int:
    existing = _read_existing_today_ai(existing_path)
    new_ai = _read_new_ai_staging(new_ai_path)
    merged = merge_records(existing, new_ai)
    _write_jsonl_atomic(merged, output_path)
    print(
        f"same-day merge existing={len(existing)} new={len(new_ai)} "
        f"final={len(merged)}",
        file=sys.stderr,
    )
    return len(merged)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Same-day AI JSONL subtraction and merge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subtract = subparsers.add_parser("subtract", help="remove papers already in existing same-day AI")
    subtract.add_argument("--current", required=True)
    subtract.add_argument("--existing", required=True)
    subtract.add_argument("--output", required=True)

    merge = subparsers.add_parser("merge", help="merge existing and newly AI-enhanced papers")
    merge.add_argument("--existing", required=True)
    merge.add_argument("--new-ai", required=True)
    merge.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "subtract":
            subtract_files(args.current, args.existing, args.output)
        else:
            merge_files(args.existing, args.new_ai, args.output)
    except SameDayMergeError as exc:
        print(f"same-day merge error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
