#!/usr/bin/env python3
"""Normalize and canonically merge arXiv and SciX Paper records.

This module is intentionally independent from the AI and web layers.  It is
also the single home for identifier and history-key normalization so that
cross-source and cross-day deduplication cannot silently drift apart.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_ARXIV_ID_RE = re.compile(
    r"(?i)(?<![a-z0-9])((?:\d{4}\.\d{4,5}|[a-z][a-z-]+/\d{7})(?:v\d+)?)(?![a-z0-9])"
)
_DOI_RE = re.compile(r"(?i)^10\.\d{4,9}/\S+$")


def _first_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            text = _first_text(item)
            if text:
                return text
    if value is None:
        return ""
    return str(value).strip()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    output: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            item = item.get("name") or item.get("full_name") or item.get("family")
        text = _first_text(item)
        if text and text not in output:
            output.append(text)
    return output


def _identifier_list(value: Any) -> list[str]:
    return _string_list(value)


def normalize_arxiv_id(value: Any) -> str | None:
    """Return an arXiv identifier without URL, prefix, or version suffix."""

    candidates = value if isinstance(value, list) else [value]
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        text = candidate.strip()
        if not text:
            continue
        match = _ARXIV_ID_RE.search(text)
        if match:
            return match.group(1).casefold().removesuffix(
                re.search(r"v\d+$", match.group(1), re.IGNORECASE).group(0).casefold()
                if re.search(r"v\d+$", match.group(1), re.IGNORECASE)
                else ""
            )
    return None


def normalize_doi(value: Any) -> str | None:
    """Normalize common DOI forms to a lowercase DOI string."""

    candidates = value if isinstance(value, list) else [value]
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        text = candidate.strip().strip("<>").casefold()
        for prefix in (
            "https://doi.org/",
            "http://doi.org/",
            "https://dx.doi.org/",
            "http://dx.doi.org/",
            "doi:",
        ):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
        text = text.strip().rstrip(".,;:)]}>")
        if _DOI_RE.match(text):
            return text
    return None


def normalize_bibcode(value: Any) -> str | None:
    text = _first_text(value)
    return text or None


def normalize_title(value: Any) -> str:
    """Create a conservative title key without deleting all math content."""

    text = unicodedata.normalize("NFKC", _first_text(value)).casefold()
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def normalize_author_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _first_text(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _record_year(record: Mapping[str, Any]) -> str:
    for key in ("year", "published", "pubdate"):
        match = re.search(r"\b(19|20)\d{2}\b", _first_text(record.get(key)))
        if match:
            return match.group(0)
    return ""


def _extract_arxiv_id(record: Mapping[str, Any]) -> str | None:
    for key in ("id", "identifier", "identifiers", "abs", "pdf"):
        value = record.get(key)
        normalized = normalize_arxiv_id(value)
        if normalized:
            return normalized
    return None


def _links_for_scix(doi: str | None, arxiv_id: str | None) -> dict[str, str]:
    """Build only links whose provenance is explicit and deterministic.

    SciX/ADS ``esources`` values are availability-type metadata such as
    ``PUB_HTML`` and ``EPRINT_PDF``.  They are not URLs and must not be used
    to invent publisher, PDF, abstract, or HTML addresses.
    """

    links = {
        "arxiv_abs": "",
        "arxiv_pdf": "",
        # DOI is metadata here, not a guessed DOI/publisher URL.
        "doi": doi or "",
    }
    if arxiv_id:
        links["arxiv_abs"] = f"https://arxiv.org/abs/{arxiv_id}"
        links["arxiv_pdf"] = f"https://arxiv.org/pdf/{arxiv_id}"
    return links


def _ensure_link_dict(record: Mapping[str, Any]) -> dict[str, str]:
    raw_links = record.get("links")
    links = {
        "arxiv_abs": "",
        "arxiv_pdf": "",
        "doi": "",
    }
    if isinstance(raw_links, Mapping):
        for key in links:
            value = raw_links.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            if key == "doi":
                links[key] = normalize_doi(value) or ""
            elif key == "arxiv_abs" and "arxiv.org/abs/" in value.casefold():
                links[key] = value.strip()
            elif key == "arxiv_pdf" and "arxiv.org/pdf/" in value.casefold():
                links[key] = value.strip()
    # Accept legacy arXiv-only names only when the value is visibly an arXiv
    # URL.  Never carry over a URL that may have been guessed from esources.
    for old_key, new_key, marker in (
        ("abstract", "arxiv_abs", "arxiv.org/abs/"),
        ("pdf", "arxiv_pdf", "arxiv.org/pdf/"),
    ):
        value = raw_links.get(old_key) if isinstance(raw_links, Mapping) else None
        if not links[new_key] and isinstance(value, str) and marker in value.casefold():
            links[new_key] = value.strip()
    if not links["arxiv_abs"] and "arxiv.org/abs/" in _first_text(record.get("abs")).casefold():
        links["arxiv_abs"] = _first_text(record.get("abs"))
    if not links["arxiv_pdf"] and "arxiv.org/pdf/" in _first_text(record.get("pdf")).casefold():
        links["arxiv_pdf"] = _first_text(record.get("pdf"))
    doi = normalize_doi(record.get("doi"))
    if doi and not links["doi"]:
        links["doi"] = doi
    return links


def normalize_arxiv_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize an existing arXiv Paper without changing its abstract."""

    normalized = copy.deepcopy(dict(record))
    arxiv_id = _extract_arxiv_id(normalized)
    title = _first_text(normalized.get("title"))
    summary = _first_text(normalized.get("summary"))
    authors = _string_list(normalized.get("authors"))
    categories = _string_list(normalized.get("categories")) or ["arxiv"]
    links = _ensure_link_dict(normalized)

    if arxiv_id:
        normalized["id"] = arxiv_id
        links["arxiv_abs"] = links["arxiv_abs"] or f"https://arxiv.org/abs/{arxiv_id}"
        links["arxiv_pdf"] = links["arxiv_pdf"] or f"https://arxiv.org/pdf/{arxiv_id}"

    normalized.update(
        {
            "id": _first_text(normalized.get("id")) or arxiv_id or "",
            "title": title,
            "summary": summary,
            "authors": authors,
            "categories": categories,
            "abs": links["arxiv_abs"],
            "pdf": links["arxiv_pdf"],
            "comment": _first_text(normalized.get("comment")),
            "source": "arxiv",
            "links": links,
        }
    )
    normalized_doi = normalize_doi(normalized.get("doi"))
    if normalized_doi:
        normalized["doi"] = normalized_doi
    else:
        normalized.pop("doi", None)
    return normalized


def normalize_scix_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Map one SciX API document to the compatible Paper shape."""

    title = _first_text(document.get("title"))
    summary = _first_text(document.get("abstract"))
    authors = _string_list(document.get("author"))
    identifiers = _identifier_list(document.get("identifier"))
    arxiv_id = normalize_arxiv_id(identifiers)
    doi = normalize_doi(document.get("doi")) or normalize_doi(identifiers)
    bibcode = normalize_bibcode(document.get("bibcode"))
    links = _links_for_scix(doi, arxiv_id)

    if arxiv_id:
        paper_id = arxiv_id
    elif bibcode:
        paper_id = f"scix:{bibcode}"
    elif doi:
        paper_id = f"doi:{doi}"
    else:
        fingerprint = "|".join(
            (normalize_title(title), normalize_author_key(authors[0] if authors else ""), _record_year(document))
        )
        paper_id = "scix:title:" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]

    paper: dict[str, Any] = {
        "id": paper_id,
        "title": title,
        "summary": summary,
        "authors": authors,
        # Compatibility/source bucket, not a scientific classification.
        "categories": ["scix"],
        "abs": links["arxiv_abs"],
        "pdf": links["arxiv_pdf"],
        "comment": "",
        "source": "scix",
        "bibcode": bibcode or "",
        "doi": doi or "",
        "journal": _first_text(document.get("pub")),
        "published": _first_text(document.get("pubdate")),
        "identifiers": identifiers,
        "esources": _string_list(document.get("esources")),
        "links": links,
    }
    return paper


def history_keys(record: Mapping[str, Any]) -> set[str]:
    """Return all stable keys used by both cross-source and cross-day dedup."""

    keys: set[str] = set()
    arxiv_id = _extract_arxiv_id(record)
    if arxiv_id:
        keys.add(f"arxiv:{arxiv_id}")

    doi = normalize_doi(record.get("doi")) or normalize_doi(record.get("identifiers"))
    if doi:
        keys.add(f"doi:{doi}")

    bibcode = normalize_bibcode(record.get("bibcode"))
    if bibcode:
        keys.add(f"bibcode:{bibcode}")

    title_key = normalize_title(record.get("title"))
    authors = _string_list(record.get("authors"))
    author_key = normalize_author_key(authors[0] if authors else "")
    year = _record_year(record)
    if title_key and author_key and year:
        keys.add(f"title-author-year:{title_key}|{author_key}|{year}")

    raw_id = _first_text(record.get("id"))
    if raw_id:
        keys.add(f"id:{raw_id.casefold()}")
    return keys


def _is_arxiv_record(record: Mapping[str, Any]) -> bool:
    source = _first_text(record.get("source"))
    if source:
        return source in {"arxiv", "arxiv+scix"}
    return bool(normalize_arxiv_id(record.get("id")))


def _merge_links(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, str]:
    merged = _ensure_link_dict(left)
    right_links = _ensure_link_dict(right)
    for key in merged:
        if not merged[key] and right_links[key]:
            merged[key] = right_links[key]
    return merged


def _merge_two(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_is_arxiv = _is_arxiv_record(left)
    right_is_arxiv = _is_arxiv_record(right)

    if right_is_arxiv and not left_is_arxiv:
        result = copy.deepcopy(dict(right))
        metadata_source = left
    else:
        result = copy.deepcopy(dict(left))
        metadata_source = right

    content_fields = ("title", "summary", "authors", "categories", "abs", "pdf", "comment")
    for field_name in content_fields:
        if not result.get(field_name) and metadata_source.get(field_name):
            result[field_name] = copy.deepcopy(metadata_source[field_name])

    for field_name in ("bibcode", "doi", "journal", "published"):
        if not _first_text(result.get(field_name)) and _first_text(metadata_source.get(field_name)):
            result[field_name] = _first_text(metadata_source.get(field_name))

    identifiers = _string_list(result.get("identifiers"))
    for identifier in _string_list(metadata_source.get("identifiers")):
        if identifier not in identifiers:
            identifiers.append(identifier)
    result["identifiers"] = identifiers
    esources = _string_list(result.get("esources"))
    for esource in _string_list(metadata_source.get("esources")):
        if esource not in esources:
            esources.append(esource)
    result["esources"] = esources
    result["links"] = _merge_links(result, metadata_source)

    if left_is_arxiv or right_is_arxiv:
        result["source"] = "arxiv+scix" if not (left_is_arxiv and right_is_arxiv) else "arxiv"
        arxiv_id = normalize_arxiv_id(left.get("id")) or normalize_arxiv_id(right.get("id"))
        if arxiv_id:
            result["id"] = arxiv_id
    else:
        result["source"] = _first_text(result.get("source")) or "scix"

    links = _ensure_link_dict(result)
    result["abs"] = links["arxiv_abs"]
    result["pdf"] = links["arxiv_pdf"]
    result["categories"] = _string_list(result.get("categories")) or ["scix"]
    result["authors"] = _string_list(result.get("authors"))
    result["title"] = _first_text(result.get("title"))
    result["summary"] = _first_text(result.get("summary"))
    result["comment"] = _first_text(result.get("comment"))
    return result


@dataclass
class MergeResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    ai_records: list[dict[str, Any]] = field(default_factory=list)
    scix_missing_abstract_count: int = 0
    canonical_missing_abstract_count: int = 0
    merged_count: int = 0


def merge_sources(
    arxiv_records: Iterable[Mapping[str, Any]],
    scix_documents: Iterable[Mapping[str, Any]],
) -> MergeResult:
    """Normalize, merge and return both all canonicals and AI-eligible ones."""

    normalized: list[dict[str, Any]] = []
    for record in arxiv_records:
        normalized.append(normalize_arxiv_record(record))

    scix_missing = 0
    for document in scix_documents:
        if not _first_text(document.get("abstract")):
            scix_missing += 1
        normalized.append(normalize_scix_document(document))

    canonical: list[dict[str, Any]] = []
    merged_count = 0
    for record in normalized:
        record_keys = history_keys(record)
        matches = [
            index
            for index, existing in enumerate(canonical)
            if record_keys.intersection(history_keys(existing))
        ]
        if not matches:
            canonical.append(record)
            continue

        target = matches[0]
        canonical[target] = _merge_two(canonical[target], record)
        merged_count += 1
        for extra_index in reversed(matches[1:]):
            canonical[target] = _merge_two(canonical[target], canonical[extra_index])
            del canonical[extra_index]

    canonical_missing = sum(
        1 for record in canonical if not _first_text(record.get("summary"))
    )
    ai_records = [
        record
        for record in canonical
        if _first_text(record.get("title")) and _first_text(record.get("summary"))
    ]
    return MergeResult(
        records=canonical,
        ai_records=ai_records,
        scix_missing_abstract_count=scix_missing,
        canonical_missing_abstract_count=canonical_missing,
        merged_count=merged_count,
    )


def load_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).exists():
        return []
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
    return records


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def write_merge_stats(result: MergeResult, path: str | Path | None) -> None:
    if path is None:
        return
    Path(path).write_text(
        json.dumps(
            {
                "canonical_count": len(result.records),
                "ai_eligible_count": len(result.ai_records),
                "merged_count": result.merged_count,
                "scix_missing_abstract_count": result.scix_missing_abstract_count,
                "canonical_missing_abstract_count": result.canonical_missing_abstract_count,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge arXiv and SciX JSONL")
    parser.add_argument("--arxiv", required=True, help="arXiv raw JSONL path")
    parser.add_argument("--scix", help="SciX raw JSONL path")
    parser.add_argument("--output", required=True, help="canonical working JSONL path")
    parser.add_argument("--stats-file")
    args = parser.parse_args(argv)

    result = merge_sources(load_jsonl(args.arxiv), load_jsonl(args.scix))
    # The working file is the AI-eligible canonical set.  Raw files retain
    # metadata-only records, including records without an abstract.
    write_jsonl(result.ai_records, args.output)
    write_merge_stats(result, args.stats_file)
    print(
        f"source merge canonical={len(result.records)} "
        f"ai_eligible={len(result.ai_records)} "
        f"merged={result.merged_count} "
        f"scix_missing_abstract_count={result.scix_missing_abstract_count}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
