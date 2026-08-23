import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from daily_arxiv.daily_arxiv.check_stats import perform_deduplication, resolve_run_date
from daily_arxiv.daily_arxiv.scix_client import (
    SCIX_API_URL,
    ScixClient,
    build_scix_query,
)
from daily_arxiv.daily_arxiv.source_merge import (
    history_keys,
    merge_sources,
    normalize_arxiv_id,
    normalize_doi,
    normalize_scix_document,
    write_jsonl,
)


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self.payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def response_with_docs(docs, num_found=None):
    return FakeResponse(
        200,
        {
            "response": {
                "numFound": len(docs) if num_found is None else num_found,
                "docs": docs,
            }
        },
    )


def arxiv_record(
    paper_id="2608.12345",
    title="Radio pulsar timing",
    summary="We report radio pulsar observations.",
    authors=None,
    **extra,
):
    record = {
        "id": paper_id,
        "title": title,
        "summary": summary,
        "authors": authors or ["A. Author"],
        "categories": ["astro-ph.HE"],
        "abs": f"https://arxiv.org/abs/{paper_id}",
        "pdf": f"https://arxiv.org/pdf/{paper_id}",
        "comment": "",
    }
    record.update(extra)
    return record


def scix_document(
    title="Radio pulsar timing",
    abstract="We report radio pulsar observations.",
    author=None,
    identifier=None,
    **extra,
):
    document = {
        "bibcode": "2026ApJ...001A",
        "title": [title],
        "abstract": [abstract] if abstract is not None else None,
        "author": author or ["A. Author"],
        "doi": "10.1234/example.1",
        "identifier": identifier or [],
        "year": 2026,
        "pub": "The Astrophysical Journal",
        "pubdate": "2026-08-23",
        "entdate": "2026-08-23",
        "database": ["astronomy"],
        "doctype": ["article"],
        "property": ["ARTICLE"],
        "esources": ["PUB_HTML", "EPRINT_PDF"],
    }
    document.update(extra)
    return document


def real_scix_shape_document():
    return {
        "bibcode": "2026arXiv260820135R",
        "title": [
            "Testing Statistical Isotropy in the FRB Sky Distribution: "
            "A Selection-Function-Aware Framework"
        ],
        "abstract": "A representative FRB abstract for offline source normalization testing.",
        "author": ["Ribeiro, Bruno W. N.", "Lemos, Thaiss"],
        "doi": ["10.48550/arXiv.2608.20135"],
        "identifier": [
            "2026arXiv260820135R",
            "arXiv:2608.20135",
            "10.48550/arXiv.2608.20135",
        ],
        "year": "2026",
        "pub": "arXiv e-prints",
        "pubdate": "2026-08-00",
        "entdate": None,
        "database": ["astronomy"],
        "doctype": "eprint",
        "property": [
            "ARTICLE",
            "EPRINT_OPENACCESS",
            "ESOURCE",
            "NOT REFEREED",
            "OPENACCESS",
        ],
        "esources": ["EPRINT_HTML", "EPRINT_PDF"],
    }


class ScixClientTests(unittest.TestCase):
    def test_query_is_broad_topical_candidate_retrieval(self):
        query = build_scix_query("2026-08-20", "2026-08-23")
        self.assertIn("database:astronomy", query)
        self.assertIn("entdate:[2026-08-20 TO 2026-08-23]", query)
        for term in ("pulsar", "magnetar", '"neutron star"', '"fast radio burst"', "FRB", '"radio transient"'):
            self.assertIn(term, query)
        for forbidden in (
            '"single pulse"',
            '"mode switching"',
            '"timing noise"',
            '"spin-down"',
            "polarization",
        ):
            self.assertNotIn(forbidden, query)

    def test_auth_header_and_token_is_not_logged(self):
        token = "secret-token-for-test"
        session = FakeSession([response_with_docs([])])
        client = ScixClient(token=token, session=session)
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            result = client.fetch("2026-08-23", "2026-08-23")
        self.assertEqual("success_empty", result.status)
        self.assertEqual(SCIX_API_URL, session.calls[0][0])
        self.assertEqual(
            f"Bearer {token}", session.calls[0][1]["headers"]["Authorization"]
        )
        self.assertNotIn(token, captured.getvalue())

    def test_request_fields_are_explicit_and_minimal(self):
        session = FakeSession([response_with_docs([])])
        ScixClient(token="token", session=session).fetch("2026-08-23", "2026-08-23")
        requested_fields = set(session.calls[0][1]["params"]["fl"].split(","))
        self.assertIn("bibcode", requested_fields)
        self.assertIn("abstract", requested_fields)
        self.assertIn("esources", requested_fields)
        self.assertNotIn("citation", requested_fields)
        self.assertNotIn("references", requested_fields)
        self.assertNotIn("metrics", requested_fields)

    def test_missing_token_is_unavailable_without_http_call(self):
        session = FakeSession([])
        result = ScixClient(token="", session=session).fetch(
            "2026-08-23", "2026-08-23"
        )
        self.assertEqual("unavailable", result.status)
        self.assertEqual([], session.calls)

    def test_pagination_until_num_found(self):
        session = FakeSession(
            [
                response_with_docs([{"bibcode": "a"}, {"bibcode": "b"}], 3),
                response_with_docs([{"bibcode": "c"}], 3),
            ]
        )
        result = ScixClient(token="token", session=session, rows=2).fetch(
            "2026-08-21", "2026-08-23"
        )
        self.assertEqual("ok", result.status)
        self.assertEqual(3, result.num_found)
        self.assertEqual(2, result.pages)
        self.assertEqual(3, len(result.docs))
        self.assertEqual([0, 2], [call[1]["params"]["start"] for call in session.calls])

    def test_empty_result(self):
        result = ScixClient(
            token="token", session=FakeSession([response_with_docs([], 0)])
        ).fetch("2026-08-23", "2026-08-23")
        self.assertEqual("success_empty", result.status)
        self.assertEqual([], result.docs)

    def test_auth_errors_are_not_retried(self):
        for status in (401, 403):
            session = FakeSession([FakeResponse(status)])
            result = ScixClient(token="token", session=session, retries=3).fetch(
                "2026-08-23", "2026-08-23"
            )
            self.assertEqual("auth_error", result.status)
            self.assertEqual(1, len(session.calls))

    def test_rate_limit_retry_after_is_supported_without_sleeping_in_test(self):
        delays = []
        session = FakeSession(
            [
                FakeResponse(429, headers={"Retry-After": "7"}),
                response_with_docs([]),
            ]
        )
        result = ScixClient(
            token="token",
            session=session,
            retries=1,
            sleep_fn=delays.append,
        ).fetch("2026-08-23", "2026-08-23")
        self.assertEqual("success_empty", result.status)
        self.assertEqual([7.0], delays)
        self.assertEqual(2, len(session.calls))

    def test_timeout_and_5xx_have_bounded_retries(self):
        timeout_session = FakeSession([TimeoutError("timeout"), TimeoutError("timeout")])
        timeout_result = ScixClient(
            token="token", session=timeout_session, retries=1, sleep_fn=lambda _: None
        ).fetch("2026-08-23", "2026-08-23")
        self.assertEqual("unavailable", timeout_result.status)
        self.assertEqual(2, len(timeout_session.calls))

        server_session = FakeSession([FakeResponse(503), FakeResponse(503)])
        server_result = ScixClient(
            token="token", session=server_session, retries=1, sleep_fn=lambda _: None
        ).fetch("2026-08-23", "2026-08-23")
        self.assertEqual("unavailable", server_result.status)
        self.assertEqual(2, len(server_session.calls))

    def test_max_pages_returns_truncated(self):
        session = FakeSession(
            [
                response_with_docs([{"bibcode": "a"}], 3),
                response_with_docs([{"bibcode": "b"}], 3),
            ]
        )
        result = ScixClient(
            token="token", session=session, rows=1, max_pages=2
        ).fetch("2026-08-23", "2026-08-23")
        self.assertEqual("truncated", result.status)
        self.assertTrue(result.truncated)
        self.assertEqual(2, result.pages)


class SourceMergeTests(unittest.TestCase):
    def test_identifier_normalization(self):
        expected = "2608.12345"
        for value in (
            "2608.12345",
            "arXiv:2608.12345",
            "arXiv:2608.12345v2",
            "https://arxiv.org/abs/2608.12345v2",
        ):
            self.assertEqual(expected, normalize_arxiv_id(value))

        expected_doi = "10.1234/example.1"
        for value in (
            "doi:10.1234/Example.1",
            "https://doi.org/10.1234/Example.1.",
            "http://dx.doi.org/10.1234/Example.1",
        ):
            self.assertEqual(expected_doi, normalize_doi(value))

    def test_scix_document_maps_to_compatible_paper(self):
        document = scix_document(
            identifier=["arXiv:2608.12345v2"],
        )
        paper = normalize_scix_document(document)
        self.assertEqual("2608.12345", paper["id"])
        self.assertEqual("Radio pulsar timing", paper["title"])
        self.assertEqual("We report radio pulsar observations.", paper["summary"])
        self.assertIsInstance(paper["authors"], list)
        self.assertEqual(["scix"], paper["categories"])
        self.assertEqual("scix", paper["source"])
        self.assertTrue(paper["abs"].startswith("https://arxiv.org/abs/"))
        self.assertEqual("10.1234/example.1", paper["doi"])
        self.assertEqual("10.1234/example.1", paper["links"]["doi"])

    def test_real_scix_response_shape_normalizes_without_url_inference(self):
        paper = normalize_scix_document(real_scix_shape_document())

        self.assertEqual("2608.20135", paper["id"])
        self.assertEqual(
            "Testing Statistical Isotropy in the FRB Sky Distribution: "
            "A Selection-Function-Aware Framework",
            paper["title"],
        )
        self.assertEqual(
            "A representative FRB abstract for offline source normalization testing.",
            paper["summary"],
        )
        self.assertEqual(
            ["Ribeiro, Bruno W. N.", "Lemos, Thaiss"], paper["authors"]
        )
        self.assertEqual(["scix"], paper["categories"])
        self.assertEqual("scix", paper["source"])
        self.assertEqual("2026arXiv260820135R", paper["bibcode"])
        self.assertEqual("10.48550/arxiv.2608.20135", paper["doi"])
        self.assertEqual("arXiv e-prints", paper["journal"])
        self.assertEqual("2026-08-00", paper["published"])
        self.assertEqual(
            real_scix_shape_document()["identifier"], paper["identifiers"]
        )
        self.assertEqual(real_scix_shape_document()["property"], paper["property"])
        self.assertEqual(["EPRINT_HTML", "EPRINT_PDF"], paper["esources"])
        self.assertEqual("https://arxiv.org/abs/2608.20135", paper["abs"])
        self.assertEqual("https://arxiv.org/pdf/2608.20135", paper["pdf"])
        self.assertNotIn("publisher", paper["links"])
        self.assertTrue(
            any(key.startswith("title-author-year:") for key in history_keys(paper))
        )

    def test_real_scix_shape_merges_once_with_arxiv_priority(self):
        arxiv = arxiv_record(
            paper_id="2608.20135v1",
            title=(
                "Testing Statistical Isotropy in the FRB Sky Distribution: "
                "A Selection-Function-Aware Framework"
            ),
            summary="arXiv abstract",
            authors=["Ribeiro, Bruno W. N."],
            categories=["astro-ph.HE"],
        )
        result = merge_sources([arxiv], [real_scix_shape_document()])

        self.assertEqual(1, len(result.records))
        self.assertEqual(1, len(result.ai_records))
        paper = result.records[0]
        self.assertEqual("2608.20135", paper["id"])
        self.assertEqual("arxiv+scix", paper["source"])
        self.assertEqual(arxiv["title"], paper["title"])
        self.assertEqual("arXiv abstract", paper["summary"])
        self.assertEqual(arxiv["authors"], paper["authors"])
        self.assertEqual(arxiv["categories"], paper["categories"])
        self.assertEqual(arxiv["abs"], paper["abs"])
        self.assertEqual(arxiv["pdf"], paper["pdf"])
        self.assertEqual("2026arXiv260820135R", paper["bibcode"])
        self.assertEqual("10.48550/arxiv.2608.20135", paper["doi"])
        self.assertEqual("arXiv e-prints", paper["journal"])
        self.assertEqual("2026-08-00", paper["published"])
        self.assertEqual(real_scix_shape_document()["property"], paper["property"])
        self.assertEqual(["EPRINT_HTML", "EPRINT_PDF"], paper["esources"])
        self.assertIn("arxiv:2608.20135", history_keys(paper))
        self.assertIn("doi:10.48550/arxiv.2608.20135", history_keys(paper))

    def test_esources_are_availability_metadata_not_urls(self):
        paper = normalize_scix_document(
            scix_document(
                identifier=[],
                esources=["PUB_HTML", "EPRINT_PDF"],
            )
        )
        self.assertEqual(["PUB_HTML", "EPRINT_PDF"], paper["esources"])
        self.assertEqual("", paper["abs"])
        self.assertEqual("", paper["pdf"])
        self.assertEqual("", paper["links"]["arxiv_abs"])
        self.assertEqual("", paper["links"]["arxiv_pdf"])
        self.assertEqual("10.1234/example.1", paper["doi"])
        self.assertEqual("10.1234/example.1", paper["links"]["doi"])
        self.assertNotIn("publisher", paper["links"])

    def test_arxiv_identifier_is_the_only_scix_abs_pdf_generation_source(self):
        paper = normalize_scix_document(
            scix_document(
                identifier=["arXiv:2608.12345v2"],
                esources=["PUB_HTML", "EPRINT_PDF"],
            )
        )
        self.assertEqual("https://arxiv.org/abs/2608.12345", paper["abs"])
        self.assertEqual("https://arxiv.org/pdf/2608.12345", paper["pdf"])
        self.assertEqual("https://arxiv.org/abs/2608.12345", paper["links"]["arxiv_abs"])
        self.assertEqual("https://arxiv.org/pdf/2608.12345", paper["links"]["arxiv_pdf"])

    def test_doi_is_metadata_not_a_guessed_publisher_or_pdf_url(self):
        paper = normalize_scix_document(
            scix_document(
                identifier=[],
                doi="https://doi.org/10.1234/Example.1",
                esources=["PUB_HTML", "EPRINT_PDF"],
            )
        )
        self.assertEqual("10.1234/example.1", paper["doi"])
        self.assertEqual("10.1234/example.1", paper["links"]["doi"])
        self.assertEqual("", paper["abs"])
        self.assertEqual("", paper["pdf"])

    def test_missing_abstract_is_retained_in_merge_result_but_not_ai_records(self):
        result = merge_sources([], [scix_document(abstract=None)])
        self.assertEqual(1, result.scix_missing_abstract_count)
        self.assertEqual(1, result.canonical_missing_abstract_count)
        self.assertEqual(1, len(result.records))
        self.assertEqual([], result.ai_records)

    def test_same_arxiv_id_merges_and_arxiv_content_has_priority(self):
        arxiv = arxiv_record(
            title="ArXiv title",
            summary="The arXiv abstract is the AI input.",
            authors=["A. Author"],
        )
        scix = scix_document(
            title="Journal title",
            abstract="A different journal abstract.",
            identifier=["arXiv:2608.12345v2"],
        )
        result = merge_sources([arxiv], [scix])
        self.assertEqual(1, len(result.ai_records))
        paper = result.ai_records[0]
        self.assertEqual("2608.12345", paper["id"])
        self.assertEqual("ArXiv title", paper["title"])
        self.assertEqual("The arXiv abstract is the AI input.", paper["summary"])
        self.assertEqual("arxiv+scix", paper["source"])
        self.assertTrue(paper["doi"])
        self.assertTrue(paper["bibcode"])

    def test_same_doi_merges(self):
        arxiv = arxiv_record(doi="10.1234/example.1")
        scix = scix_document(identifier=[])
        result = merge_sources([arxiv], [scix])
        self.assertEqual(1, len(result.records))
        self.assertEqual("arxiv+scix", result.records[0]["source"])

    def test_bibcode_duplicate_merges(self):
        first = scix_document(title="First title")
        second = scix_document(title="Second title", abstract="Another abstract")
        result = merge_sources([], [first, second])
        self.assertEqual(1, len(result.records))
        self.assertEqual(1, result.merged_count)

    def test_title_author_year_guard_merges_but_title_only_does_not(self):
        legacy = arxiv_record(
            paper_id="legacy-record-a",
            title="A title shared by two records",
            authors=["A. Author"],
            published="2026-08-23",
        )
        matching = scix_document(
            title="A title shared by two records",
            author=["A. Author"],
            identifier=[],
            doi="",
            bibcode="",
            year=2026,
        )
        matching.pop("esources", None)
        merged = merge_sources([legacy], [matching])
        self.assertEqual(1, len(merged.records))

        different_author = scix_document(
            title="A title shared by two records",
            author=["Different Author"],
            identifier=[],
            doi="",
            bibcode="",
            year=2026,
        )
        different_author.pop("esources", None)
        not_merged = merge_sources([legacy], [different_author])
        self.assertEqual(2, len(not_merged.records))

    def test_history_keys_include_canonical_fields(self):
        paper = arxiv_record(
            doi="10.1234/Example.1",
            bibcode="2026ApJ...001A",
            published="2026-08-23",
        )
        keys = history_keys(paper)
        self.assertIn("arxiv:2608.12345", keys)
        self.assertIn("doi:10.1234/example.1", keys)
        self.assertIn("bibcode:2026ApJ...001A", keys)
        self.assertTrue(any(key.startswith("title-author-year:") for key in keys))

    def test_scix_failure_preserves_arxiv_and_overlapping_records_merge_once(self):
        arxiv = arxiv_record()
        preserved = merge_sources([arxiv], [])
        self.assertEqual(1, len(preserved.ai_records))
        self.assertEqual("arxiv", preserved.ai_records[0]["source"])

        duplicate_scix = scix_document(identifier=["arXiv:2608.12345v2"])
        overlapping = merge_sources([], [duplicate_scix, duplicate_scix])
        self.assertEqual(1, len(overlapping.ai_records))

    def test_history_dedup_uses_explicit_date_and_does_not_touch_history_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "work" / "data"
            history_dir = root / "history" / "data"
            data_dir.mkdir(parents=True)
            history_dir.mkdir(parents=True)

            today_paper = arxiv_record(
                paper_id="2608.99999",
                doi="10.1234/history.1",
            )
            history_paper = arxiv_record(
                paper_id="legacy-history-id",
                doi="10.1234/HISTORY.1",
            )
            write_jsonl([today_paper], data_dir / "2026-08-23.jsonl")
            write_jsonl([history_paper], history_dir / "2026-08-22.jsonl")

            status = perform_deduplication(
                run_date="2026-08-23",
                data_dir=data_dir,
                history_dir=history_dir,
            )
            self.assertEqual("no_new_content", status)
            self.assertFalse((data_dir / "2026-08-23.jsonl").exists())
            self.assertTrue((history_dir / "2026-08-22.jsonl").exists())
            self.assertEqual("2026-08-23", resolve_run_date("2026-08-23").isoformat())

    def test_filter_and_ai_boundary_can_use_one_fake_call_per_canonical_paper(self):
        result = merge_sources(
            [arxiv_record(title="Pulsar paper", summary="pulsar abstract")],
            [scix_document(identifier=["arXiv:2608.12345v2"])],
        )
        calls = []

        def fake_ai_call(paper):
            calls.append(paper["id"])

        for paper in result.ai_records:
            fake_ai_call(paper)
        self.assertEqual(["2608.12345"], calls)


if __name__ == "__main__":
    unittest.main()
