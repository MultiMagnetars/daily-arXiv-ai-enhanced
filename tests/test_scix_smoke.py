import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from daily_arxiv.daily_arxiv.scix_client import ScixFetchResult
from daily_arxiv.daily_arxiv.scix_smoke import (
    SMOKE_MAX_PAGES,
    SMOKE_RETRIES,
    SMOKE_ROWS,
    classify_client_result,
    main as smoke_main,
    run_smoke,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "run.yml"
SMOKE_PATH = ROOT_DIR / "daily_arxiv" / "daily_arxiv" / "scix_smoke.py"


class FakeSmokeClient:
    def __init__(self, result, **kwargs):
        self.result = result
        self.kwargs = kwargs
        self.calls = []

    def fetch(self, start_date, end_date):
        self.calls.append((start_date, end_date))
        return self.result


def factory_for(result, holder):
    def factory(**kwargs):
        client = FakeSmokeClient(result, **kwargs)
        holder["client"] = client
        return client

    return factory


class ScixSmokeTests(unittest.TestCase):
    def test_existing_client_is_called_with_rows_one_and_max_pages_one(self):
        holder = {}
        result = ScixFetchResult(status="success_empty", pages=1)
        smoke = run_smoke(
            client_factory=factory_for(result, holder),
            run_date=date(2026, 8, 23),
            token="secret-token",
        )

        self.assertEqual("PASS_EMPTY", smoke.status)
        self.assertEqual(
            {
                "token": "secret-token",
                "rows": SMOKE_ROWS,
                "max_pages": SMOKE_MAX_PAGES,
                "retries": SMOKE_RETRIES,
            },
            holder["client"].kwargs,
        )
        self.assertEqual([("2026-08-20", "2026-08-23")], holder["client"].calls)

    def test_truncated_bounded_sample_is_limited_connection_pass_and_exit_zero(self):
        result = ScixFetchResult(
            docs=[{"bibcode": "2026arXiv260820135R"}],
            status="truncated",
            num_found=10,
            pages=1,
            truncated=True,
        )
        smoke = run_smoke(
            client_factory=factory_for(result, {}),
            run_date=date(2026, 8, 23),
            token="secret-token",
        )

        self.assertEqual("PASS_API_CONNECTION_LIMITED", smoke.status)
        self.assertIn("client_status: truncated", smoke.report)
        self.assertIn("limited_sample: true", smoke.report)
        self.assertIn("numFound: 10", smoke.report)
        self.assertIn("docs_received: 1", smoke.report)
        with patch(
            "daily_arxiv.daily_arxiv.scix_smoke.run_smoke", return_value=smoke
        ), patch("sys.stdout"):
            self.assertEqual(0, smoke_main())

    def test_truncated_without_a_document_remains_response_shape_failure(self):
        result = ScixFetchResult(
            docs=[],
            status="truncated",
            num_found=10,
            pages=1,
            truncated=True,
        )
        smoke = run_smoke(
            client_factory=factory_for(result, {}),
            run_date=date(2026, 8, 23),
            token="token",
        )
        self.assertEqual("FAIL_RESPONSE_SHAPE", smoke.status)
        self.assertNotIn("limited_sample: true", smoke.report)

    def test_truncated_with_more_than_one_document_remains_response_shape_failure(self):
        result = ScixFetchResult(
            docs=[{"bibcode": "one"}, {"bibcode": "two"}],
            status="truncated",
            num_found=10,
            pages=1,
            truncated=True,
        )
        smoke = run_smoke(
            client_factory=factory_for(result, {}),
            run_date=date(2026, 8, 23),
            token="token",
        )
        self.assertEqual("FAIL_RESPONSE_SHAPE", smoke.status)
        self.assertNotIn("limited_sample: true", smoke.report)

    def test_sanitized_output_has_types_and_bounded_abstract(self):
        secret = "secret-token"
        doc = {
            "bibcode": "2026arXiv260820135R",
            "title": [
                "Testing Statistical Isotropy in the FRB Sky Distribution: "
                "A Selection-Function-Aware Framework"
            ],
            "abstract": "abstract " + ("x" * 500) + " " + secret,
            "author": ["A. Author"],
            "doi": ["10.48550/arXiv.2608.20135"],
            "identifier": [
                "2026arXiv260820135R",
                "arXiv:2608.20135",
                "10.48550/arXiv.2608.20135",
            ],
            "year": "2026",
            "pub": "arXiv",
            "pubdate": "2026-08-23",
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
        holder = {}
        result = ScixFetchResult(docs=[doc], status="ok", num_found=1, pages=1)
        smoke = run_smoke(
            client_factory=factory_for(result, holder),
            run_date=date(2026, 8, 23),
            token=secret,
        )

        self.assertEqual("PASS_API_CONNECTION", smoke.status)
        report = smoke.report
        self.assertNotIn(secret, report)
        self.assertNotIn("Authorization", report)
        self.assertIn("title_type: list", report)
        self.assertIn("doi_type: list", report)
        self.assertIn("identifier_type: list", report)
        self.assertIn("esources_type: list", report)
        self.assertIn("abstract_type: str", report)
        self.assertIn("year_type: str", report)
        self.assertIn("pub_type: str", report)
        self.assertIn("pubdate_type: str", report)
        self.assertIn("entdate_type: NoneType", report)
        self.assertIn("doctype_type: str", report)
        self.assertIn("abstract_present: true", report)
        self.assertIn("abstract_length:", report)
        self.assertIn("abstract_preview:", report)
        self.assertIn('normalized_doi: "10.48550/arxiv.2608.20135"', report)
        self.assertIn('normalized_arxiv_id: "2608.20135"', report)
        self.assertIn("https://arxiv.org/abs/2608.20135", report)
        self.assertIn("https://arxiv.org/pdf/2608.20135", report)

    def test_success_empty_is_a_pass_without_record_output(self):
        result = ScixFetchResult(status="success_empty", num_found=0, pages=1)
        smoke = run_smoke(
            client_factory=factory_for(result, {}),
            run_date=date(2026, 8, 23),
            token="token",
        )
        self.assertEqual("PASS_EMPTY", smoke.status)
        self.assertIn("status: PASS_EMPTY", smoke.report)
        self.assertIn("docs_received: 0", smoke.report)
        self.assertNotIn("abstract_present:", smoke.report)

    def test_failure_statuses_are_explicit_and_network_is_not_called_by_harness(self):
        cases = (
            (ScixFetchResult(status="auth_error", error="HTTP 401"), "FAIL_AUTH"),
            (ScixFetchResult(status="rate_limited", error="HTTP 429"), "FAIL_RATE_LIMIT"),
            (ScixFetchResult(status="unavailable", error="HTTP transport error: TimeoutError"), "FAIL_NETWORK"),
            (ScixFetchResult(status="unavailable", error="Invalid SciX response: JSONDecodeError"), "FAIL_RESPONSE_SHAPE"),
        )
        for result, expected in cases:
            with self.subTest(expected=expected):
                smoke = run_smoke(
                    client_factory=factory_for(result, {}),
                    run_date=date(2026, 8, 23),
                    token="token",
                )
                self.assertEqual(expected, smoke.status)
                self.assertIn(f"status: {expected}", smoke.report)

    def test_esources_are_rendered_raw_and_not_converted_to_urls(self):
        doc = {
            "title": ["Radio transient"],
            "abstract": ["A short abstract."],
            "identifier": [],
            "doi": [],
            "esources": ["PUB_HTML", "EPRINT_PDF"],
        }
        smoke = run_smoke(
            client_factory=factory_for(
                ScixFetchResult(docs=[doc], status="ok", num_found=1, pages=1), {}
            ),
            run_date=date(2026, 8, 23),
            token="token",
        )
        self.assertIn('esources: ["PUB_HTML", "EPRINT_PDF"]', smoke.report)
        self.assertNotIn("publisher.example", smoke.report)
        self.assertNotIn("doi.org/", smoke.report)
        self.assertNotIn("arxiv.org/abs/", smoke.report)
        self.assertNotIn("arxiv.org/pdf/", smoke.report)

    def test_smoke_has_no_production_or_remote_side_effect_entry_points(self):
        source = SMOKE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("merge_sources", source)
        self.assertNotIn("write_jsonl", source)
        self.assertNotIn("enhance.py", source)
        self.assertNotIn("langchain", source)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("git push", source)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual([], list(Path(temp_dir).iterdir()))

    def test_workflow_has_manual_astro_custom_guard_and_preserves_build_paths(self):
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn("scix-smoke:", workflow)
        self.assertIn(
            "if: ${{ github.event_name == 'workflow_dispatch' && github.ref_name == 'astro-custom' && inputs.validation_mode == 'scix-smoke' }}",
            workflow,
        )
        self.assertIn(
            "if: ${{ github.event_name != 'workflow_dispatch' || github.ref_name != 'astro-custom' || inputs.validation_mode == 'normal' }}",
            workflow,
        )
        self.assertIn(
            "if: ${{ github.event_name == 'workflow_dispatch' && github.ref_name == 'astro-custom' && inputs.validation_mode == 'scix-e2e' }}",
            workflow,
        )
        self.assertIn('cron: "30 17 * * *"', workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("validation_mode:", workflow)
        self.assertIn("- normal", workflow)
        self.assertIn("- scix-smoke", workflow)
        self.assertIn("- scix-e2e", workflow)
        self.assertIn("SCIX_API_TOKEN: ${{ secrets.SCIX_API_TOKEN }}", workflow)
        smoke_block = workflow.split("  scix-smoke:", 1)[1].split("\n  scix-e2e:", 1)[0]
        self.assertIn("python -m daily_arxiv.daily_arxiv.scix_smoke", smoke_block)
        self.assertNotIn("source_merge.py", smoke_block)
        self.assertNotIn("check_stats.py", smoke_block)
        self.assertNotIn("enhance.py", smoke_block)
        self.assertNotIn("git push", smoke_block)
        e2e_block = workflow.split("  scix-e2e:", 1)[1].split("\n  build:", 1)[0]
        self.assertIn("python -m daily_arxiv.daily_arxiv.scix_e2e_validate", e2e_block)
        self.assertIn('output-dir "${RUNNER_TEMP}/scix_e2e_output"', e2e_block)
        self.assertNotIn("git push", e2e_block)

    def test_status_classifier_never_exposes_client_auth_header(self):
        result = ScixFetchResult(status="ok", docs=[])
        self.assertEqual("PASS_API_CONNECTION", classify_client_result(result))
        self.assertNotIn("Authorization", str(result))


if __name__ == "__main__":
    unittest.main()
