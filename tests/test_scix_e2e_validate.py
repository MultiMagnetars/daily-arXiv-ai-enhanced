import ast
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch
from typing import Dict, List, Optional, Tuple

from daily_arxiv.daily_arxiv.scix_client import ScixFetchResult
from daily_arxiv.daily_arxiv.scix_e2e_validate import (
    MAX_AI_ITEMS,
    SCIX_MAX_PAGES,
    SCIX_ROWS,
    create_scix_client,
    run_validation,
)
from daily_arxiv.daily_arxiv.source_merge import write_jsonl


ROOT_DIR = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT_DIR / "daily_arxiv" / "daily_arxiv" / "scix_e2e_validate.py"
WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "run.yml"

# Load the actual production filter helper definitions without importing the
# LLM runtime or making any network request during offline tests.
ENHANCE_SOURCE = ROOT_DIR / "ai" / "enhance.py"
ENHANCE_TREE = ast.parse(ENHANCE_SOURCE.read_text(encoding="utf-8"))
FILTER_NODES = [
    node
    for node in ENHANCE_TREE.body
    if isinstance(node, ast.FunctionDef)
    and node.name in {"parse_filter_keywords", "filter_papers_by_keywords"}
]
FILTER_NAMESPACE = {
    "Dict": Dict,
    "List": List,
    "Optional": Optional,
    "Tuple": Tuple,
}
exec(
    compile(ast.Module(body=FILTER_NODES, type_ignores=[]), str(ENHANCE_SOURCE), "exec"),
    FILTER_NAMESPACE,
)
parse_filter_keywords = FILTER_NAMESPACE["parse_filter_keywords"]
filter_papers_by_keywords = FILTER_NAMESPACE["filter_papers_by_keywords"]


class FakeScixClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def fetch(self, start_date, end_date):
        self.calls.append((start_date, end_date))
        return self.result


def scix_doc(index=1, title=None, summary=None):
    arxiv_id = f"2608.{index:05d}"
    return {
        "bibcode": f"2026arXiv{index:05d}V",
        "title": [title or f"Pulsar validation paper {index}"],
        "abstract": summary or "A representative pulsar abstract.",
        "author": ["A. Author"],
        "doi": [f"10.48550/arXiv.{arxiv_id}"],
        "identifier": [f"arXiv:{arxiv_id}"],
        "year": "2026",
        "pub": "arXiv e-prints",
        "pubdate": "2026-08-00",
        "entdate": None,
        "database": ["astronomy"],
        "doctype": "eprint",
        "property": ["ARTICLE"],
        "esources": ["EPRINT_HTML", "EPRINT_PDF"],
    }


def arxiv_record(arxiv_id="2608.00001", summary="arXiv abstract"):
    return {
        "id": f"{arxiv_id}v1",
        "title": "Pulsar validation paper",
        "summary": summary,
        "authors": ["A. Author"],
        "categories": ["astro-ph.HE"],
        "abs": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
    }


def ai_result(tldr="A concise validation result.", secret=""):
    return {
        "abstract_translation": f"中文翻译 {secret}",
        "tldr": f"{tldr} {secret}".strip(),
        "motivation": "研究动机",
        "method": "研究方法",
        "result": "研究结果",
        "conclusion": "研究结论",
    }


def ai_factory(calls, result=None, error=None, secret=""):
    def factory():
        def run(item):
            calls.append(item["id"])
            if error is not None:
                raise error
            return ai_result(secret=secret) if result is None else result

        return run

    return factory


def validate(
    *,
    docs,
    keywords="pulsar",
    arxiv_records=None,
    history_dir=None,
    ai_runner_factory=None,
    output_dir=None,
    status="ok",
    num_found=None,
    secret="",
):
    result = ScixFetchResult(
        docs=docs,
        status=status,
        num_found=len(docs) if num_found is None else num_found,
        pages=1,
        truncated=status == "truncated",
    )
    fake_client = FakeScixClient(result)
    return run_validation(
        scix_client=fake_client,
        arxiv_records=arxiv_records or [],
        arxiv_raw_source_date="2026-08-23" if arxiv_records else None,
        history_dir=history_dir,
        filter_keywords_raw=keywords,
        filter_parser=parse_filter_keywords,
        filter_function=filter_papers_by_keywords,
        ai_runner_factory=ai_runner_factory,
        run_date=date(2026, 8, 23),
        output_dir=output_dir,
        secret=secret,
    )


class ScixE2EValidationTests(unittest.TestCase):
    def test_validation_client_bounds_are_rows_at_most_ten_and_one_page(self):
        with patch("daily_arxiv.daily_arxiv.scix_e2e_validate.ScixClient") as client_class:
            create_scix_client()
        kwargs = client_class.call_args.kwargs
        self.assertLessEqual(kwargs["rows"], 10)
        self.assertEqual(SCIX_ROWS, kwargs["rows"])
        self.assertEqual(SCIX_MAX_PAGES, kwargs["max_pages"])
        self.assertEqual(10, SCIX_ROWS)
        self.assertEqual(1, SCIX_MAX_PAGES)

    def test_eligible_ten_is_hard_capped_at_two_and_limited_fetch_passes(self):
        calls = []
        result = validate(
            docs=[scix_doc(index=i) for i in range(1, 11)],
            keywords="pulsar",
            ai_runner_factory=ai_factory(calls),
            status="truncated",
            num_found=20,
        )
        self.assertEqual("PASS_E2E_LIMITED", result.status)
        self.assertEqual(10, result.report["eligible_before_cap"])
        self.assertEqual(MAX_AI_ITEMS, result.report["processed_after_cap"])
        self.assertTrue(result.report["capped"])
        self.assertEqual(MAX_AI_ITEMS, result.report["ai_invocations"])
        self.assertEqual(MAX_AI_ITEMS, len(calls))
        self.assertTrue(result.report["limited_fetch"])

    def test_eligible_one_invokes_ai_once(self):
        calls = []
        result = validate(
            docs=[scix_doc()],
            keywords="pulsar",
            ai_runner_factory=ai_factory(calls),
        )
        self.assertEqual("PASS_E2E", result.status)
        self.assertEqual(1, result.report["eligible_after_filter"])
        self.assertEqual(1, result.report["processed_after_cap"])
        self.assertEqual(1, result.report["ai_invocations"])
        self.assertEqual(1, len(calls))

    def test_eligible_zero_invokes_ai_zero_times(self):
        factory_called = []
        result = validate(
            docs=[scix_doc()],
            keywords="magnetar",
            ai_runner_factory=lambda: factory_called.append(True),
        )
        self.assertEqual("PASS_NO_AI_CANDIDATES", result.status)
        self.assertEqual(0, result.report["eligible_after_filter"])
        self.assertEqual(0, result.report["processed_after_cap"])
        self.assertEqual(0, result.report["ai_invocations"])
        self.assertEqual([], factory_called)

    def test_no_filter_hit_is_not_an_api_failure(self):
        result = validate(
            docs=[scix_doc(title="Radio pulsar study")],
            keywords="FRB",
            ai_runner_factory=None,
        )
        self.assertEqual("PASS_NO_AI_CANDIDATES", result.status)
        self.assertEqual(1, result.report["scix_docs_received"])
        self.assertEqual(0, result.report["eligible_after_filter"])

    def test_canonical_cross_source_merge_and_single_ai_candidate(self):
        calls = []
        arxiv = arxiv_record("2608.00001", "arXiv abstract has priority.")
        scix = scix_doc(
            index=1,
            title="Pulsar validation paper",
            summary="SciX abstract should not replace arXiv abstract.",
        )
        result = validate(
            docs=[scix],
            keywords="pulsar",
            arxiv_records=[arxiv],
            ai_runner_factory=ai_factory(calls),
        )
        self.assertEqual("PASS_E2E", result.status)
        self.assertTrue(result.report["real_cross_source_duplicate_found"])
        self.assertEqual(1, result.report["canonical_after_merge"])
        self.assertEqual(1, result.report["ai_invocations"])
        self.assertEqual(1, len(calls))
        self.assertEqual("arxiv+scix", result.report["papers"][0]["source"])

    def test_history_dedup_removes_before_filter_and_ai(self):
        calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            history_root = Path(temp_dir)
            write_jsonl(
                [arxiv_record("2608.00001")],
                history_root / "2026-08-22.jsonl",
            )
            result = validate(
                docs=[scix_doc(index=1)],
                keywords="pulsar",
                history_dir=history_root,
                ai_runner_factory=ai_factory(calls),
            )
        self.assertEqual("PASS_NO_AI_CANDIDATES", result.status)
        self.assertEqual(1, result.report["history_duplicates_removed"])
        self.assertEqual(0, result.report["canonical_after_history_dedup"])
        self.assertEqual([], calls)

    def test_matched_keywords_are_reported_for_each_processed_paper(self):
        calls = []
        result = validate(
            docs=[
                scix_doc(
                    title="Pulsar and FRB validation paper",
                    summary="The summary mentions magnetar as well.",
                )
            ],
            keywords="pulsar,FRB,magnetar",
            ai_runner_factory=ai_factory(calls),
        )
        self.assertEqual("PASS_E2E", result.status)
        self.assertEqual(
            ["pulsar", "FRB", "magnetar"],
            result.report["papers"][0]["matched_keywords"],
        )
        self.assertEqual(
            {"pulsar": 1, "FRB": 1, "magnetar": 1},
            result.report["matched_keyword_counts"],
        )

    def test_ai_result_has_exact_six_fields_and_presence_is_reported(self):
        result = validate(
            docs=[scix_doc()],
            keywords="pulsar",
            ai_runner_factory=ai_factory([]),
        )
        paper = result.report["papers"][0]
        self.assertTrue(paper["abstract_translation_present"])
        for field in (
            "tldr_present",
            "motivation_present",
            "method_present",
            "result_present",
            "conclusion_present",
        ):
            self.assertTrue(paper[field])
        self.assertEqual(len("中文翻译 "), paper["abstract_translation_length"])

    def test_ai_exception_is_fail_ai(self):
        result = validate(
            docs=[scix_doc()],
            keywords="pulsar",
            ai_runner_factory=ai_factory([], error=RuntimeError("not logged")),
        )
        self.assertEqual("FAIL_AI", result.status)
        self.assertEqual(1, result.report["ai_invocations"])

    def test_schema_failure_is_fail_output_schema(self):
        invalid = {"tldr": "missing five required fields"}
        result = validate(
            docs=[scix_doc()],
            keywords="pulsar",
            ai_runner_factory=ai_factory([], result=invalid),
        )
        self.assertEqual("FAIL_OUTPUT_SCHEMA", result.status)

    def test_report_is_written_only_to_supplied_temp_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "runner-output"
            result = validate(
                docs=[scix_doc()],
                keywords="FRB",
                output_dir=output_dir,
            )
            self.assertEqual("PASS_NO_AI_CANDIDATES", result.status)
            report_path = output_dir / "scix_e2e_validation_report.json"
            self.assertTrue(report_path.exists())
            self.assertNotIn("data", {path.name for path in output_dir.iterdir()})

    def test_secret_and_authorization_are_not_in_report(self):
        secret = "secret-api-value"
        result = validate(
            docs=[scix_doc(title=secret)],
            keywords="pulsar",
            ai_runner_factory=ai_factory([], secret=secret),
            secret=secret,
        )
        rendered = json.dumps(result.report, ensure_ascii=False)
        self.assertNotIn(secret, rendered)
        self.assertNotIn("Authorization", rendered)

    def test_harness_has_no_git_or_production_data_write_entry_point(self):
        source = HARNESS_PATH.read_text(encoding="utf-8")
        self.assertNotIn("git push", source)
        self.assertNotIn("git add", source)
        self.assertNotIn("git checkout", source)
        self.assertNotIn("write_jsonl", source)
        self.assertNotIn("data/*.jsonl", source)

    def test_real_identifier_and_doi_normalization_are_reported(self):
        result = validate(
            docs=[scix_doc(index=1)],
            keywords="pulsar",
            ai_runner_factory=ai_factory([]),
        )
        paper = result.report["papers"][0]
        self.assertEqual("2608.00001", paper["arxiv_id"])
        self.assertEqual("2026arXiv00001V", paper["bibcode"])
        self.assertEqual("10.48550/arxiv.2608.00001", paper["doi"])

    def test_workflow_has_three_validation_modes_and_unaffected_schedule(self):
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn("validation_mode:", workflow)
        self.assertIn("- normal", workflow)
        self.assertIn("- scix-smoke", workflow)
        self.assertIn("- scix-e2e", workflow)
        self.assertIn("inputs.validation_mode == 'scix-e2e'", workflow)
        self.assertIn("inputs.validation_mode == 'scix-smoke'", workflow)
        self.assertIn('cron: "30 17 * * *"', workflow)
        self.assertIn("python -m daily_arxiv.daily_arxiv.scix_e2e_validate", workflow)

    def test_smoke_and_e2e_jobs_do_not_run_normal_build_on_custom_validation(self):
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "if: ${{ github.event_name != 'workflow_dispatch' || github.ref_name != 'astro-custom' || inputs.validation_mode == 'normal' }}",
            workflow,
        )
        self.assertIn(
            "if: ${{ github.event_name == 'workflow_dispatch' && github.ref_name == 'astro-custom' && inputs.validation_mode == 'scix-e2e' }}",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
