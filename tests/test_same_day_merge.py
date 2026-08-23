import json
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from daily_arxiv.daily_arxiv.same_day_merge import (
    main,
    merge_files,
    merge_records,
    subtract_files,
    subtract_records,
)


def paper(paper_id, **extra):
    record = {
        "id": paper_id,
        "title": f"Paper {paper_id}",
        "summary": f"Summary {paper_id}",
        "authors": ["A. Author"],
        "categories": ["astro-ph.HE"],
    }
    record.update(extra)
    return record


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


class SameDayMergeTests(unittest.TestCase):
    def test_zero_keyword_matches_keep_existing_without_ai_or_empty_final(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current = root / "current.jsonl"
            existing = root / "2026-08-23_AI_enhanced_Chinese.jsonl"
            new_candidates = root / "2026-08-23_new.jsonl"
            staged_ai = root / "2026-08-23_new_AI_enhanced_Chinese.jsonl"
            final = root / "2026-08-23_AI_enhanced_Chinese.jsonl"
            write_jsonl(existing, [paper("A"), paper("B")])
            write_jsonl(current, [paper("A"), paper("B"), paper("C"), paper("D")])

            new_count = subtract_files(current, existing, new_candidates)
            self.assertEqual(2, new_count)
            self.assertEqual(["C", "D"], [row["id"] for row in json_lines(new_candidates)])

            enhance = load_enhance_module()
            old_argv = sys.argv
            try:
                sys.argv = [
                    "enhance.py",
                    "--data",
                    str(new_candidates),
                ]
                with patch.dict(
                    os.environ,
                    {"FILTER_KEYWORDS": "magnetar", "LANGUAGE": "Chinese"},
                    clear=False,
                ), patch.object(
                    enhance,
                    "process_all_items",
                    side_effect=AssertionError("DeepSeek processing must not be called"),
                ), contextlib.redirect_stderr(io.StringIO()) as stderr:
                    enhance.main()
            finally:
                sys.argv = old_argv

            self.assertEqual("", staged_ai.read_text(encoding="utf-8"))
            self.assertIn("筛选后唯一论文数量: 0", stderr.getvalue())

            merge_files(existing, staged_ai, final)
            self.assertEqual(["A", "B"], [row["id"] for row in json_lines(final)])

    def test_existing_absent_current_ab_returns_ab(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current = root / "current.jsonl"
            output = root / "new.jsonl"
            write_jsonl(current, [paper("A"), paper("B")])

            self.assertEqual(2, subtract_files(current, root / "missing.jsonl", output))
            self.assertEqual(["A", "B"], [row["id"] for row in json_lines(output)])

    def test_existing_ab_current_abcd_returns_cd(self):
        result = subtract_records(
            [paper("A"), paper("B"), paper("C"), paper("D")],
            [paper("A"), paper("B")],
        )
        self.assertEqual(["C", "D"], [row["id"] for row in result])

    def test_existing_abcd_current_abcd_returns_empty(self):
        result = subtract_records(
            [paper("A"), paper("B"), paper("C"), paper("D")],
            [paper("A"), paper("B"), paper("C"), paper("D")],
        )
        self.assertEqual([], result)

    def test_merge_existing_ab_new_cd_returns_abcd(self):
        result = merge_records(
            [paper("A"), paper("B")],
            [paper("C"), paper("D")],
        )
        self.assertEqual(["A", "B", "C", "D"], [row["id"] for row in result])

    def test_arxiv_and_scix_duplicate_merge_to_one(self):
        existing = paper("2608.12345", source="arxiv")
        scix = paper(
            "2026arXiv260812345X",
            source="scix",
            identifiers=["arXiv:2608.12345"],
        )
        result = merge_records([existing], [scix])
        self.assertEqual([existing], result)

    def test_doi_duplicate_merge_to_one(self):
        existing = paper("doi-a", doi="10.1234/Example.1")
        duplicate = paper("doi-b", doi="https://doi.org/10.1234/example.1")
        result = merge_records([existing], [duplicate])
        self.assertEqual([existing], result)

    def test_bibcode_duplicate_merge_to_one(self):
        existing = paper("bib-a", bibcode="2026ApJ...001A")
        duplicate = paper("bib-b", bibcode="2026ApJ...001A")
        result = merge_records([existing], [duplicate])
        self.assertEqual([existing], result)

    def test_yesterday_records_are_not_loaded_for_today(self):
        today_existing = []
        yesterday = paper("A")
        today = paper("C")
        result = merge_records(today_existing, [today])
        self.assertEqual([today], result)
        self.assertNotIn(yesterday, result)

    def test_existing_today_file_missing_is_first_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            new_ai = root / "new_ai.jsonl"
            output = root / "final.jsonl"
            write_jsonl(new_ai, [paper("A"), paper("B")])

            merge_files(root / "missing_existing.jsonl", new_ai, output)
            self.assertEqual(["A", "B"], [row["id"] for row in json_lines(output)])

    def test_malformed_existing_fails_without_overwriting_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing = root / "existing.jsonl"
            current = root / "current.jsonl"
            output = root / "new.jsonl"
            existing.write_text('{"id":"A"}\nnot-json\n', encoding="utf-8")
            write_jsonl(current, [paper("C")])
            output.write_text("sentinel\n", encoding="utf-8")

            self.assertEqual(
                2,
                main(
                    [
                        "subtract",
                        "--current",
                        str(current),
                        "--existing",
                        str(existing),
                        "--output",
                        str(output),
                    ]
                ),
            )
            self.assertEqual("sentinel\n", output.read_text(encoding="utf-8"))

    def test_malformed_new_ai_fails_without_overwriting_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing = root / "existing.jsonl"
            new_ai = root / "new_ai.jsonl"
            output = root / "final.jsonl"
            write_jsonl(existing, [paper("A")])
            new_ai.write_text('{"id":"C"}\nnot-json\n', encoding="utf-8")
            output.write_text("sentinel\n", encoding="utf-8")

            self.assertEqual(2, main(["merge", "--existing", str(existing), "--new-ai", str(new_ai), "--output", str(output)]))
            self.assertEqual("sentinel\n", output.read_text(encoding="utf-8"))

    def test_empty_new_ai_keeps_valid_existing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing = root / "existing.jsonl"
            new_ai = root / "new_ai.jsonl"
            output = root / "final.jsonl"
            write_jsonl(existing, [paper("A"), paper("B")])
            new_ai.write_text("", encoding="utf-8")

            merge_files(existing, new_ai, output)
            self.assertEqual(
                ["A", "B"], [row["id"] for row in json_lines(output)]
            )

    def test_existing_absent_valid_new_ai_is_final(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            new_ai = root / "new_ai.jsonl"
            output = root / "final.jsonl"
            write_jsonl(new_ai, [paper("C"), paper("D")])

            merge_files(root / "missing_existing.jsonl", new_ai, output)
            self.assertEqual(["C", "D"], [row["id"] for row in json_lines(output)])

    def test_repeated_merge_is_idempotent(self):
        existing = [paper("A"), paper("B")]
        new_ai = [paper("B"), paper("C")]
        once = merge_records(existing, new_ai)
        twice = merge_records(once, once)
        self.assertEqual(once, twice)

    def test_language_paths_do_not_cross_accumulate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chinese_existing = root / "2026-08-23_AI_enhanced_Chinese.jsonl"
            english_existing = root / "2026-08-23_AI_enhanced_English.jsonl"
            new_english = root / "2026-08-23_new_AI_enhanced_English.jsonl"
            output = root / "2026-08-23_AI_enhanced_English.jsonl"
            write_jsonl(chinese_existing, [paper("ZH")])
            write_jsonl(new_english, [paper("EN")])

            merge_files(english_existing, new_english, output)
            self.assertEqual(["EN"], [row["id"] for row in json_lines(output)])

    def test_fallback_id_identity_is_deterministic(self):
        existing = [{"id": "fallback-1", "AI": {"tldr": "old"}}]
        new = [{"id": "fallback-1", "AI": {"tldr": "new"}}]
        self.assertEqual(existing, merge_records(existing, new))


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_enhance_module():
    repo_root = Path(__file__).resolve().parents[1]
    ai_dir = repo_root / "ai"
    module_name = "enhance_zero_keyword_test"
    spec = importlib.util.spec_from_file_location(module_name, ai_dir / "enhance.py")
    module = importlib.util.module_from_spec(spec)
    old_cwd = Path.cwd()
    sys.path.insert(0, str(ai_dir))
    try:
        os.chdir(ai_dir)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
        sys.path.pop(0)
    return module


if __name__ == "__main__":
    unittest.main()
