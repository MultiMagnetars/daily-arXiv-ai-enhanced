import ast
import os
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = ROOT_DIR / "ai"

# Load only the two pure filtering helpers so these tests do not require
# optional LLM/runtime dependencies.
source_path = AI_DIR / "enhance.py"
source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
helper_nodes = [
    node
    for node in source_tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in {"parse_filter_keywords", "filter_papers_by_keywords"}
]
helper_namespace = {
    "Dict": Dict,
    "List": List,
    "Optional": Optional,
    "Tuple": Tuple,
}
exec(
    compile(
        ast.Module(body=helper_nodes, type_ignores=[]),
        str(source_path),
        "exec",
    ),
    helper_namespace,
)
parse_filter_keywords = helper_namespace["parse_filter_keywords"]
filter_papers_by_keywords = helper_namespace["filter_papers_by_keywords"]


def paper(paper_id, title="", summary=""):
    return {
        "id": paper_id,
        "title": title,
        "summary": summary,
    }


class KeywordFilterTests(unittest.TestCase):
    def test_case_insensitive_match(self):
        papers = [paper("1", title="NEUTRON STAR OBSERVATIONS")]
        keywords = parse_filter_keywords("neutron star")

        filtered, counts = filter_papers_by_keywords(papers, keywords)

        self.assertEqual(["1"], [item["id"] for item in filtered])
        self.assertEqual({"neutron star": 1}, counts)

    def test_title_match(self):
        papers = [paper("1", title="Magnetar bursts")]
        keywords = parse_filter_keywords("magnetar")

        filtered, counts = filter_papers_by_keywords(papers, keywords)

        self.assertEqual(["1"], [item["id"] for item in filtered])
        self.assertEqual(1, counts["magnetar"])

    def test_summary_match(self):
        papers = [paper("1", summary="We measure a black hole accretion disk.")]
        keywords = parse_filter_keywords("black hole")

        filtered, counts = filter_papers_by_keywords(papers, keywords)

        self.assertEqual(["1"], [item["id"] for item in filtered])
        self.assertEqual(1, counts["black hole"])

    def test_space_phrase_and_hyphen_phrase(self):
        papers = [
            paper("1", summary="A black hole candidate is studied."),
            paper("2", summary="We evaluate a self-supervised model."),
        ]
        keywords = parse_filter_keywords("black hole,self-supervised")

        filtered, counts = filter_papers_by_keywords(papers, keywords)

        self.assertEqual(["1", "2"], [item["id"] for item in filtered])
        self.assertEqual(1, counts["black hole"])
        self.assertEqual(1, counts["self-supervised"])

    def test_or_match_and_per_keyword_counts(self):
        papers = [
            paper("1", title="Black hole and neutron star systems"),
            paper("2", summary="A magnetar observation"),
            paper("3", title="Unrelated paper"),
        ]
        keywords = parse_filter_keywords("black hole,neutron star,magnetar")

        filtered, counts = filter_papers_by_keywords(papers, keywords)

        self.assertEqual(["1", "2"], [item["id"] for item in filtered])
        self.assertEqual(1, counts["black hole"])
        self.assertEqual(1, counts["neutron star"])
        self.assertEqual(1, counts["magnetar"])

    def test_empty_filter_keywords_passes_all(self):
        papers = [
            paper("1", title="First paper"),
            paper("2", summary="Second paper"),
        ]

        for raw_value in (None, "", "   ", ", ,"):
            with self.subTest(raw_value=raw_value):
                environment = (
                    {}
                    if raw_value is None
                    else {"FILTER_KEYWORDS": raw_value}
                )
                with patch.dict(os.environ, environment, clear=True):
                    configured_value = os.environ.get("FILTER_KEYWORDS")
                keywords = parse_filter_keywords(configured_value)
                filtered, counts = filter_papers_by_keywords(papers, keywords)

                self.assertEqual(["1", "2"], [item["id"] for item in filtered])
                self.assertEqual({}, counts)

    def test_zero_matches_returns_empty_result(self):
        papers = [
            paper("1", title="Magnetar paper"),
            paper("2", summary="Neutron star paper"),
        ]
        keywords = parse_filter_keywords("black hole")

        filtered, counts = filter_papers_by_keywords(papers, keywords)

        self.assertEqual([], filtered)
        self.assertEqual({"black hole": 0}, counts)


if __name__ == "__main__":
    unittest.main()
