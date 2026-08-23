import ast
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = ROOT_DIR / "ai"


class FakeResponse:
    def model_dump(self):
        return {
            "abstract_translation": "这是完整中文翻译……",
            "tldr": "核心结论",
            "motivation": "研究动机",
            "method": "研究方法",
            "result": "研究结果",
            "conclusion": "研究结论",
        }


class CapturingChain:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def invoke(self, payload):
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return FakeResponse()


class AbstractTranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from langchain.prompts import ChatPromptTemplate
        from ai.structure import Structure

        cls.Structure = Structure
        cls.ChatPromptTemplate = ChatPromptTemplate

        previous_cwd = Path.cwd()
        sys.path.insert(0, str(AI_DIR))
        os.chdir(AI_DIR)
        try:
            module_spec = importlib.util.spec_from_file_location(
                "phase4_enhance_for_tests",
                AI_DIR / "enhance.py",
            )
            cls.enhance = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(cls.enhance)
        finally:
            os.chdir(previous_cwd)

    def test_structure_has_exactly_six_fields_and_model_dump(self):
        expected = [
            "abstract_translation",
            "tldr",
            "motivation",
            "method",
            "result",
            "conclusion",
        ]
        self.assertEqual(expected, list(self.Structure.model_fields))
        model = self.Structure(**FakeResponse().model_dump())
        self.assertEqual(expected, list(model.model_dump()))

    def test_prompt_has_exactly_three_variables_and_renders(self):
        system_prompt = (AI_DIR / "system.txt").read_text(encoding="utf-8")
        template_prompt = (AI_DIR / "template.txt").read_text(encoding="utf-8")
        prompt = self.ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", template_prompt),
            ]
        )

        self.assertEqual({"language", "title", "content"}, set(prompt.input_variables))
        messages = prompt.format_messages(
            language="Chinese",
            title="Single-pulse variability of a radio pulsar",
            content="We report observations of $10^{-6}$ with DM and RM; may suggest...",
        )
        rendered = "\n".join(message.content for message in messages)
        self.assertIn("Single-pulse variability of a radio pulsar", rendered)
        self.assertIn("We report observations of", rendered)
        self.assertIn("10^{-6}", rendered)
        self.assertIn("abstract_translation", rendered)
        self.assertIn("Task 2", rendered)
        self.assertIn("exactly the six fields", rendered)

    def test_process_single_item_invokes_chain_once_and_writes_six_fields(self):
        chain = CapturingChain()
        item = {
            "id": "test",
            "title": "A radio pulsar",
            "summary": "We report observations of a pulsar.",
        }

        result = self.enhance.process_single_item(chain, item, "Chinese")

        self.assertEqual(1, len(chain.calls))
        self.assertEqual(
            {
                "language": "Chinese",
                "title": "A radio pulsar",
                "content": "We report observations of a pulsar.",
            },
            chain.calls[0],
        )
        self.assertEqual(set(FakeResponse().model_dump()), set(result["AI"]))

    def test_output_parser_and_generic_exception_fallbacks_have_six_fields(self):
        parser_error = self.enhance.langchain_core.exceptions.OutputParserException(
            "structured output failed"
        )
        for error in (parser_error, RuntimeError("request failed")):
            with self.subTest(error=type(error).__name__):
                chain = CapturingChain(error=error)
                result = self.enhance.process_single_item(
                    chain,
                    {
                        "id": "test",
                        "title": "A radio pulsar",
                        "summary": "We report observations.",
                    },
                    "Chinese",
                )
                self.assertEqual(
                    {
                        "abstract_translation",
                        "tldr",
                        "motivation",
                        "method",
                        "result",
                        "conclusion",
                    },
                    set(result["AI"]),
                )

    def test_future_fallback_source_has_six_fields(self):
        source = (AI_DIR / "enhance.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        six_field_dicts = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = {
                    key.value
                    for key in node.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                if {"abstract_translation", "tldr", "conclusion"}.issubset(keys):
                    six_field_dicts.append(keys)
        self.assertGreaterEqual(len(six_field_dicts), 2)

    def test_future_exception_fallback_has_six_fields(self):
        class DummyLLM:
            def with_structured_output(self, *args, **kwargs):
                return self

        class DummyPrompt:
            def __or__(self, other):
                return object()

        data = [
            {
                "id": "future-error",
                "title": "A radio pulsar",
                "summary": "We report observations.",
            }
        ]
        with patch.object(
            self.enhance,
            "ChatOpenAI",
            return_value=DummyLLM(),
        ), patch.object(
            self.enhance.ChatPromptTemplate,
            "from_messages",
            return_value=DummyPrompt(),
        ), patch.object(
            self.enhance,
            "process_single_item",
            side_effect=RuntimeError("future failed"),
        ):
            result = self.enhance.process_all_items(data, "fake-model", "Chinese", 1)

        self.assertEqual(
            {
                "abstract_translation",
                "tldr",
                "motivation",
                "method",
                "result",
                "conclusion",
            },
            set(result[0]["AI"]),
        )

    def test_index_loads_mathjax_config_before_cdn(self):
        index_html = (ROOT_DIR / "index.html").read_text(encoding="utf-8")
        config_position = index_html.index("window.MathJax =")
        script_position = index_html.index(
            "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
        )

        self.assertLess(config_position, script_position)
        self.assertIn("['$', '$']", index_html)
        self.assertIn("['\\\\(', '\\\\)']", index_html)
        self.assertIn("['$$', '$$']", index_html)
        self.assertIn("['\\\\[', '\\\\]']", index_html)
        self.assertIn("'script'", index_html)
        self.assertIn("'pre'", index_html)

    def test_frontend_new_old_data_xss_and_latex(self):
        node_script = r"""
const fs = require('fs');
const vm = require('vm');

const source = fs.readFileSync('js/app.js', 'utf8');
const elements = {};
function element(id) {
  if (!elements[id]) {
    elements[id] = {
      innerHTML: '',
      textContent: '',
      href: '',
      style: {},
      scrollTop: 0,
      classList: {
        add: function() {},
        remove: function() {},
        contains: function() { return false; }
      }
    };
  }
  return elements[id];
}

const document = {
  addEventListener: function() {},
  body: { style: {} },
  getElementById: function(id) { return element(id); }
};
const context = {
  console: console,
  document: document,
  window: {},
  Date: Date,
  setTimeout: setTimeout,
  clearTimeout: clearTimeout
};
vm.createContext(context);
vm.runInContext(source, context);

const newRecord = {
  id: 'test',
  title: 'J1637$-$4642 radio pulsar',
  authors: ['A. Author'],
  categories: ['astro-ph.HE'],
  summary: 'We report observations of $10^{-6}$.',
  abs: 'https://arxiv.org/abs/1234.5678',
  AI: {
    abstract_translation: "<script>alert(1)</script><img src=x onerror=alert(1)> & < > \" ' $10^{-6}$ $\\Delta\\nu/\\nu$ $\\dot{\\nu}$ $\\sim 100$ $\\pm 0.1$ $\\approx 0.0187$ J1637$-$4642 $B(t)/B_0$ $$E = mc^2$$ \\(\\nu = 1/P\\) \\[\\dot{P} > 0\\] 归一化频率变化为 $\\Delta\\nu/\\nu \\sim 2.7\\times10^{-6}$。",
    tldr: '核心结论',
    motivation: '动机',
    method: '方法',
    result: '结果',
    conclusion: '结论'
  }
};
const parsed = context.parseJsonlData(JSON.stringify(newRecord), '2026-01-01');
const paper = parsed['astro-ph.HE'][0];
if (paper.summary !== '核心结论') throw new Error('homepage summary is not tldr');
if (!paper.abstractTranslation) throw new Error('translation was not parsed');

context.showPaperDetails(paper, 1);
let html = element('modalBody').innerHTML;
if (!element('modalTitle').innerHTML.includes('J1637$-$4642')) throw new Error('title LaTeX was not preserved');
if (!html.includes('<details open><summary>中文直译')) throw new Error('Chinese translation is not expanded');
if (!html.includes('<summary>English original</summary>')) throw new Error('English original is missing');
if (html.includes('<script>alert(1)</script>') || html.includes('<img src=x')) throw new Error('translation HTML was not escaped');
if (!html.includes('&lt;script&gt;')) throw new Error('escaped translation is missing');
for (const sample of [
  '$10^{-6}$',
  '$\\Delta\\nu/\\nu$',
  '$\\dot{\\nu}$',
  '$\\sim 100$',
  '$\\pm 0.1$',
  '$\\approx 0.0187$',
  'J1637$-$4642',
  '$B(t)/B_0$',
  '$$E = mc^2$$',
  '\\(\\nu = 1/P\\)',
  '\\[\\dot{P} &gt; 0\\]',
  '归一化频率变化为 $\\Delta\\nu/\\nu \\sim 2.7\\times10^{-6}$。'
]) {
  if (!html.includes(sample)) throw new Error(`LaTeX sample is missing: ${sample}`);
}

const oldRecord = {
  id: 'old',
  title: 'Old paper',
  authors: ['A. Author'],
  categories: ['astro-ph.HE'],
  summary: 'Old English abstract.',
  abs: 'https://arxiv.org/abs/1234.5678',
  AI: {
    tldr: '旧结论',
    motivation: '旧动机',
    method: '旧方法',
    result: '旧结果',
    conclusion: '旧结论'
  }
};
const oldParsed = context.parseJsonlData(JSON.stringify(oldRecord), '2026-01-01');
const oldPaper = oldParsed['astro-ph.HE'][0];
if (oldPaper.abstractTranslation !== '') throw new Error('old data fallback failed');
context.showPaperDetails(oldPaper, 1);
html = element('modalBody').innerHTML;
if (html.includes('中文直译')) throw new Error('old data rendered empty translation');
if (!html.includes('English original')) throw new Error('old data lost English original');

const missingSummaryRecord = Object.assign({}, oldRecord, {
  AI: Object.assign({}, oldRecord.AI)
});
delete missingSummaryRecord.summary;
delete missingSummaryRecord.AI.tldr;
const missingSummaryParsed = context.parseJsonlData(
  JSON.stringify(missingSummaryRecord),
  '2026-01-01'
);
context.showPaperDetails(missingSummaryParsed['astro-ph.HE'][0], 1);

const invalidValues = [null, '', 123, ['bad']];
for (const value of invalidValues) {
  const record = Object.assign({}, oldRecord, {
    AI: Object.assign({}, oldRecord.AI, { abstract_translation: value })
  });
  const invalidParsed = context.parseJsonlData(JSON.stringify(record), '2026-01-01');
  if (invalidParsed['astro-ph.HE'][0].abstractTranslation !== '') {
    throw new Error('invalid translation value was not safely ignored');
  }
}

const modalBody = element('modalBody');
const typesetTargets = [];
const clearTargets = [];
context.window.MathJax = {
  typesetClear: function(containers) { clearTargets.push(containers[0]); },
  typesetPromise: function(containers) {
    typesetTargets.push(containers[0]);
    return Promise.resolve();
  }
};
context.showPaperDetails(oldPaper, 1);
context.showPaperDetails(paper, 1);
if (typesetTargets.length !== 2 || typesetTargets.some(target => target !== modalBody)) {
  throw new Error('MathJax was not scoped to modalBody for each modal render');
}
if (clearTargets.length !== 2 || clearTargets.some(target => target !== modalBody)) {
  throw new Error('MathJax typesetClear was not scoped to modalBody');
}

let warningCount = 0;
const originalWarn = context.console.warn;
context.console.warn = function(message) {
  if (String(message).includes('MathJax typesetting failed')) warningCount += 1;
};
context.window.MathJax = {
  typesetClear: function() {},
  typesetPromise: function() {
    return Promise.reject(new Error('typeset failed'));
  }
};
context.showPaperDetails(oldPaper, 1);

setTimeout(() => {
  if (warningCount !== 1) throw new Error('MathJax rejection was not handled with one warning');
  context.console.warn = originalWarn;
  context.window.MathJax = undefined;
  context.showPaperDetails(paper, 1);
  console.log('frontend checks PASS');
}, 0);
"""
        completed = subprocess.run(
            ["node", "-e", node_script],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("frontend checks PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main()
