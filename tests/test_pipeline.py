import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from claims_forge import cli
from claims_forge.artifact_validator import validate_agent_artifact
from claims_forge.assemble import DEFAULT_CONDITIONS, build_artifact, grounded_conditions
from claims_forge.gateway import Budget, parse_json_object
from claims_forge.pipeline import Candidate, _select, apply_reviews, fallback_claims, resolve_evidence
from claims_forge.source import Page, _mark_references, clean_text, locate_quote, render_for_model
import time


def pages_fixture() -> list[Page]:
    return [
        Page("p-span-0001", 1, "Abstract\nWe found that compound X reduced tumour growth by 40% in mice."),
        Page("p-span-0002", 2, "Results\nCompound X increased survival significantly compared with vehicle control."),
    ]


def candidate(statement: str, importance: str = "central", pages=(1,), quote: str = "") -> Candidate:
    return Candidate(statement=statement, conditions="in mice", importance=importance, pages=list(pages), quote=quote)


class SourceTests(unittest.TestCase):
    def test_clean_text_joins_hyphenated_line_breaks(self):
        self.assertEqual(clean_text("sig-\nnificant  result"), "significant result")

    def test_references_are_excluded_from_model_view(self):
        pages = [Page(f"p-span-{i:04d}", i, f"body {i}") for i in range(1, 4)]
        pages.append(Page("p-span-0004", 4, "conclusion text\nReferences\n[1] Someone"))
        pages.append(Page("p-span-0005", 5, "[2] Other"))
        marked = _mark_references(pages)
        self.assertEqual(marked[3].body, "conclusion text")
        self.assertEqual(marked[4].body, "")
        rendered = render_for_model(marked, 10_000)
        self.assertNotIn("Someone", rendered)
        self.assertIn("[PAGE 4]", rendered)

    def test_long_papers_keep_first_and_last_pages(self):
        pages = [Page(f"p-span-{i:04d}", i, str(i) * 1000) for i in range(1, 11)]
        rendered = render_for_model(pages, 3500)
        self.assertIn("[PAGE 1]", rendered)
        self.assertIn("[PAGE 10]", rendered)
        self.assertLessEqual(len(rendered), 3700)

    def test_locate_quote_ignores_whitespace_and_returns_page_text(self):
        pages = [Page("p-span-0001", 1, "The effect\nwas large and durable.")]
        located = locate_quote(pages, "effect was large", [1])
        self.assertIsNotNone(located)
        self.assertEqual(located[1], "effect\nwas large")


class SelectionTests(unittest.TestCase):
    def test_central_claims_rank_first_and_output_is_capped(self):
        rows = [candidate(f"Finding number {i} concerns distinct topic alpha{i} beta{i} gamma{i}",
                          "minor" if i < 6 else "central") for i in range(14)]
        selected = _select(rows)
        self.assertEqual(len(selected), 10)
        self.assertEqual(selected[0].importance, "central")

    def test_near_duplicates_are_removed(self):
        rows = [candidate("Compound X reduced tumour growth in mice by 40 percent"),
                candidate("Compound X reduced tumour growth by 40 percent in mice")]
        self.assertEqual(len(_select(rows)), 1)

    def test_reviews_drop_fix_and_preserve_unreviewed(self):
        rows = [candidate("Claim zero about topic one"), candidate("Claim one about topic two"),
                candidate("Claim two about topic three")]
        result = apply_reviews(rows, [
            {"id": "c0", "verdict": "drop"},
            {"id": "c1", "verdict": "fix", "statement": "Fixed claim about topic two", "importance": "minor"},
        ])
        self.assertEqual([row.statement for row in result], ["Fixed claim about topic two", "Claim two about topic three"])
        self.assertEqual(result[0].importance, "minor")


class EvidenceTests(unittest.TestCase):
    def test_verified_quote_adds_its_page(self):
        cited, quote = resolve_evidence(candidate("x", pages=(2,), quote="compound X reduced tumour growth"), pages_fixture())
        self.assertEqual({page.number for page in cited}, {1, 2})
        self.assertEqual(quote, "compound X reduced tumour growth")

    def test_abstract_page_is_always_cited(self):
        cited, _ = resolve_evidence(candidate("survival increased", pages=(2,), quote="Compound X increased survival"), pages_fixture())
        self.assertIn(1, [page.number for page in cited])

    def test_invented_quote_falls_back_to_real_page_text(self):
        pages = pages_fixture()
        _, quote = resolve_evidence(candidate("survival increased with compound X", pages=(2,), quote="never written here anywhere"), pages)
        self.assertIn(quote, pages[1].text)

    def test_artifact_passes_validator_and_cites_page_spans(self):
        rows = [candidate("Compound X reduced tumour growth by 40% in mice", quote="reduced tumour growth by 40%"),
                candidate("Compound X increased survival compared with vehicle control", pages=(2,)),
                candidate("Compound X has a third distinct effect on angiogenesis markers", pages=(2,))]
        artifact = build_artifact({"paper_id": "p", "title": "T", "year": "bad", "authors": None}, pages_fixture(), rows, {})
        self.assertEqual(validate_agent_artifact(artifact), [])
        self.assertEqual(artifact.logic.claims[0].sources[0].span_ids, ["p-span-0001"])
        self.assertEqual(artifact.logic.claims[0].metadata["importance"], "central")

    def test_conditions_kept_only_when_cited_pages_use_those_words(self):
        pages = pages_fixture()
        self.assertEqual(grounded_conditions("in rats", pages[:1]), DEFAULT_CONDITIONS)
        self.assertEqual(grounded_conditions("in mice", pages[:1]), "in mice")
        self.assertEqual(grounded_conditions("tumour growth", pages[:1]), "tumour growth")
        self.assertEqual(grounded_conditions("in murine macrophages", pages), DEFAULT_CONDITIONS)

    def test_fallback_always_produces_a_claim(self):
        self.assertTrue(fallback_claims([Page("p-span-0001", 1, "No sentence cues here at all")]))


class GatewayTests(unittest.TestCase):
    def test_parse_tolerates_fences_and_prose(self):
        self.assertEqual(parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_object('Here: {"a": 1} done'), {"a": 1})

    def test_parse_salvages_rows_from_truncated_output(self):
        truncated = '{"claims": [{"statement": "one"}, {"statement": "two"}, {"statement": "thr'
        self.assertEqual(parse_json_object(truncated), {"claims": [{"statement": "one"}, {"statement": "two"}]})

    def test_budget_keeps_a_safety_margin(self):
        budget = Budget(token_limit=1000, deadline=time.monotonic() + 600)
        self.assertTrue(budget.admits(500, 300))
        self.assertFalse(budget.admits(700, 300))


class _Gateway(BaseHTTPRequestHandler):
    reply = "{}"
    models: list[str] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).models.append(body["model"])
        event = {"choices": [{"delta": {"content": type(self).reply}}]}
        usage = {"choices": [], "usage": {"total_tokens": 1000}}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(f"data: {json.dumps(event)}\n\ndata: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())

    def log_message(self, *args):
        pass


class EndToEndTests(unittest.TestCase):
    def run_cli(self, reply: str):
        _Gateway.reply, _Gateway.models = reply, []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Gateway)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "paper.pdf").write_bytes(b"%PDF-1.4")
            request = {"schema_version": cli.REQUEST_SCHEMA, "paper": {"paper_id": "p", "title": "T"}, "pdf_path": "paper.pdf",
                       "inference": {"model": "a/b", "allowed_models": ["a/b"], "stream": True, "max_output_tokens": 8192,
                                     "budget": {"token_limit": 200000, "remaining_tokens": 200000}}}
            (root / "request.json").write_text(json.dumps(request))
            env = {"CLAIMS_INFERENCE_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1", "CLAIMS_INFERENCE_TOKEN": "t"}
            with mock.patch.dict("os.environ", env), mock.patch.object(cli, "read_pages", return_value=pages_fixture()):
                cli.run(root / "request.json", root / "out" / "output.json")
            return json.loads((root / "out" / "output.json").read_text())

    def test_full_run_writes_valid_document_and_uses_gateway_prefix(self):
        claims = [{"statement": f"Compound X changes distinct biomarker number{i} level{i} markedly in mice",
                   "conditions": "mice", "importance": "central", "pages": [1], "quote": "reduced tumour growth"} for i in range(4)]
        output = self.run_cli(json.dumps({"claims": claims}))
        self.assertEqual(len(output["logic"]["claims"]), 4)
        self.assertTrue(all(model == "openrouter/a/b" for model in _Gateway.models))
        self.assertEqual(output["paper"]["paper_id"], "p")

    def test_unusable_model_output_still_writes_a_document(self):
        output = self.run_cli("not json at all")
        self.assertGreaterEqual(len(output["logic"]["claims"]), 1)


class ContractTests(unittest.TestCase):
    def test_source_has_no_disallowed_literals(self):
        root = Path(__file__).resolve().parents[1] / "src" / "claims_forge"
        text = "".join(path.read_text(encoding="utf-8") for path in root.glob("*.py")).casefold()
        for marker in ("input" + ".json", "claims_competition_rounds" + "_dir", "competition_round" + "_id"):
            self.assertNotIn(marker, text)


if __name__ == "__main__":
    unittest.main()
