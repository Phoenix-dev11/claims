"""Local approximation of the competition's Silver scoring.

Runs extraction systems on real papers through a local gateway, then judges
every claim with the subnet's own eligibility instructions and the same model.
It approximates the real validator; it is not the validator.

Usage: eval/.venv/bin/python eval/harness.py --tag v1 --papers 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval" / "data"
CACHE = ROOT / "eval" / "cache"
RESULTS = ROOT / "eval" / "results"
CLAIMS_REPO = Path(os.environ.get("CLAIMS_REPO", str(ROOT / "claims-run")))
SKILLS = CLAIMS_REPO / "validator" / "agent_v1" / "skills"
UPSTREAM = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v4-flash-0731"
WEIGHTS = {"central": 0.70, "supporting": 0.30, "minor": 0.10}
MINOR_CAP = 0.05
GATES = {
    "propositional_completeness": "Is the exact candidate a complete, evaluable scientific proposition rather than a fragment, label, topic, or underspecified assertion?",
    "author_assertion": "Does the paper itself assert the exact candidate, rather than merely mention, cite, hypothesize, or describe it as background?",
    "paper_original_support": "Do the candidate's linked source spans provide direct support generated or analyzed by this paper for every material part of the exact candidate?",
    "argument_sufficiency": "Do the linked evidence and reasoning support the candidate without an omitted material inferential step or unsupported component?",
    "fidelity": "Is the exact candidate faithful to the paper, including direction, population, conditions, modality, uncertainty, and material qualifications?",
    "author_marked_salience": "Is the candidate presented by the authors as a meaningful paper finding rather than an incidental observation, method detail, or peripheral fact?",
}


def api_key() -> str:
    for line in Path("/root/work/sn111/.env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("OPENROUTER_API_KEY missing from .env")


class Ledger:
    def __init__(self):
        self.lock = threading.Lock()
        self.rows: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "calls": 0})

    def add(self, label: str, usage: dict) -> None:
        with self.lock:
            row = self.rows[label]
            row["calls"] += 1
            row["tokens"] += int(usage.get("total_tokens") or 0)
            row["cost"] += float(usage.get("cost") or 0.0)

    def total(self, prefix: str) -> dict[str, float]:
        out = {"tokens": 0, "cost": 0.0, "calls": 0}
        with self.lock:
            for label, row in self.rows.items():
                if label.startswith(prefix):
                    for key in out:
                        out[key] += row[key]
        return out

    def tokens(self, label: str) -> int:
        with self.lock:
            return int(self.rows[label]["tokens"]) if label in self.rows else 0


LEDGER = Ledger()
KEY = ""


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        label = self.path.strip("/").split("/")[0]
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = body["model"]
        while model.startswith("openrouter/"):
            model = model[len("openrouter/"):]
        body["model"] = model
        body["usage"] = {"include": True}
        request = urllib.request.Request(
            UPSTREAM, data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=900)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        with response:
            self.send_response(200)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            usage: dict = {}
            if body.get("stream"):
                for line in response:
                    self.wfile.write(line)
                    self.wfile.flush()
                    if line.startswith(b"data:") and b'"usage"' in line:
                        try:
                            usage = json.loads(line[5:]).get("usage") or usage
                        except ValueError:
                            pass
            else:
                payload = response.read()
                self.wfile.write(payload)
                try:
                    usage = json.loads(payload).get("usage") or {}
                except ValueError:
                    pass
        LEDGER.add(label, usage)

    def log_message(self, *args):
        pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


def page_context(pdf: Path, paper_id: str) -> dict[str, str]:
    import pdf_inspector

    result = pdf_inspector.extract_pages_markdown(str(pdf))
    spans: dict[str, str] = {}
    for page in getattr(result, "pages", []) or []:
        markdown = str(getattr(page, "markdown", "") or "").strip()
        if markdown:
            spans[f"{paper_id}-span-{len(spans) + 1:04d}"] = markdown
    return spans


def run_system(system: str, paper: dict, pdf: Path, out_dir: Path, port: int, token_limit: int) -> dict | None:
    output = out_dir / f"{system}.json"
    if output.exists():
        return json.loads(output.read_text())
    label = f"{system}-{paper['paper_id']}"
    env = {**os.environ, "CLAIMS_INFERENCE_BASE_URL": f"http://127.0.0.1:{port}/{label}/v1",
           "CLAIMS_INFERENCE_TOKEN": "local", "PYTHONPATH": str(ROOT / "src")}
    log = out_dir / f"{system}.log"
    work = out_dir / f"{system}-work"
    work.mkdir(exist_ok=True)
    if system == "forge":
        request = {
            "schema_version": "claims_competition_request_v1",
            "paper": {"paper_id": paper["paper_id"], "title": paper["title"]},
            "pdf_path": str(pdf),
            "inference": {"model": MODEL, "allowed_models": [MODEL], "stream": True, "max_output_tokens": 8192,
                          "budget": {"token_limit": token_limit, "dollar_limit_usd": 0.1,
                                     "remaining_tokens": token_limit, "remaining_dollar_limit_usd": 0.1}},
        }
        path = work / "request.json"
        path.write_text(json.dumps(request))
        command = [sys.executable, "-m", "claims_forge.cli", "--request", str(path), "--output", str(output)]
    elif system in {"reference", "reference32"}:
        env["PYTHONPATH"] = str(CLAIMS_REPO)
        if system == "reference32":
            env["REF_MAX_TOKENS"] = "32768"
        command = [sys.executable, str(ROOT / "eval" / "run_reference.py"), str(pdf), paper["paper_id"], paper["title"], str(output)]
    else:
        raise ValueError(system)
    with log.open("wb") as handle:
        try:
            subprocess.run(command, env=env, cwd=work, stdout=handle, stderr=subprocess.STDOUT, timeout=1500, check=False)
        except subprocess.TimeoutExpired:
            return None
    tokens = LEDGER.tokens(label)
    if not output.exists():
        return None
    (out_dir / f"{system}.tokens").write_text(str(tokens))
    return json.loads(output.read_text())


def claims_of(artifact: dict | None) -> list[dict]:
    if not artifact:
        return []
    evidence = {row.get("evidence_id"): row for row in artifact.get("evidence", {}).get("records", [])}
    claims = sorted(artifact.get("logic", {}).get("claims", []), key=_rank)[:10]
    rows = []
    for claim in claims:
        spans = {sid for ref in claim.get("sources", []) for sid in ref.get("span_ids", [])}
        for eid in claim.get("evidence_ids", []):
            for ref in (evidence.get(eid) or {}).get("source_refs", []):
                spans.update(ref.get("span_ids", []))
        rows.append({"statement": claim["statement"], "qualifier": claim.get("conditions"), "spans": sorted(spans)})
    return rows


def _rank(claim: dict) -> int:
    value = str((claim.get("metadata") or {}).get("importance") or "").lower()
    return {"central": 0, "supporting": 1, "": 1, "minor": 2}.get(value, 1)


def chat(port: int, label: str, system: str, user: str, max_tokens: int = 8192) -> dict:
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": max_tokens, "stream": False,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    for attempt in range(4):
        request = urllib.request.Request(f"http://127.0.0.1:{port}/{label}/v1/chat/completions", data=body,
                                         method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                content = json.load(response)["choices"][0]["message"]["content"] or ""
            start, end = content.find("{"), content.rfind("}")
            return json.loads(content[start: end + 1])
        except Exception as exc:  # noqa: BLE001 - transient provider or parse failure.
            if attempt == 3:
                return {"error": type(exc).__name__}
            time.sleep(2 * (attempt + 1))
    return {"error": "retries"}


_JUDGE_LOCK = threading.Lock()


def cached(kind: str, key_data: object, compute):
    CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps([kind, key_data], sort_keys=True).encode()).hexdigest()
    path = CACHE / f"{digest}.json"
    if path.exists():
        return json.loads(path.read_text())
    value = compute()
    if isinstance(value, dict) and "error" in value:
        return value
    path.write_text(json.dumps(value))
    return value


JUDGE_SCHEMA = (
    '\n\nReturn ONLY JSON: {"atoms": [{"atom": "...", "supported": true, "cited_span_ids": []}], '
    '"gates": {"<gate name>": true}, "reviewed_span_ids": [], "selected": true}. '
    '"selected" is true only if every atom is supported and every gate passes.'
)


JUDGE_NOTES: dict[str, dict] = {}


def _why(assessment: dict) -> dict:
    if "error" in assessment:
        return {"error": assessment["error"]}
    gates = assessment.get("gates") or {}
    return {"selected": assessment.get("selected"),
            "failed_gates": [name for name, ok in gates.items() if ok is not True],
            "unsupported": [row.get("atom") for row in assessment.get("atoms") or []
                            if isinstance(row, dict) and row.get("supported") is not True][:3]}


def judge_claim(port: int, paper: dict, claim: dict, context: dict[str, str]) -> bool:
    spans = {sid: context[sid] for sid in claim["spans"] if sid in context}
    if not spans:
        JUDGE_NOTES[claim["statement"]] = {"error": "no matching spans"}
        return False
    task = {"paper": {"title": paper["title"]},
            "case": {"candidate_ref": "A", "statement": claim["statement"], "qualifier": claim["qualifier"],
                     "source_span_ids": list(spans)},
            "eligibility_gates": GATES, "source_spans": spans}
    user = json.dumps(task, ensure_ascii=False)

    def assess(role: str, extra: str = "") -> dict:
        system = (SKILLS / f"claims-silver-eligibility-selection-{role}" / "SKILL.md").read_text() + JUDGE_SCHEMA + extra
        return cached(f"judge:{role}", [system, user], lambda: chat(port, f"judge-{paper['paper_id']}", system, user))

    negative = assess("negative")
    positive = assess("positive")
    left, right = _selected(negative), _selected(positive)
    JUDGE_NOTES[claim["statement"]] = {"negative": _why(negative), "positive": _why(positive)}
    if left == right:
        return left
    tie_user = user + "\n\nPrimary assessments:\n" + json.dumps({"negative": negative, "positive": positive}, ensure_ascii=False)
    system = (SKILLS / "claims-silver-eligibility-selection-tiebreak" / "SKILL.md").read_text() + JUDGE_SCHEMA
    verdict = cached("judge:tiebreak", [system, tie_user], lambda: chat(port, f"judge-{paper['paper_id']}", system, tie_user))
    JUDGE_NOTES[claim["statement"]]["tiebreak"] = _why(verdict)
    return _selected(verdict)


def _selected(assessment: dict) -> bool:
    if "error" in assessment or assessment.get("selected") is not True:
        return False
    gates = assessment.get("gates")
    atoms = assessment.get("atoms")
    return (isinstance(gates, dict) and all(value is True for value in gates.values())
            and isinstance(atoms, list) and all(row.get("supported") is True for row in atoms if isinstance(row, dict)))


CLUSTER = """You merge accepted scientific claims from one paper into canonical units.
Two claims belong to the same unit only if they state the same finding, with no material difference in direction, population, conditions, or magnitude. Then rate each unit by the paper's own emphasis:
"central": a headline finding in the abstract or conclusion; "supporting": a genuine result that backs the headline findings; "minor": a narrow secondary result.
Return ONLY JSON: {"units": [{"importance": "central|supporting|minor", "members": ["claim ids"]}]}. Every claim id appears in exactly one unit."""


def build_units(port: int, paper: dict, accepted: dict[str, list[str]]) -> list[dict]:
    rows = [{"id": f"{system}:{index}", "statement": text}
            for system, claims in sorted(accepted.items()) for index, text in enumerate(claims)]
    if not rows:
        return []
    user = json.dumps({"paper_title": paper["title"], "claims": rows}, ensure_ascii=False)
    result = cached("cluster", [CLUSTER, user], lambda: chat(port, f"judge-{paper['paper_id']}", CLUSTER, user))
    known = {row["id"] for row in rows}
    units: list[dict] = []
    seen: set[str] = set()
    for unit in result.get("units", []) if isinstance(result, dict) else []:
        members = [member for member in unit.get("members", []) if member in known and member not in seen]
        if members:
            seen.update(members)
            units.append({"importance": unit.get("importance") if unit.get("importance") in WEIGHTS else "supporting", "members": members})
    units.extend({"importance": "supporting", "members": [row["id"]]} for row in rows if row["id"] not in seen)
    return units


def coverage(units: list[dict], system: str) -> float:
    if not units:
        return 0.0

    def total(rows: list[dict]) -> float:
        major = sum(WEIGHTS[row["importance"]] for row in rows if row["importance"] != "minor")
        minor = sum(WEIGHTS[row["importance"]] for row in rows if row["importance"] == "minor")
        return major, minor

    all_major, all_minor = total(units)
    covered_major, covered_minor = total([row for row in units if any(m.startswith(system + ":") for m in row["members"])])
    cap = (MINOR_CAP / (1 - MINOR_CAP)) * all_major
    denominator = all_major + min(all_minor, cap)
    return (covered_major + min(covered_minor, cap)) / denominator if denominator else 0.0


def evaluate(paper: dict, args, port: int, systems: list[str]) -> dict | None:
    pdf = DATA / "pdfs" / f"{paper['paper_id']}.pdf"
    out_dir = RESULTS / args.tag / paper["paper_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    context = page_context(pdf, paper["paper_id"])
    if not context:
        return None
    # The reference artifact is shared across tags so each version meets the same opponent.
    results: dict = {}
    accepted: dict[str, list[str]] = {}
    for system in systems:
        directory = out_dir
        if system.startswith("reference"):
            directory = RESULTS / f"_{system}" / paper["paper_id"]
            directory.mkdir(parents=True, exist_ok=True)
        artifact = run_system(system, paper, pdf, directory, port, args.token_limit)
        rows = claims_of(artifact)
        with ThreadPoolExecutor(max(1, len(rows))) as judges:
            verdicts = list(judges.map(lambda row: judge_claim(port, paper, row, context), rows))
        accepted[system] = [row["statement"] for row, ok in zip(rows, verdicts) if ok]
        token_file = directory / f"{system}.tokens"
        results[system] = {"claims": len(rows), "accepted": sum(verdicts), "ok": artifact is not None,
                           "tokens": int(token_file.read_text()) if token_file.exists() else 0,
                           "missing_span": sum(1 for row in rows if not any(s in context for s in row["spans"])),
                           "detail": [{"statement": row["statement"], "accepted": ok, "spans": row["spans"], "why": JUDGE_NOTES.get(row["statement"])}
                                      for row, ok in zip(rows, verdicts)]}
    units = build_units(port, paper, accepted)
    for system in systems:
        results[system]["score"] = coverage(units, system)
    (out_dir / "scored.json").write_text(json.dumps({"paper": paper["paper_id"], "results": results, "units": units}, indent=1, ensure_ascii=False))
    return {system: {key: value for key, value in row.items() if key != "detail"} for system, row in results.items()}


def main() -> None:
    global KEY
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--papers", type=int, default=12)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--systems", default="forge,reference")
    parser.add_argument("--token-limit", type=int, default=120_000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    KEY = api_key()
    systems = args.systems.split(",")
    papers = json.loads((DATA / "batch.json").read_text())["papers"][args.offset: args.offset + args.papers]
    server = Server(("127.0.0.1", 0), Proxy)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with ThreadPoolExecutor(args.workers) as pool:
        rows = [row for row in pool.map(lambda paper: _safe(paper, args, port, systems), papers) if row]
    report(rows, systems, args.tag)


def _safe(paper, args, port, systems):
    try:
        return evaluate(paper, args, port, systems)
    except Exception as exc:  # noqa: BLE001 - report and continue with other papers.
        print(f"FAILED {paper['paper_id']}: {type(exc).__name__}: {exc}", flush=True)
        return None


def report(rows: list[dict], systems: list[str], tag: str) -> None:
    print(f"\n=== {tag}: {len(rows)} papers")
    for system in systems:
        cells = [row[system] for row in rows]
        print(f"{system:10s} score {statistics.mean(c['score'] for c in cells):.3f}  "
              f"claims {statistics.mean(c['claims'] for c in cells):.1f}  "
              f"accepted {statistics.mean(c['accepted'] for c in cells):.1f}  "
              f"failed {sum(not c['ok'] for c in cells)}  "
              f"median tokens {statistics.median(c['tokens'] for c in cells):.0f}  "
              f"unmatched spans {sum(c['missing_span'] for c in cells)}")
    if len(systems) == 2:
        gains = [row[systems[0]]["score"] - row[systems[1]]["score"] for row in rows]
        error = statistics.stdev(gains) / len(gains) ** 0.5 if len(gains) > 1 else 0.0
        wins = sum(g > 0 for g in gains)
        print(f"gain {systems[0]} - {systems[1]}: {statistics.mean(gains):+.3f} (SE {error:.3f}; better on {wins}/{len(gains)})")
    for prefix in (*systems, "judge"):
        total = LEDGER.total(prefix + "-")
        print(f"spend {prefix:10s} ${total['cost']:.3f}  tokens {int(total['tokens'])}  calls {int(total['calls'])}")


if __name__ == "__main__":
    main()
