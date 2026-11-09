"""Extract a structured claims document from one PDF."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .artifact_validator import validate_agent_artifact
from .assemble import build_artifact
from .gateway import Budget, Client
from .pipeline import Candidate, extract_claims, fallback_claims
from .source import read_pages

REQUEST_SCHEMA = "claims_competition_request_v1"
WALL_CLOCK_SECONDS = 1500


def run(request_path: Path, output_path: Path) -> None:
    started = time.monotonic()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError(f"expected schema {REQUEST_SCHEMA}")
    paper = dict(request["paper"])
    paper_id = str(paper["paper_id"])
    pdf_path = Path(request["pdf_path"])
    if not pdf_path.is_absolute():
        pdf_path = request_path.parent / pdf_path
    pages = read_pages(pdf_path, paper_id)

    inference = request["inference"]
    allowance = inference.get("budget") or {}
    token_limit = int(allowance.get("remaining_tokens") or allowance.get("token_limit") or 100_000)
    budget = Budget(token_limit=token_limit, deadline=started + WALL_CLOCK_SECONDS)
    client = Client(
        model=str(inference["model"]), stream=bool(inference.get("stream", True)),
        max_output_tokens=min(8192, int(inference.get("max_output_tokens", 8192))), budget=budget,
    )
    try:
        candidates = extract_claims(pages, client)
    except Exception as exc:  # Always return a valid document if the PDF was readable.
        print(f"pipeline error, using fallback: {type(exc).__name__}", file=sys.stderr)
        candidates = []
    candidates = candidates or fallback_claims(pages)

    usage = {"model_calls": budget.calls, "measured_tokens": budget.used_tokens,
             "elapsed_seconds": round(time.monotonic() - started, 1)}
    artifact = _valid_artifact(paper, pages, candidates, usage)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(artifact.logic.claims)} claims using {budget.calls} calls and "
          f"{budget.used_tokens} tokens of {token_limit}", file=sys.stderr)


def _valid_artifact(paper: dict, pages: list, candidates: list[Candidate], usage: dict):
    artifact = build_artifact(paper, pages, candidates, usage)
    problems = validate_agent_artifact(artifact)
    if problems:
        print("output failed validation, using fallback claims", file=sys.stderr)
        artifact = build_artifact(paper, pages, fallback_claims(pages), usage)
        problems = validate_agent_artifact(artifact)
        if problems:
            raise ValueError("; ".join(item["message"] for item in problems))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.request, args.output)


if __name__ == "__main__":
    main()
