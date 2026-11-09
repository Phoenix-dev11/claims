"""Build the structured output document from checked claims."""

from __future__ import annotations

import re
from typing import Any

from .artifact_models import (
    Artifact, Claim, EvidenceLayer, EvidenceRecord, Logic, Paper, SourceRef, SrcLayer, TraceNode,
)
from .pipeline import Candidate, resolve_evidence
from .source import Page

_WORD = re.compile(r"[a-z0-9]+")
DEFAULT_CONDITIONS = "Within the study population, system and methods reported in this paper."
_PAPER_FIELDS = set(Paper.model_fields)


def build_artifact(paper: dict[str, Any], pages: list[Page], candidates: list[Candidate],
                   usage: dict[str, Any]) -> Artifact:
    claims: list[Claim] = []
    records: list[EvidenceRecord] = []
    children: list[TraceNode] = []
    for index, candidate in enumerate(candidates, start=1):
        cited, quote = resolve_evidence(candidate, pages)
        claim_id, evidence_id = f"C{index:02d}", f"EV{index:02d}"
        refs = _source_refs(cited, quote)
        conditions = grounded_conditions(candidate.conditions, cited)
        claims.append(Claim(
            claim_id=claim_id, statement=candidate.statement, conditions=conditions, status="supported",
            falsification_criteria=f"Data from a comparable analysis contradicting this finding under these conditions: {conditions}",
            evidence_ids=[evidence_id], sources=refs, metadata={"importance": candidate.importance},
        ))
        records.append(EvidenceRecord(
            evidence_id=evidence_id, title=f"Evidence for claim {index}", role="support", summary=quote,
            evidence_method="reported in the paper", source_refs=refs, linked_claim_ids=[claim_id],
        ))
        children.append(TraceNode(
            node_id=f"N{index:02d}", node_type="finding", support_level="explicit",
            summary=candidate.statement, source_refs=refs, evidence=[evidence_id],
        ))
    paper_model = Paper.model_validate(_paper_fields(paper))
    paper_model.claims_summary = [claim.statement for claim in claims]
    return Artifact(
        paper=paper_model,
        logic=Logic(claims=claims, key_insight=claims[0].statement),
        evidence=EvidenceLayer(records=records),
        trace=TraceNode(
            node_id="Q0", node_type="question", support_level="inferred",
            summary=paper_model.title or paper_model.paper_id, children=children,
        ),
        src=SrcLayer(environment=["claims-forge"]),
        metadata={"pipeline_name": "claims_forge", "output_schema": "agent_v1", "source_type": "pdf",
                  "page_spans": len(pages), **usage},
    )


def grounded_conditions(conditions: str, cited: list[Page]) -> str:
    """Keep stated conditions only if the cited pages actually use those words.

    Reviewers treat conditions as part of the claim and read only cited pages,
    so an unsupported word here can reject an otherwise correct claim.
    """
    text = " ".join(page.text for page in cited).casefold()
    words = [word for word in _WORD.findall(conditions.casefold()) if len(word) > 3]
    if words and all(word in text for word in words):
        return conditions
    return DEFAULT_CONDITIONS


def _source_refs(cited: list[Page], quote: str) -> list[SourceRef]:
    refs: list[SourceRef] = []
    for position, page in enumerate(cited):
        # The quote is only guaranteed to appear on the first cited page.
        refs.append(SourceRef(
            source_id=page.span_id, source_type="span", span_ids=[page.span_id],
            quote=quote if position == 0 else None, role="result",
        ))
    return refs


def _paper_fields(paper: dict[str, Any]) -> dict[str, Any]:
    fields = {key: value for key, value in paper.items() if key in _PAPER_FIELDS and value is not None}
    fields["paper_id"] = str(paper["paper_id"])
    fields["title"] = str(paper.get("title") or paper["paper_id"])
    for key in ("authors", "keywords"):
        value = fields.get(key)
        if value is not None and not isinstance(value, list):
            fields.pop(key)
        elif isinstance(value, list):
            fields[key] = [str(item) for item in value]
    if not isinstance(fields.get("year"), int) or isinstance(fields.get("year"), bool):
        fields.pop("year", None)
    for key in ("venue", "doi", "domain", "abstract"):
        if key in fields and not isinstance(fields[key], str):
            fields.pop(key)
    fields.pop("claims_summary", None)
    return fields
