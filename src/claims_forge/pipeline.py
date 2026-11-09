"""Turn page spans into a ranked list of checked claims."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field

from . import prompts
from .gateway import BudgetExhausted, Client, GatewayError, estimate_tokens
from .source import Page, locate_quote, render_for_model

MAX_CLAIMS = 10
CANDIDATE_LIMIT = 18
EXTRACT_OUTPUT_TOKENS = 8192
GAP_OUTPUT_TOKENS = 4096
AUDIT_OUTPUT_TOKENS = 8192
MAX_CITED_PAGES = 6
MAX_SOURCE_CHARS = 170_000
MIN_SOURCE_CHARS = 12_000
_IMPORTANCE_RANK = {"central": 0, "supporting": 1, "minor": 2}
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from in into is it of on or that the their this to was were which with".split()
)
_FINDING_CUE = re.compile(
    r"\b(we (?:show|find|found|demonstrate|report|observed)|results? (?:show|suggest|indicate)|"
    r"significant|increased|decreased|associated with|reduced|improved|conclude)\b",
    re.IGNORECASE,
)


@dataclass
class Candidate:
    statement: str
    conditions: str = ""
    importance: str = "supporting"
    pages: list[int] = field(default_factory=list)
    quote: str = ""


def extract_claims(pages: list[Page], client: Client) -> list[Candidate]:
    source = _render_within_budget(pages, client)
    if not source:
        return fallback_claims(pages)
    candidates = _first_pass(source, client)
    if not candidates:
        return fallback_claims(pages)
    candidates = _gap_fill(source, candidates, client)
    return _select(_audit(source, candidates, client))


def _render_within_budget(pages: list[Page], client: Client) -> str:
    """Spend at most about 40% of the allowance on one view of the paper."""
    share = int(client.budget.spendable * 0.40 * 3.2)
    limit = min(MAX_SOURCE_CHARS, share)
    return render_for_model(pages, limit) if limit >= MIN_SOURCE_CHARS else ""


def _first_pass(source: str, client: Client) -> list[Candidate]:
    system = prompts.EXTRACT.replace("{limit}", str(CANDIDATE_LIMIT))
    try:
        payload = client.ask_json(system, source, output_tokens=EXTRACT_OUTPUT_TOKENS)
    except (GatewayError, BudgetExhausted) as exc:
        print(f"extraction call failed: {exc}", file=sys.stderr)
        return []
    return _parse_candidates(payload.get("claims"))


def _gap_fill(source: str, found: list[Candidate], client: Client) -> list[Candidate]:
    """Look once more for missed findings, only when the audit stays affordable."""
    listing = json.dumps([row.statement for row in found], ensure_ascii=False)
    user = f"EXISTING CLAIMS:\n{listing}\n\nPAPER:\n{source}"
    audit_cost = _audit_cost(source, found) + 800
    gap_cost = estimate_tokens(prompts.GAP_FILL) + estimate_tokens(user) + GAP_OUTPUT_TOKENS
    if client.budget.remaining < gap_cost + audit_cost:
        return found
    try:
        payload = client.ask_json(prompts.GAP_FILL, user, output_tokens=GAP_OUTPUT_TOKENS)
    except (GatewayError, BudgetExhausted) as exc:
        print(f"gap-fill call failed: {exc}", file=sys.stderr)
        return found
    return _distinct([*found, *_parse_candidates(payload.get("claims"))])


def _audit_cost(source: str, candidates: list[Candidate]) -> int:
    listing = sum(len(row.statement) + len(row.conditions) + 120 for row in candidates)
    return estimate_tokens(prompts.AUDIT) + estimate_tokens(source) + int(listing / 3.2) + AUDIT_OUTPUT_TOKENS


def _audit(source: str, candidates: list[Candidate], client: Client) -> list[Candidate]:
    """Review every candidate; on failure keep the unreviewed list."""
    listing = [
        {"id": f"c{index}", "statement": row.statement, "conditions": row.conditions,
         "importance": row.importance, "pages": row.pages, "quote": row.quote}
        for index, row in enumerate(candidates)
    ]
    user = f"CANDIDATES:\n{json.dumps(listing, ensure_ascii=False)}\n\nPAPER:\n{source}"
    try:
        payload = client.ask_json(prompts.AUDIT, user, output_tokens=AUDIT_OUTPUT_TOKENS)
    except (GatewayError, BudgetExhausted) as exc:
        print(f"audit call failed, keeping unreviewed claims: {exc}", file=sys.stderr)
        return candidates
    return apply_reviews(candidates, payload.get("reviews"))


def apply_reviews(candidates: list[Candidate], reviews: object) -> list[Candidate]:
    if not isinstance(reviews, list):
        return candidates
    by_id = {str(row.get("id")): row for row in reviews if isinstance(row, dict)}
    kept: list[Candidate] = []
    for index, original in enumerate(candidates):
        review = by_id.get(f"c{index}")
        if review is None:
            kept.append(original)  # An unreviewed claim costs nothing to submit.
            continue
        if str(review.get("verdict", "")).lower() == "drop" or review.get("duplicate_of"):
            continue
        statement = _text(review.get("statement")) or original.statement
        kept.append(Candidate(
            statement=statement,
            conditions=_text(review.get("conditions")) or original.conditions,
            importance=_importance(review.get("importance"), original.importance),
            pages=_pages(review.get("pages")) or original.pages,
            quote=_text(review.get("quote")) or original.quote,
        ))
    return kept or candidates


def _select(candidates: list[Candidate]) -> list[Candidate]:
    """Rank by importance, keeping the model's order within each tier."""
    ranked = sorted(_distinct(candidates), key=lambda row: _IMPORTANCE_RANK[row.importance])
    return ranked[:MAX_CLAIMS]


def _distinct(candidates: list[Candidate], threshold: float = 0.72) -> list[Candidate]:
    kept: list[Candidate] = []
    signatures: list[set[str]] = []
    for row in candidates:
        signature = _signature(row.statement)
        if signature and all(_overlap(signature, other) < threshold for other in signatures):
            kept.append(row)
            signatures.append(signature)
    return kept


def _signature(statement: str) -> set[str]:
    return {word for word in _WORD.findall(statement.casefold()) if word not in _STOPWORDS}


def _overlap(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def _parse_candidates(rows: object) -> list[Candidate]:
    if not isinstance(rows, list):
        return []
    parsed: list[Candidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        statement = _text(row.get("statement"))
        if len(statement) < 25:
            continue
        parsed.append(Candidate(
            statement=statement, conditions=_text(row.get("conditions")),
            importance=_importance(row.get("importance"), "supporting"),
            pages=_pages(row.get("pages")), quote=_text(row.get("quote")),
        ))
    return parsed


def _text(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _importance(value: object, default: str) -> str:
    text = _text(value).lower()
    return text if text in _IMPORTANCE_RANK else default


def _pages(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    pages: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in pages:
            pages.append(number)
    return pages[:MAX_CITED_PAGES]


def fallback_claims(pages: list[Page]) -> list[Candidate]:
    """Last resort without model output: quote the paper's own finding sentences."""
    found: list[Candidate] = []
    for page in pages[:4] + pages[-3:]:
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(page.body.split())):
            if 60 <= len(sentence) <= 400 and _FINDING_CUE.search(sentence):
                found.append(Candidate(statement=sentence, pages=[page.number], quote=sentence))
    distinct = _distinct(found)[:MAX_CLAIMS]
    if distinct:
        return distinct
    opening = " ".join(pages[0].body.split())[:300]
    return [Candidate(statement=opening, pages=[pages[0].number], quote=opening)]


def resolve_evidence(candidate: Candidate, pages: list[Page]) -> tuple[list[Page], str]:
    """Return the pages to cite and a quote that really appears on them."""
    by_number = {page.number: page for page in pages}
    cited = [by_number[number] for number in candidate.pages if number in by_number]
    located = locate_quote(pages, candidate.quote, candidate.pages) if candidate.quote else None
    if located is not None:
        page, quote = located
        if page not in cited:
            cited.insert(0, page)
        return _with_opening_page(cited, pages), quote
    if not cited:
        cited = [_best_page(candidate.statement, pages)]
    return _with_opening_page(cited, pages), _closest_sentence(candidate.statement, cited[0])


def _with_opening_page(cited: list[Page], pages: list[Page]) -> list[Page]:
    """Reviewers read only cited pages, so always include the abstract page."""
    return cited if pages[0] in cited else [*cited, pages[0]]


def _best_page(statement: str, pages: list[Page]) -> Page:
    signature = _signature(statement)
    return max(pages, key=lambda page: (len(signature & _signature(page.body)), -page.number))


def _closest_sentence(statement: str, page: Page) -> str:
    signature = _signature(statement)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", page.text) if 20 <= len(part.strip()) <= 500]
    if not sentences:
        return page.text[:300].strip()
    return max(sentences, key=lambda part: len(signature & _signature(part)))
