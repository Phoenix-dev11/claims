"""Read a PDF into page-level spans.

Each non-empty page becomes one span, numbered in page order. Reviewers read a
paper one page at a time, so a claim that cites a page lets them read the text
that supports it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

_REFERENCE_HEADING = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography|literature cited|works cited)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SPACE = re.compile(r"[ \t ]+")
_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")


@dataclass(frozen=True)
class Page:
    span_id: str
    number: int
    text: str
    reference_start: int | None = None

    @property
    def body(self) -> str:
        """Page text without the trailing bibliography."""
        return self.text if self.reference_start is None else self.text[: self.reference_start].rstrip()


def clean_text(raw: str) -> str:
    text = raw.replace("\r", "\n").replace("­", "")
    text = _HYPHEN_BREAK.sub("", text)
    lines = [_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def read_pages(pdf_path: Path, paper_id: str) -> list[Page]:
    pages: list[Page] = []
    for number, pdf_page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
        try:
            text = clean_text(pdf_page.extract_text() or "")
        except Exception:  # A single malformed page must not lose the paper.
            text = ""
        if text:
            pages.append(Page(f"{paper_id}-span-{len(pages) + 1:04d}", number, text))
    if not pages:
        raise ValueError("PDF has no extractable text")
    return _mark_references(pages)


def _mark_references(pages: list[Page]) -> list[Page]:
    """Find where the bibliography begins, ignoring early false matches."""
    earliest_page = len(pages) // 2
    for index in range(earliest_page, len(pages)):
        match = _REFERENCE_HEADING.search(pages[index].text)
        if match is None:
            continue
        marked = list(pages[:index])
        page = pages[index]
        marked.append(Page(page.span_id, page.number, page.text, match.start()))
        marked.extend(Page(p.span_id, p.number, p.text, 0) for p in pages[index + 1:])
        return marked
    return pages


def render_for_model(pages: list[Page], max_chars: int) -> str:
    """Render page-labelled text, shortening long papers from the middle."""
    bodies = [(page.number, page.body) for page in pages if page.body]
    total = sum(len(body) for _, body in bodies)
    if total > max_chars:
        bodies = _trim(bodies, max_chars)
    return "\n\n".join(f"[PAGE {number}]\n{body}" for number, body in bodies)


def _trim(bodies: list[tuple[int, str]], max_chars: int) -> list[tuple[int, str]]:
    """Keep the opening and closing pages, where findings are summarised."""
    kept: dict[int, tuple[int, str]] = {}
    remaining = max_chars
    left, right = 0, len(bodies) - 1
    take_left = True
    while left <= right and remaining > 0:
        index = left if take_left else right
        number, body = bodies[index]
        kept[index] = (number, body[:remaining])
        remaining -= len(kept[index][1])
        if take_left:
            left += 1
        else:
            right -= 1
        take_left = not take_left
    return [kept[index] for index in sorted(kept)]


def locate_quote(pages: list[Page], quote: str, preferred: list[int]) -> tuple[Page, str] | None:
    """Find a quote on a preferred page first, then anywhere in the paper."""
    target = _normalise(quote)
    if len(target) < 12:
        return None
    by_number = {page.number: page for page in pages}
    ordered = [by_number[number] for number in preferred if number in by_number]
    ordered += [page for page in pages if page not in ordered]
    for page in ordered:
        exact = _exact_span(page.text, target)
        if exact is not None:
            return page, exact
    return None


def _normalise(value: str) -> str:
    return " ".join(value.split()).casefold()


def _exact_span(text: str, normalised_target: str) -> str | None:
    """Map a whitespace-insensitive match back to the page's own characters."""
    characters: list[str] = []
    positions: list[int] = []
    previous_space = True
    for index, character in enumerate(text):
        if character.isspace():
            if previous_space:
                continue
            characters.append(" ")
            positions.append(index)
            previous_space = True
        else:
            characters.append(character.casefold()[0])
            positions.append(index)
            previous_space = False
    start = "".join(characters).find(normalised_target)
    if start < 0:
        return None
    return text[positions[start]: positions[start + len(normalised_target) - 1] + 1]
