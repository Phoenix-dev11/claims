"""Budget-aware client for the competition's chat-completions gateway."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_TRANSIENT = {408, 409, 425, 429, 500, 502, 503, 504}
CHARS_PER_TOKEN = 3.2  # Deliberately conservative for numeric scientific text.
CALL_TIMEOUT_SECONDS = 420
SAFETY_MARGIN = 0.12


class GatewayError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class BudgetExhausted(RuntimeError):
    """A further model call could overrun the paper's token allowance."""


@dataclass
class Budget:
    token_limit: int
    deadline: float
    used_tokens: int = 0
    calls: int = 0

    @property
    def spendable(self) -> int:
        return int(self.token_limit * (1 - SAFETY_MARGIN))

    @property
    def remaining(self) -> int:
        return self.spendable - self.used_tokens

    def admits(self, prompt_tokens: int, output_tokens: int) -> bool:
        return (prompt_tokens + output_tokens <= self.remaining
                and time.monotonic() + 30 < self.deadline)


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 8


class Client:
    def __init__(self, *, model: str, stream: bool, max_output_tokens: int, budget: Budget):
        self.base = os.environ["CLAIMS_INFERENCE_BASE_URL"].rstrip("/")
        self.token = os.environ["CLAIMS_INFERENCE_TOKEN"]
        self.model = model
        self.stream = stream
        self.max_output_tokens = max_output_tokens
        self.budget = budget
        self.reasoning_off = True

    def ask_json(self, system: str, user: str, *, output_tokens: int, temperature: float = 0.0) -> dict[str, Any]:
        """Return one JSON object, with one repair attempt on malformed output."""
        output_tokens = min(output_tokens, self.max_output_tokens)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last: Exception | None = None
        for attempt in range(2):
            raw = self._complete(messages, output_tokens, temperature)
            try:
                return parse_json_object(raw)
            except ValueError as exc:
                last = exc
                print(f"model output was not valid JSON on attempt {attempt + 1}", file=sys.stderr)
                messages = [
                    *messages[:2],
                    {"role": "assistant", "content": raw[:4000]},
                    {"role": "user", "content": "That was not one complete JSON object. Return only the JSON object, shorter if needed."},
                ]
        raise GatewayError(f"no parseable model output: {last}")

    def _complete(self, messages: list[dict[str, str]], output_tokens: int, temperature: float) -> str:
        prompt_tokens = sum(estimate_tokens(row["content"]) for row in messages)
        if not self.budget.admits(prompt_tokens, output_tokens):
            raise BudgetExhausted("insufficient budget for another call")
        delay = 2.0
        for attempt in range(3):
            try:
                return self._send(messages, output_tokens, temperature, prompt_tokens)
            except GatewayError as exc:
                if not exc.retryable or attempt == 2 or not self.budget.admits(prompt_tokens, output_tokens):
                    raise
                print(f"transient gateway error, retrying: {exc}", file=sys.stderr)
                time.sleep(delay)
                delay *= 2
        raise GatewayError("gateway retries exhausted")

    def _send(self, messages: list[dict[str, str]], output_tokens: int, temperature: float, prompt_tokens: int) -> str:
        body: dict[str, Any] = {
            "model": "openrouter/" + self.model.removeprefix("openrouter/"),
            "messages": messages, "temperature": temperature,
            "max_tokens": output_tokens, "stream": self.stream,
        }
        if self.stream:
            body["stream_options"] = {"include_usage": True}
        if self.reasoning_off:
            body["reasoning"] = {"enabled": False}
        request = Request(
            self.base + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
            method="POST",
        )
        self.budget.calls += 1
        timeout = min(CALL_TIMEOUT_SECONDS, max(5.0, self.budget.deadline - time.monotonic()))
        try:
            with urlopen(request, timeout=timeout) as response:
                content, usage = _read_stream(response) if self.stream else _read_whole(response)
        except HTTPError as exc:
            if exc.code in {400, 422} and self.reasoning_off:
                # Some gateways reject the reasoning field; repeat without it.
                self.reasoning_off = False
                raise GatewayError(f"gateway HTTP {exc.code}", retryable=True) from None
            # Report status only: provider messages can echo request content.
            raise GatewayError(f"gateway HTTP {exc.code}", retryable=exc.code in _TRANSIENT) from None
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise GatewayError(f"gateway connection failed: {type(exc).__name__}", retryable=True) from None
        total = _total_tokens(usage)
        self.budget.used_tokens += total if total else prompt_tokens + estimate_tokens(content)
        return content


def _read_whole(response) -> tuple[str, dict[str, Any]]:
    payload = json.load(response)
    return str(payload["choices"][0]["message"].get("content") or ""), payload.get("usage") or {}


def _read_stream(response) -> tuple[str, dict[str, Any]]:
    chunks: list[str] = []
    usage: dict[str, Any] = {}
    for raw_line in response:
        line = raw_line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except ValueError:
            continue
        if isinstance(event, dict) and "error" in event:
            raise GatewayError("gateway stream error", retryable=True)
        usage = event.get("usage") or usage
        for choice in event.get("choices", []) or []:
            chunks.append(str((choice.get("delta") or {}).get("content") or ""))
    return "".join(chunks), usage


def _total_tokens(usage: dict[str, Any]) -> int:
    total = usage.get("total_tokens")
    if isinstance(total, int) and total > 0:
        return total
    parts = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
    return sum(value for value in parts if isinstance(value, int))


def parse_json_object(value: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE)
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = _parse_embedded(text)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def _parse_embedded(text: str) -> Any:
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start: end + 1])
        except ValueError:
            pass
    salvaged = _salvage_rows(text)
    if salvaged is None:
        raise ValueError("no JSON object found")
    return salvaged


def _salvage_rows(text: str) -> dict[str, list[dict[str, Any]]] | None:
    """Recover complete rows from output that was cut off mid-array."""
    match = re.search(r'"(claims|reviews)"\s*:\s*\[', text)
    if match is None:
        return None
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    index = match.end()
    while True:
        next_object = text.find("{", index)
        if next_object < 0:
            break
        try:
            row, index = decoder.raw_decode(text, next_object)
        except ValueError:
            break
        if isinstance(row, dict):
            rows.append(row)
    return {match.group(1): rows} if rows else None
