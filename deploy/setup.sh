#!/usr/bin/env bash
# Install the consensus miner next to this repository. Safe to run again.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLAIMS_COMMIT=7d5b68c

command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
cd "$ROOT"
[ -d claims-run/.git ] || git clone https://github.com/DeSciClaims/Claims.git claims-run
git -C claims-run checkout -q "$CLAIMS_COMMIT"
cd claims-run
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install -r "$ROOT/deploy/locked-requirements.txt"
uv pip install -e . --no-deps
[ -f "$ROOT/.env" ] || { cp "$ROOT/deploy/env.example" "$ROOT/.env"; chmod 600 "$ROOT/.env"; echo "Created .env: fill it in before starting."; }
echo "Setup complete. Edit .env, then run: deploy/run-miner.sh --claims.dry-run"
