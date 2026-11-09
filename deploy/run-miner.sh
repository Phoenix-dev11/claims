#!/usr/bin/env bash
# Start the Claims consensus miner using the settings in .env.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a
source "$ROOT/.env"
source "$ROOT/deploy/miner.env"
set +a
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in .env}"
: "${WALLET_NAME:?Set WALLET_NAME in .env}"
: "${HOTKEY_NAME:?Set HOTKEY_NAME in .env}"
: "${EXTERNAL_IP:?Set EXTERNAL_IP in .env}"
AXON_PORT="${AXON_PORT:-8091}"
export SUBNET_CLAIMS_AGENT_CACHE_DIR="$ROOT/cache"

# Reviewed papers are cached on disk; remove copies older than 12 hours.
(while sleep 3600; do
  find "$ROOT/cache/consensus_sources" -type f -mmin +720 -delete 2>/dev/null || true
done) >/dev/null 2>&1 &
CLEANER=$!
trap 'kill "$CLEANER" 2>/dev/null || true' EXIT

cd "$ROOT/claims-run"
.venv/bin/python -m neurons.miner \
  --netuid 111 \
  --wallet.name "$WALLET_NAME" \
  --wallet.hotkey "$HOTKEY_NAME" \
  --subtensor.network finney \
  --axon.ip 0.0.0.0 \
  --axon.external_ip "$EXTERNAL_IP" \
  --axon.port "$AXON_PORT" \
  --axon.external_port "$AXON_PORT" \
  --claims.pipeline agent_v1 \
  --claims.network mainnet \
  --claims.output-dir "$ROOT/outputs/miner" \
  --logging.info \
  "$@"
