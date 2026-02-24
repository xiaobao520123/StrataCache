#!/usr/bin/env bash
set -euo pipefail

# Minimal launcher to run vLLM with StrataCache connector.
#
# Usage:
#   ./tools/run_vllm_with_stratacache.sh <model> [port] [-- <extra vllm args...>]
#
# Example:
#   STRATACACHE_CPU_CAP_MB=61440 STRATACACHE_CHUNK_SIZE=256 ./tools/run_vllm_with_stratacache.sh Qwen/Qwen2.5-7B-Instruct 8000
#
# Common env knobs (read directly by the connector via os.environ):
#   - STRATACACHE_CPU_CAP_MB        (default: 61440)  CPU tier capacity in MB
#   - STRATACACHE_CHUNK_SIZE        (default: 256)    chunk size (tokens)
#   - STRATACACHE_USE_CXL           (default: 0)      enable CXL tier
#   - STRATACACHE_CXL_DAX_DEVICE    (default: "")     required if STRATACACHE_USE_CXL=1
#   - STRATACACHE_WRITEBACK         (default: 0)      writeback policy toggle
#   - STRATACACHE_BUNDLE_LAYERS     (default: 1)      store one artifact per chunk (all layers bundled)
#   - STRATACACHE_LOG_STATS         (default: 1)      enable req_done stats line
#   - STRATACACHE_PROFILE           (default: 0)      dump connector hook timing summary at exit

MODEL="${1:-}"
PORT="8000"
if [[ -z "${MODEL}" ]]; then
  echo "Usage: $0 <model> [port]" >&2
  exit 2
fi

# consume model
shift

# optional port (second positional) unless it looks like an option
if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  PORT="${1}"
  shift
fi

# allow explicit separator for extra args
if [[ "${1:-}" == "--" ]]; then
  shift
fi

# Ensure `stratacache` is importable even if not installed editable.
STRATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${STRATA_DIR}/src"

KV_TRANSFER_CONFIG='{"kv_connector":"StrataCacheConnectorV1","kv_connector_module_path":"stratacache.adapters.vllm.connector_v1","kv_role":"kv_both"}'

vllm serve "${MODEL}" \
  --port "${PORT}" \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
  "$@"

