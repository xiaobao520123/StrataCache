# StrataCache (standalone tiered cache for vLLM)

StrataCache is a **standalone** tiered cache library that can be used with **vLLM v0.13.0** via a KVConnector (similar to “vLLM + LMCache”), but implemented independently.

Key properties:

- **Generalized objects** (Artifacts): not hard-coded to “KV cache”
- **Layered coordination** (TierChain / MemoryLayer)
- **Per-link semantics**: write-through / write-back
- Backends (v0.1): **CPUMemoryLayer** + **CxlMemoryLayer**

This project does **not** import `lmcache`.

## 1) Setup (uv + venv)

From the `stratacache/` directory:

```bash
python -m pip install -U uv
uv venv -p 3.12 .venv
source .venv/bin/activate
```

Install StrataCache (editable) + deps:

```bash
uv pip install -e .
uv pip install "vllm==0.13.0"
```

Note: `vllm` pulls in `torch`. If you need a specific CUDA build of torch, follow vLLM’s install guidance and install the appropriate torch wheel first.

## 2) (Optional) Build CXL shared library

If you want to use `CxlMemoryLayer`, build the shared library from repo root:

```bash
cd ..
make -f Makefile.cxl_shm
cd stratacache
```

The binding loads `libcxl_shm.so` from common locations including the repo root. You can override with:

- `STRATACACHE_CXL_LIB=/abs/path/to/libcxl_shm.so`

## 3) Run vLLM with StrataCache connector

### Option A: Run via provided launcher script (recommended)

From the `stratacache/` directory:

```bash
STRATACACHE_USE_CXL=0 \
STRATACACHE_WRITEBACK=0 \
STRATACACHE_CPU_CAP_MB=512 \
STRATACACHE_CHUNK_SIZE=256 \
STRATACACHE_TENSOR_CODEC=stable \
STRATACACHE_LOG_STATS=1 \
STRATACACHE_LOG_EVERY=50 \
./tools/run_vllm_with_stratacache.sh sshleifer/tiny-gpt2 8000
```

If you have CXL DAX available:

```bash
STRATACACHE_USE_CXL=1 \
STRATACACHE_CXL_DAX_DEVICE=/dev/dax1.0 \
./tools/run_vllm_with_stratacache.sh Qwen/Qwen2.5-7B-Instruct 8000
```

### Option B: Run vLLM directly (manual)

Make sure `stratacache` is importable by vLLM workers:

- If you installed `-e ./stratacache`, nothing else is needed.
- Otherwise export `PYTHONPATH=$(pwd)/stratacache/src`.

Example:

```bash
vllm serve sshleifer/tiny-gpt2 \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"StrataCacheConnectorV1","kv_connector_module_path":"stratacache.adapters.vllm.connector_v1","kv_role":"kv_both","kv_connector_extra_config":{"stratacache.use_cxl":"0","stratacache.writeback":"0","stratacache.cpu_capacity_mb":"512"}}'
```

## Connector knobs

- **`STRATACACHE_CHUNK_SIZE`**: chunk size for prefix reuse (default: 256). Only full chunks are saved/loaded in v0.1.
- **`STRATACACHE_TENSOR_CODEC`**: `stable` (default) or `torchsave`. `stable` uses a versioned header + raw bytes (no pickle).
- **`STRATACACHE_LOG_STATS`**: `1/0` enable basic hit/miss + bytes counters.
- **`STRATACACHE_LOG_EVERY`**: log period (default: 50).

## 4) Send two requests (to see reuse)

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","prompt":"Hello!","max_tokens":16,"temperature":0.0}'
```

Send the same prompt again; the connector should report matched tokens and load KV from StrataCache.

## 5) Run tests (no pytest)

```bash
PYTHONPATH=/home/mh/LMCache/stratacache/src python /home/mh/LMCache/stratacache/tests/run_all.py
```

## Layout

- `docs/DESIGN.md`: architecture + responsibilities
- `docs/IMPLEMENTATION_LOG.md`: implementation notes (dirty/codec/flush semantics)
- `src/stratacache/`: library source
- `tests/`: assert-based tests (no pytest)
