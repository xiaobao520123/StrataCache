# StrataCache Implementation Log

This log records what has been implemented inside `stratacache/` and why, in a way that stays **independent from lmcache**.

## 2026-02-06

- Created standalone subproject directory `stratacache/`.
- Added initial architecture doc: `docs/DESIGN.md`.
- Added packaging skeleton: `pyproject.toml` + `src/` layout.
- Implemented core generalized object model:
  - `core/artifact.py` (`ArtifactId`, `ArtifactType`, `ArtifactMeta`)
  - `core/record_codec.py` (meta+payload record encoding)
  - `core/keycodec.py` (stable short keys for CXL name limit)
- Implemented backends:
  - `backend/cpu_store.py` (`CpuMemoryLayer`, in-process LRU-by-bytes store)
  - `backend/cxl/` (`ctypes` binding + `CxlMemoryLayer` storing encoded records)
- Implemented tier coordination + write semantics:
  - `tiering/chain.py` (`TierChain`: read-through, promotion, write-through/write-back)
  - `writeback/manager.py` (dirty tracking + background flush worker)
- Added developer smoke test:
  - `tools/smoke.py` (CPU-only by default; optional CXL tier)

- Added assert-based tests (no pytest):
  - `tests/run_all.py` + `tests/test_*.py` (includes write-back convergence + optional torch tests)
- Implemented vLLM v0.13.0 connector (KVConnector v1) for StrataCache:
  - `adapters/vllm/connector_v1.py` (`StrataCacheConnectorV1`)
  - Scheduler-side: prompt-hash + manifest lookup to compute matched tokens (conservative full-prompt reuse)
  - Worker-side: per-layer KV gather/scatter using `slot_mapping` and store/load through `TierChain`
- Added vLLM launcher script:
  - `tools/run_vllm_with_stratacache.sh` (expects to run from `stratacache/` dir; shifts positional args correctly)
- Updated `stratacache/README.md` with end-to-end workflow (uv venv -> install -> run vLLM -> curl test)

Next planned (v0.1):

- Add vLLM adapter implementation (target a specific vLLM version).
- Decide artifact granularity + id namespacing rules for engine integration.
- Add minimal unit tests for `TierChain` semantics.

## Notes on key semantics (v0.1)

### Dirty tracking

- A key is **dirty** at hop `Li -> L(i+1)` iff `Li` may contain a more recent value than `L(i+1)` for the same `ArtifactId`.
- Dirty is tracked per-hop (by upper-tier index), not globally.
- Dirty is cleared when a write-back **flush hop** successfully writes to the lower tier.
- If the artifact disappears from the upper tier (evicted/deleted) before flush, we treat it as **clean** for that hop (nothing to persist).

### Codec / record format

To keep the core generalized (not KV-specific), backends store a single record:

- `record = header + meta_json + payload`
- header format: magic `SC01` + uint32 `meta_len`
- `meta_json` is UTF-8 JSON for `ArtifactMeta`
- payload is opaque bytes (interpreted by adapters)

This makes CPU/CXL backends store the same logical unit without hardcoding “KV cache”.

### Write-back flush semantics

- `TierChain.flush(artifact_id)` is **best-effort** and flushes until stable, so multi-hop write-back chains (e.g. L0->L1 write-back, L1->L2 write-back) can propagate in one call.
- `TierChain.flush()` without an id drains a snapshot of dirty keys (bounded by `max_items`).
- The background write-back worker retries on failures with a small delay; v0.1 does **not** provide crash recovery guarantees.

## vLLM connector notes (vs LMCache)

This section summarizes intentional differences between StrataCache’s vLLM connector and LMCache’s implementation.

- StrataCache stores **generalized artifacts**. For vLLM KV we store **per-layer, per-chunk** gathered tensors as payload bytes (default codec: stable `SCT0` header + raw bytes).
- Matched-token logic in v0.1 is **chunked prefix-based**: scheduler walks per-prefix chunk manifests (`chunk_end`) to compute matched tokens (and subtracts 1 token on full-prompt hits for logits recomputation).
- We do not implement LMCache’s richer request tracking (chunk boundaries, partial chunks discard, decode-save policy, disaggregation specs, remote lookup client/server).
- We do not implement async layerwise pipelines yet (v0.1 is synchronous save/load on the connector hooks).
- Keying is currently **prefix-hash + chunk_end** scoped (plus model tag). This enables prefix reuse across different suffixes at chunk granularity.

Limitations / TODOs:

Completed (v0.1):
- Implemented **chunked/prefix reuse** for vLLM connector via per-prefix chunk manifests (scheduler computes matched tokens by walking chunk_end manifests).
- Implemented **stable, versioned tensor codec** (magic `SCT0` + JSON header + raw bytes) as the default payload format for KV chunks.
- Added **basic metrics/logging** knobs for connector hit/miss, matched tokens, and bytes stored/loaded.

Stats semantics (v0.1):
- StrataCache exposes **token-level hit rate** similar to LMCache’s observability:
  - `external_match_rate = external_matched_tokens_total / prompt_tokens_total` (scheduler-side; “how many could be loaded”)
  - `external_load_rate = external_loaded_tokens_total / prompt_tokens_total` (worker-side; “how many were actually loaded”, best-effort)
  - `gpu_hit_rate = gpu_hit_tokens_total / prompt_tokens_total` (vLLM local prefix cache hit tokens from `num_computed_tokens`)
- The connector also keeps a derived request-level notion:
  - **hit**: `matched_tokens > 0` for a scheduler lookup call (i.e., at least 1 token can be loaded beyond `num_computed_tokens`)
  - **miss**: `matched_tokens == 0` (including cases where some cached prefix exists but is not beyond `num_computed_tokens`)
- Per-layer accounting:
  - `bytes_loaded` is attributed to the tier where `TierChain.fetch()` hit (`FetchResult.hit_tier`)
  - `bytes_stored` is attributed to tier0, and additionally to consecutive write-through lower tiers (best-effort approximation in v0.1)
  - `tokens_loaded/tokens_stored` are attributed best-effort based on layer0 progress (per-chunk boundaries)

Tensor codec rationale:
- StrataCache’s storage contract is **bytes payload + metadata**. For vLLM KV we must serialize gathered tensors into bytes.
- `torch.save` is pickle-based (security/brittleness) and not ideal as a persistent or cross-version format.
- The `stable` codec (`SCT0` + JSON header + raw bytes) is deterministic, versioned, and avoids pickle while keeping dependencies minimal.
- LMCache has similar needs but solves them differently: it primarily stores structured KV objects via its engine/backends and uses custom (de)serialization for remote/backends (e.g., its serde modules), rather than using `torch.save` as the main on-disk/offload representation for local KV.

Follow-ups:
- Extend prefix reuse beyond full chunks (optional partial-chunk support).
- Improve performance (avoid repeated hashing, async IO, background flush batching).

- vLLM 0.13.0 custom connector loading: pass `kv_connector` as a class name and `kv_connector_module_path` as the module path (do not use `module:Class` in kv_connector).

External hit-rate debugging notes:
- If `External prefix cache hit rate` stays at 0, first check whether we are actually producing metadata and storing chunks:
  - Enable `STRATACACHE_DEBUG=1` and look for logs:
    - `alloc debug: ... blocks=...`
    - `meta debug: ... meta_requests=...`
    - `save entry debug: ... meta_requests=...`
- Common root causes:
  - Block id extraction failed (vLLM v1 uses `KVCacheBlocks.get_block_ids()`; connectors must call it).
  - Only storing full chunks while prompts are mostly shorter than chunk size; enable `STRATACACHE_SAVE_PARTIAL_CHUNKS=1`.
  - With `enable_chunked_prefill=True`, prompt KV is computed across multiple steps while the request appears in `scheduled_cached_reqs`. The connector must include cached requests in `build_connector_meta` and **only save KV that is actually computed so far** (otherwise you store garbage and thrash capacity).

## 2026-02-09 (performance profiling + save-path refactor)

STRATACACHE_PROFILE=1 STRATACACHE_CPU_CAP_MB=61440 STRATACACHE_CHUNK_SIZE=256   /home/mh/LMCache/stratacache/tools/run_vllm_with_stratacache.sh Qwen/Qwen2.5-7B-Instruct 8000
INFO connector_v1.py:133: StrataCache(vLLM) connector_profile(pid=3764274):
(EngineCore_DP0 pid=3764274)   - worker.save_kv_layer: count=9576 total_ms=359141.57 avg_ms=37.504 p50_ms=11.274 p95_ms=44.683
(EngineCore_DP0 pid=3764274)   - worker.save_kv_layer.bundle.encode_bundle: count=10959 total_ms=130813.88 avg_ms=11.937 p50_ms=11.368 p95_ms=17.854
(EngineCore_DP0 pid=3764274)   - scheduler.build_connector_meta: count=770 total_ms=47944.33 avg_ms=62.265 p50_ms=61.512 p95_ms=88.698
(EngineCore_DP0 pid=3764274)   - scheduler.get_num_new_matched_tokens: count=420 total_ms=3455.70 avg_ms=8.228 p50_ms=8.369 p95_ms=8.640
(EngineCore_DP0 pid=3764274)   - worker.start_load_kv: count=770 total_ms=426.70 avg_ms=0.554 p50_ms=0.005 p95_ms=6.403
(EngineCore_DP0 pid=3764274)   - worker.save_kv_layer.bundle.store: count=10959 total_ms=330.35 avg_ms=0.030 p50_ms=0.025 p95_ms=0.033
(EngineCore_DP0 pid=3764274)   - scheduler.request_finished: count=100 total_ms=13.54 avg_ms=0.135 p50_ms=0.156 p95_ms=0.185
total: 400s+

LMCache INFO: LMCache(vLLM) connector_profile(pid=3832339):
(EngineCore_DP0 pid=3832339)   - worker.wait_for_save: count=808 total_ms=85358.93 avg_ms=105.642 p50_ms=0.002 p95_ms=310.593
(EngineCore_DP0 pid=3832339)   - worker.start_load_kv: count=810 total_ms=2887.24 avg_ms=3.564 p50_ms=0.004 p95_ms=47.402
(EngineCore_DP0 pid=3832339)   - scheduler.get_num_new_matched_tokens: count=399 total_ms=229.07 avg_ms=0.574 p50_ms=0.082 p95_ms=2.661
(EngineCore_DP0 pid=3832339)   - scheduler.build_connector_meta: count=810 total_ms=209.32 avg_ms=0.258 p50_ms=0.059 p95_ms=0.680
(EngineCore_DP0 pid=3832339)   - scheduler.build_connector_meta.loop_cached: count=810 total_ms=126.21 avg_ms=0.156 p50_ms=0.039 p95_ms=0.420
(EngineCore_DP0 pid=3832339)   - scheduler.build_connector_meta.loop_new: count=810 total_ms=55.04 avg_ms=0.068 p50_ms=0.000 p95_ms=0.487
(EngineCore_DP0 pid=3832339)   - scheduler.build_connector_meta.cleanup_finished: count=810 total_ms=15.52 avg_ms=0.019 p50_ms=0.001 p95_ms=0.152
(EngineCore_DP0 pid=3832339)   - worker.wait_for_layer_load: count=9632 total_ms=5.24 avg_ms=0.001 p50_ms=0.001 p95_ms=0.001
(EngineCore_DP0 pid=3832339)   - worker.save_kv_layer: count=9632 total_ms=3.36 avg_ms=0.000 p50_ms=0.000 p95_ms=0.000
(EngineCore_DP0 pid=3832339)   - scheduler.request_finished: count=100 total_ms=0.13 avg_ms=0.001 p50_ms=0.001 p95_ms=0.002 (vllm_v1_adapter.py:146:lmcache.integration.vllm.vllm_v1_adapter)

Problem:
- With external hit rate correct, StrataCache end-to-end latency was still far worse than LMCache under identical settings.
- Profiling (`STRATACACHE_PROFILE=1`) showed the dominant cost was in the **worker-side save hook**:
  - `worker.save_kv_layer` consumed ~hundreds of seconds wall time.
  - The per-call median was ~10ms+ and it was invoked ~`num_layers × num_steps` times.
- Root cause: StrataCache did **heavy, synchronization-inducing work inside `save_kv_layer`**:
  - gather from paged KV (`index_select + contiguous`)
  - device→CPU transfers + tensor serialization to bytes (stable codec)
  - bundle packing (`encode_bundle`) for per-chunk aggregation
  - This blocks the attention forward path and destroys throughput even if hits are high.

Fix (LMCache-like strategy):
- `adapters/vllm/connector_v1.py` was updated so in bundle mode:
  - `save_kv_layer` becomes a **cheap enqueue** of per-layer KV tensor references for `(req_id, chunk_end)`.
  - The heavy gather/stack/encode/store work is moved to `wait_for_save` (called once per step at forward_context exit).
- Introduced a faster bundle format `bundleT`:
  - store **one tensor containing all layers for a chunk** (stacked on dim0)
  - encode/decode happens once per chunk (instead of once per layer + bundle packing)
  - artifacts use `:bundleT` ids; load/match logic prefers `bundleT` but falls back to old `:bundle` and `:layer=0`.

Expected impact:
- Greatly reduce time spent inside `save_kv_layer` (attention-forward hook), aligning execution shape with LMCache:
  - StrataCache’s heavy save work is now concentrated in `wait_for_save`, reducing per-layer synchronization overhead.
- Reduce serialization overhead by removing `encode_bundle` and per-layer encodes for bundled chunks (encode stacked tensor once).

INFO connector_v1.py:133: StrataCache(vLLM) connector_profile(pid=3794994):
(EngineCore_DP0 pid=3794994)   - worker.wait_for_save: count=768 total_ms=142493.20 avg_ms=185.538 p50_ms=0.003 p95_ms=906.782
(EngineCore_DP0 pid=3794994)   - worker.wait_for_save.bundleT.gather_stack: count=10959 total_ms=74266.48 avg_ms=6.777 p50_ms=3.355 p95_ms=119.730
(EngineCore_DP0 pid=3794994)   - worker.wait_for_save.bundleT.encode_tensor: count=10959 total_ms=67712.36 avg_ms=6.179 p50_ms=11.092 p95_ms=18.213
(EngineCore_DP0 pid=3794994)   - worker.save_kv_layer: count=9576 total_ms=52053.87 avg_ms=5.436 p50_ms=4.289 p95_ms=8.555
(EngineCore_DP0 pid=3794994)   - scheduler.build_connector_meta: count=770 total_ms=47760.63 avg_ms=62.027 p50_ms=61.481 p95_ms=87.201
(EngineCore_DP0 pid=3794994)   - scheduler.get_num_new_matched_tokens: count=420 total_ms=3433.72 avg_ms=8.176 p50_ms=8.310 p95_ms=8.515
(EngineCore_DP0 pid=3794994)   - worker.start_load_kv: count=770 total_ms=397.34 avg_ms=0.516 p50_ms=0.005 p95_ms=6.425
(EngineCore_DP0 pid=3794994)   - worker.wait_for_save.bundleT.store: count=10959 total_ms=181.28 avg_ms=0.017 p50_ms=0.015 p95_ms=0.021
(EngineCore_DP0 pid=3794994)   - scheduler.request_finished: count=100 total_ms=13.31 avg_ms=0.133 p50_ms=0.157 p95_ms=0.180
total: 267s

Follow-up (same investigation):
- Subsequent profiling showed two additional hotspots:
  - `worker.save_kv_layer` still incurred noticeable overhead due to per-layer recomputation of chunk boundaries and prefix hashes.
    - Fix: in bundle mode, only **layer0** computes `chunk_ends` + prefix hashes and creates pending entries; other layers only attach KV references using the per-request `chunk_ends` computed by layer0.
  - `scheduler.build_connector_meta` was unexpectedly expensive due to rebuilding `slot_mapping` tensors every scheduler step.
    - Fix: cache `slot_mapping` by request id and **extend incrementally** when new blocks are appended, instead of rebuilding from scratch.

INFO connector_v1.py:133: StrataCache(vLLM) connector_profile(pid=3802862):
(EngineCore_DP0 pid=3802862)   - worker.wait_for_save: count=768 total_ms=188771.10 avg_ms=245.796 p50_ms=0.003 p95_ms=1056.148
(EngineCore_DP0 pid=3802862)   - worker.wait_for_save.bundleT.gather_stack: count=10959 total_ms=120368.62 avg_ms=10.984 p50_ms=3.350 p95_ms=236.402
(EngineCore_DP0 pid=3802862)   - worker.wait_for_save.bundleT.encode_tensor: count=10959 total_ms=67877.51 avg_ms=6.194 p50_ms=11.117 p95_ms=18.243
(EngineCore_DP0 pid=3802862)   - scheduler.build_connector_meta: count=770 total_ms=46539.29 avg_ms=60.441 p50_ms=60.362 p95_ms=85.693
(EngineCore_DP0 pid=3802862)   - scheduler.get_num_new_matched_tokens: count=420 total_ms=3436.48 avg_ms=8.182 p50_ms=8.352 p95_ms=8.596
(EngineCore_DP0 pid=3802862)   - worker.save_kv_layer: count=9576 total_ms=2326.12 avg_ms=0.243 p50_ms=0.018 p95_ms=0.061
(EngineCore_DP0 pid=3802862)   - worker.start_load_kv: count=770 total_ms=407.80 avg_ms=0.530 p50_ms=0.005 p95_ms=6.399
(EngineCore_DP0 pid=3802862)   - worker.wait_for_save.bundleT.store: count=10959 total_ms=188.23 avg_ms=0.017 p50_ms=0.015 p95_ms=0.024
(EngineCore_DP0 pid=3802862)   - scheduler.request_finished: count=100 total_ms=12.71 avg_ms=0.127 p50_ms=0.145 p95_ms=0.188
total: 261s

- Even after moving heavy work to `wait_for_save`, the dominant remaining cost was `bundleT.gather_stack`, which originally used `index_select` for every chunk/layer.
  - Fix: add a fast path in `_gather_by_slots` / `_scatter_by_slots` to detect the common prefill case where `slot_mapping` is a contiguous increasing range (no `-1` masked slots) and use simple slicing instead of `index_select`.
  - This reduces GPU-side gather/scatter overhead for prompt prefill, improving end-to-end latency while keeping correctness for masked/non-contiguous mappings.

INFO connector_v1.py:133: StrataCache(vLLM) connector_profile(pid=3808946):
(EngineCore_DP0 pid=3808946)   - worker.wait_for_save: count=768 total_ms=178481.19 avg_ms=232.397 p50_ms=0.003 p95_ms=862.815
(EngineCore_DP0 pid=3808946)   - worker.wait_for_save.bundleT.encode_tensor: count=10959 total_ms=96526.41 avg_ms=8.808 p50_ms=11.051 p95_ms=150.869
(EngineCore_DP0 pid=3808946)   - worker.wait_for_save.bundleT.gather_stack: count=10959 total_ms=81435.53 avg_ms=7.431 p50_ms=1.745 p95_ms=4.463
(EngineCore_DP0 pid=3808946)   - scheduler.build_connector_meta: count=770 total_ms=46493.19 avg_ms=60.381 p50_ms=60.715 p95_ms=85.570
(EngineCore_DP0 pid=3808946)   - scheduler.get_num_new_matched_tokens: count=420 total_ms=3440.60 avg_ms=8.192 p50_ms=8.317 p95_ms=8.618
(EngineCore_DP0 pid=3808946)   - worker.save_kv_layer: count=9576 total_ms=2406.56 avg_ms=0.251 p50_ms=0.019 p95_ms=0.049
(EngineCore_DP0 pid=3808946)   - worker.start_load_kv: count=770 total_ms=399.14 avg_ms=0.518 p50_ms=0.005 p95_ms=6.412
(EngineCore_DP0 pid=3808946)   - worker.wait_for_save.bundleT.store: count=10959 total_ms=188.91 avg_ms=0.017 p50_ms=0.016 p95_ms=0.024
(EngineCore_DP0 pid=3808946)   - scheduler.request_finished: count=100 total_ms=12.85 avg_ms=0.129 p50_ms=0.147 p95_ms=0.190
total: 251s

- Further follow-up for long-prompt repeated-request workloads:
  - Observed that `wait_for_save.bundleT.gather_stack` + `wait_for_save.bundleT.encode_tensor` still dominated total runtime even after hook reshaping.
  - Root cause: StrataCache still re-gathered/re-encoded chunks that were already present in external cache from earlier requests.
  - Fix: add cross-request dedup in `wait_for_save`:
    - before gather/encode/store, check existence of `bundleT` (and fallback old `bundle`/`layer0`) for `(prefix_hash, chunk_end)`;
    - if exists, skip heavy save path and only advance per-request save watermark.

[2026-02-10 04:37:53] INFO connector_v1.py:133: StrataCache(vLLM) connector_profile(pid=3821887):
(EngineCore_DP0 pid=3821887)   - worker.wait_for_save: count=768 total_ms=119984.79 avg_ms=156.230 p50_ms=0.003 p95_ms=416.482
(EngineCore_DP0 pid=3821887)   - worker.wait_for_save.bundleT.encode_tensor: count=2808 total_ms=62819.53 avg_ms=22.372 p50_ms=11.762 p95_ms=217.356
(EngineCore_DP0 pid=3821887)   - worker.wait_for_save.bundleT.gather_stack: count=2808 total_ms=56953.50 avg_ms=20.283 p50_ms=1.744 p95_ms=124.622
(EngineCore_DP0 pid=3821887)   - scheduler.build_connector_meta: count=770 total_ms=46417.03 avg_ms=60.282 p50_ms=61.895 p95_ms=84.920
(EngineCore_DP0 pid=3821887)   - scheduler.get_num_new_matched_tokens: count=420 total_ms=3432.49 avg_ms=8.173 p50_ms=8.311 p95_ms=8.489
(EngineCore_DP0 pid=3821887)   - worker.save_kv_layer: count=9576 total_ms=2365.26 avg_ms=0.247 p50_ms=0.019 p95_ms=0.050
(EngineCore_DP0 pid=3821887)   - worker.start_load_kv: count=770 total_ms=397.18 avg_ms=0.516 p50_ms=0.004 p95_ms=6.406
(EngineCore_DP0 pid=3821887)   - worker.wait_for_save.bundleT.store: count=2808 total_ms=47.17 avg_ms=0.017 p50_ms=0.015 p95_ms=0.024
(EngineCore_DP0 pid=3821887)   - scheduler.request_finished: count=100 total_ms=13.85 avg_ms=0.138 p50_ms=0.161 p95_ms=0.191
total: 192s

- Additional scheduler-path alignment with LMCache (this step):
  - Removed redundant full-prompt hashing in `build_connector_meta` (`prompt_hash` field is not used by worker hot path).
  - Reduced cached-request merge overhead when there are no new block ids.
  - Added scheduler-side per-request cleanup (`_prompt_tokens`, `_alloc_blocks`, `_num_external_tokens`) in `request_finished`.
  - Result: `scheduler.build_connector_meta` dropped from ~`60 ms/call` to ~`1.07 ms/call` (`46417 ms -> 820 ms` total), and end-to-end latency improved from ~`192s` to ~`146s`.

INFO connector_v1.py:133: StrataCache(vLLM) connector_profile(pid=3827006):
(EngineCore_DP0 pid=3827006)   - worker.wait_for_save: count=768 total_ms=120200.66 avg_ms=156.511 p50_ms=0.002 p95_ms=419.889
(EngineCore_DP0 pid=3827006)   - worker.wait_for_save.bundleT.encode_tensor: count=2808 total_ms=62579.20 avg_ms=22.286 p50_ms=11.642 p95_ms=218.168
(EngineCore_DP0 pid=3827006)   - worker.wait_for_save.bundleT.gather_stack: count=2808 total_ms=57412.13 avg_ms=20.446 p50_ms=1.758 p95_ms=124.888
(EngineCore_DP0 pid=3827006)   - scheduler.get_num_new_matched_tokens: count=418 total_ms=3428.59 avg_ms=8.202 p50_ms=8.287 p95_ms=8.531
(EngineCore_DP0 pid=3827006)   - worker.save_kv_layer: count=9576 total_ms=2326.62 avg_ms=0.243 p50_ms=0.019 p95_ms=0.050
(EngineCore_DP0 pid=3827006)   - scheduler.build_connector_meta: count=770 total_ms=820.22 avg_ms=1.065 p50_ms=1.047 p95_ms=2.122
(EngineCore_DP0 pid=3827006)   - scheduler.build_connector_meta.loop_cached: count=770 total_ms=779.30 avg_ms=1.012 p50_ms=1.011 p95_ms=1.971
(EngineCore_DP0 pid=3827006)   - worker.start_load_kv: count=770 total_ms=393.37 avg_ms=0.511 p50_ms=0.004 p95_ms=6.378
(EngineCore_DP0 pid=3827006)   - scheduler.build_connector_meta.add_req.total: count=9762 total_ms=376.44 avg_ms=0.039 p50_ms=0.022 p95_ms=0.282
(EngineCore_DP0 pid=3827006)   - scheduler.build_connector_meta.add_req.slot_mapping: count=9762 total_ms=208.37 avg_ms=0.021 p50_ms=0.007 p95_ms=0.250
(EngineCore_DP0 pid=3827006)   - scheduler.build_connector_meta.add_req.append_meta: count=9762 total_ms=54.76 avg_ms=0.006 p50_ms=0.004 p95_ms=0.018
(EngineCore_DP0 pid=3827006)   - scheduler.build_connector_meta.add_req.compute_fields: count=9762 total_ms=49.22 avg_ms=0.005 p50_ms=0.005 p95_ms=0.010
(EngineCore_DP0 pid=3827006)   - worker.wait_for_save.bundleT.store: count=2808 total_ms=46.34 avg_ms=0.017 p50_ms=0.015 p95_ms=0.022
(EngineCore_DP0 pid=3827006)   - scheduler.build_connector_meta.loop_new: count=770 total_ms=30.92 avg_ms=0.040 p50_ms=0.000 p95_ms=0.323
(EngineCore_DP0 pid=3827006)   - scheduler.request_finished: count=100 total_ms=13.34 avg_ms=0.133 p50_ms=0.155 p95_ms=0.196
total: 146.6s

INFO connector_v1.py:133: StrataCache(vLLM) connector_profile(...):
total: 143s (same workload, no significant delta from 146.6s)

- Remaining main latency gap vs LMCache (same workload profile):
  - `worker.wait_for_save`: Strata `120200 ms` vs LMCache `85359 ms` (largest remaining gap).
  - `scheduler.request_finished`: Strata `13.34 ms/call` vs LMCache `0.001 ms/call` (cleanup/log path overhead).
  - `scheduler.get_num_new_matched_tokens`: Strata `3429 ms` vs LMCache `229 ms` (chunk-existence probe path still heavier).

- Why StrataCache still "encodes tensor" while LMCache appears to store tensors directly:
  - StrataCache v0.1 storage contract is `payload: bytes` + `ArtifactMeta` across all tiers (`CpuMemoryLayer`, `CxlMemoryLayer`, `TierChain`); this keeps backends generic and identical for CPU/CXL.
  - LMCache's engine/backends are KV-specialized and can keep tensor-like structures deeper in its pipeline; StrataCache currently normalizes to bytes at connector boundary.
  - Therefore, "no-encode at all" is not possible without redesigning tier/backends API from bytes to typed tensor blobs.

- Encode-path improvement (this step):
  - Kept bytes-based backend contract, but removed redundant in-band tensor header for stable codec by default.
  - New default path (`STRATACACHE_TENSOR_HEADER_IN_PAYLOAD=0`):
    - payload stores raw tensor bytes only;
    - dtype/shape/codec move to `ArtifactMeta.attrs` (`tensor_codec=stable_raw`, `tensor_dtype`, `tensor_shape`);
    - decode path uses metadata attrs and remains backward-compatible with old payload-header format.
  - This reduces per-chunk JSON header encode/decode overhead and payload size while preserving backend compatibility.