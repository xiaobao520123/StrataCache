# StrataCache Design (v0.1)

## Goals

- **Standalone**: must not import `lmcache`.
- **Generalized objects**: stored items are **Artifacts**, not “KV cache”.
- **Tier coordination**: multiple backends managed as an ordered chain.
- **Write semantics per link**: each adjacent pair can be **write-through** or **write-back**.
- **Backends (minimal)**: CPU in-process store + CXL DAX-backed store.
- **Inference-engine friendly**: first-class integration target is **vLLM** (adapter layer).

Non-goals (v0.1):

- Cross-process correctness guarantees (e.g., crash-safe write-back journal)
- Full distributed remote backend
- Aggressive optimization (prefetch, batching, GPU DMA paths)

## Core abstractions

### Artifact

An **Artifact** is a generalized cached object:

- `ArtifactId`: stable identity (string), namespaced (engine/model/session/rank, etc.)
- `ArtifactType`: semantic category (KV blocks, tensor shard, MoE weights, memory entry…)
- `ArtifactMeta`: JSON-serializable metadata (shape/dtype/layout/ttl/pin/priority/engine_hints…)
- `payload`: bytes (opaque to core; adapters interpret)

The core library does not assume “KV layout” — KV is one possible `ArtifactType`.

### MemoryLayer (storage backend)

`MemoryLayer` is a minimal interface:

- `get(id) -> (payload, meta)` (raise on miss)
- `put(id, payload, meta)`
- `delete(id)` / `exists(id)`
- `stat()` / `capacity()` (best-effort for policy)

Layer implementations:

- `CpuMemoryLayer`: in-process dict + optional LRU capacity by bytes
- `CxlMemoryLayer`: uses `libcxl_shm.so` to store records on a DAX device

### TierChain (aka StrataChain) and LinkPolicy

`TierChain` is an ordered list of layers: `L0 -> L1 -> L2 -> ...`

Each link `Li -> L(i+1)` has a **LinkPolicy**:

- `WRITE_THROUGH`: a `store()` to `Li` also writes to lower tier synchronously
- `WRITE_BACK`: a `store()` to `Li` marks dirty and enqueues a background flush to lower tier

Read path:

- Search tiers from top to bottom.
- On hit in lower tier, optionally **promote** to upper tiers.
  - Promotions must **not** create dirty entries if the lower tier already holds the same bytes.

### Write-back manager

For write-back links, we need minimal machinery:

- `DirtyTracker`: which `ArtifactId` is dirty at which tier index
- `WritebackWorker`: background thread consuming a queue of flush tasks
- `flush(id)` / `flush_all()` to force persistence

v0.1 correctness model:

- Best-effort eventual persistence for write-back.
- No crash recovery guarantees (interfaces are shaped to add WAL later).

## Module layout

- `stratacache/core/`: Artifact types, encoding utilities, errors
- `stratacache/backend/`: MemoryLayer interface + CPU/CXL implementations
- `stratacache/tiering/`: TierChain coordinator + link policies
- `stratacache/writeback/`: dirty tracking + background write-back
- `stratacache/migration/`: stubs for migration planner/executor (v0.1 minimal)
- `stratacache/adapters/`: inference engine adapters (vLLM first)

## vLLM integration plan (sketch)

vLLM integration will be done via an adapter that translates vLLM’s internal cacheable units into `Artifact`s.

Key points to decide when implementing:

- Artifact granularity: block-level (paged KV blocks) vs session-level.
- Identity: include model hash + TP/PP rank + worker id to avoid collisions.
- Lifecycle hooks: allocation/free, swap-out/in, prefill vs decode.

