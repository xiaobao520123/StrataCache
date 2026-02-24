# vLLM adapter (planned)

This directory will contain the vLLM integration layer.

v0.1 only defines the direction:

- Translate vLLM cacheable units into `stratacache.core.Artifact`.
- Generate stable `ArtifactId` that includes model + rank + worker namespaces.
- Use `TierChain.fetch()` / `TierChain.store()` to implement swap-in/out or reuse.

Implementation will depend on the vLLM version we target (internal APIs differ).
