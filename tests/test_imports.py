from __future__ import annotations


def run() -> None:
    import stratacache
    import stratacache.backend
    import stratacache.core
    import stratacache.tiering
    import stratacache.writeback

    # vLLM adapter should be safe to import even if vllm isn't installed.
    import stratacache.adapters.vllm  # noqa: F401

    assert isinstance(stratacache.__version__, str)

