class StrataCacheError(Exception):
    """Base error for stratacache."""


class ArtifactNotFound(StrataCacheError, KeyError):
    """Raised when an artifact id is missing from a backend."""


class BackendError(StrataCacheError):
    """Raised on backend I/O or invariant failures."""


class CodecError(StrataCacheError):
    """Raised when encoding/decoding an artifact record fails."""

