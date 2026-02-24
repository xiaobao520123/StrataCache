from .chain import FetchResult, TierChain
from .policy import LinkPolicy, StoreReason

__all__ = ["FetchResult", "TierChain", "LinkPolicy", "StoreReason"]

# Naming alias to better match "strata/layers" vocabulary (v0.1)
StrataChain = TierChain
