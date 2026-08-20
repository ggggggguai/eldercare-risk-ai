"""V3.3.3-r4 current-state optimisation infrastructure.

The package is intentionally separate from the sealed r3 implementation.  It
may read the frozen r3 scoring population as an adaptive-development boundary,
but never rewrites r2/r3 artifacts.
"""

from .contract import R4_PROTOCOL_VERSION

__all__ = ["R4_PROTOCOL_VERSION"]
