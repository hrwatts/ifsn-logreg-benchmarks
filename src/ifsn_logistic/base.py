"""Base constants and type aliases used across the package."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


ArrayLike = np.ndarray
DiagnosticsDict = Dict[str, Any]

PROB_CLIP_EPS: float = 1e-15
"""Epsilon for clipping probabilities away from 0 and 1 in log-loss."""

DEFAULT_ALPHA: float = 1e-4
"""Fallback regularization if the user passes an effectively infinite ``C``."""

SM_DENOM_FLOOR: float = 1e-12
"""Floor for the Sherman-Morrison denominator to avoid division by zero."""
