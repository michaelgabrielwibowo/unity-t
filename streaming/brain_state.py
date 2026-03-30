"""
BrainState — canonical output of the streaming inference engine.

Each prediction step produces one BrainState containing the full
fsaverage5 vertex vector, pre-computed ROI averages, and low-dimensional
PCA components suitable for sonification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FSAVERAGE5_VERTICES_PER_HEMI = 10_242
FSAVERAGE5_TOTAL_VERTICES = 2 * FSAVERAGE5_VERTICES_PER_HEMI  # 20 484
NUM_PCA_COMPONENTS = 8
NUM_ROI_REGIONS = 75  # Destrieux atlas approximate count


@dataclass(slots=True)
class BrainState:
    """Immutable snapshot of a predicted brain state.

    Attributes
    ----------
    timestamp : float
        Wall-clock time (``time.time()``) when this prediction was produced.
    vertices : np.ndarray
        Shape ``(20484,)`` float32 — full fsaverage5 cortical surface
        (left hemisphere first, then right hemisphere).
    roi_averages : dict[str, float]
        Pre-computed mean activation per Destrieux atlas region.
    pca_components : np.ndarray
        Shape ``(8,)`` float32 — first 8 PCA components of the vertex
        vector, pre-computed from a reference PCA basis.
    sequence_id : int
        Monotonically increasing counter (0-based) for ordering.
    latency_ms : float
        End-to-end inference latency in milliseconds for this step.
    """

    timestamp: float
    vertices: np.ndarray  # (20484,) float32
    roi_averages: Dict[str, float] = field(default_factory=dict)
    pca_components: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_PCA_COMPONENTS, dtype=np.float32)
    )
    sequence_id: int = 0
    latency_ms: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def left_hemisphere(self) -> np.ndarray:
        """Return the left-hemisphere vertices ``(10242,)``."""
        return self.vertices[:FSAVERAGE5_VERTICES_PER_HEMI]

    @property
    def right_hemisphere(self) -> np.ndarray:
        """Return the right-hemisphere vertices ``(10242,)``."""
        return self.vertices[FSAVERAGE5_VERTICES_PER_HEMI:]

    @property
    def global_mean(self) -> float:
        """Mean activation across the whole cortical surface."""
        return float(np.mean(self.vertices))

    @property
    def global_std(self) -> float:
        """Standard deviation of activation across the cortical surface."""
        return float(np.std(self.vertices))

    def normalized(self, vmin: float = -3.0, vmax: float = 3.0) -> np.ndarray:
        """Return vertices clipped to *[vmin, vmax]* then scaled to *[0, 1]*."""
        clipped = np.clip(self.vertices, vmin, vmax)
        return (clipped - vmin) / (vmax - vmin)

    def to_chunks(self, chunk_size: int = 5000) -> list[np.ndarray]:
        """Split vertices into OSC-friendly chunks."""
        return [
            self.vertices[i : i + chunk_size]
            for i in range(0, len(self.vertices), chunk_size)
        ]

    @staticmethod
    def empty() -> "BrainState":
        """Return a zero-initialized BrainState (useful as a sentinel)."""
        return BrainState(
            timestamp=time.time(),
            vertices=np.zeros(FSAVERAGE5_TOTAL_VERTICES, dtype=np.float32),
        )


@dataclass
class BrainStateDelta:
    """Optional: difference between two consecutive brain states.

    Useful for detecting which regions changed most between time steps,
    and for driving particle effects in Unity.
    """

    delta_vertices: np.ndarray  # (20484,) signed float32
    max_increase_idx: int
    max_decrease_idx: int
    mean_abs_change: float

    @classmethod
    def from_states(cls, prev: BrainState, curr: BrainState) -> "BrainStateDelta":
        delta = curr.vertices - prev.vertices
        return cls(
            delta_vertices=delta,
            max_increase_idx=int(np.argmax(delta)),
            max_decrease_idx=int(np.argmin(delta)),
            mean_abs_change=float(np.mean(np.abs(delta))),
        )
