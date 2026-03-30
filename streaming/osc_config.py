"""
OSC transport configuration for the TRIBE v2 streaming stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class OSCConfig:
    """Configuration for OSC message routing.

    Attributes
    ----------
    unity_ip / unity_port : str / int
        Target for Unity brain visualisation (full vertex data + ROI + meta).
    sc_ip / sc_port : str / int
        Target for SuperCollider sonification (ROI + PCA + meta).
    pd_ip / pd_port : str / int
        Target for Pure Data sonification (alternate to SC).
    chunk_size : int
        Maximum number of float32 values per OSC message (UDP friendly).
    heartbeat_hz : float
        Rate of ``/tribe/heartbeat`` keep-alive messages.
    send_full_vertices : bool
        Whether to stream the complete ~20 k vertex array to Unity.
        Disable to save bandwidth and only stream ROI / PCA.
    enabled_targets : list[str]
        Which targets to actually send to (``"unity"``, ``"sc"``, ``"pd"``).
    """

    # -- Unity target --
    unity_ip: str = "127.0.0.1"
    unity_port: int = 9000

    # -- SuperCollider target --
    sc_ip: str = "127.0.0.1"
    sc_port: int = 57120

    # -- Pure Data target --
    pd_ip: str = "127.0.0.1"
    pd_port: int = 9001

    # -- Transport tuning --
    chunk_size: int = 5000
    heartbeat_hz: float = 10.0
    send_full_vertices: bool = True
    enabled_targets: List[str] = field(
        default_factory=lambda: ["unity", "sc"]
    )

    # -------------------------------------------------------------------
    # OSC address constants
    # -------------------------------------------------------------------
    # Full vertex data (chunked)
    ADDR_BRAIN_FULL_PREFIX: str = "/tribe/brain/full"
    # ROI averages
    ADDR_BRAIN_ROI: str = "/tribe/brain/roi"
    # PCA low-dimensional components
    ADDR_BRAIN_PCA: str = "/tribe/brain/pca"
    # Metadata (timestamp, sequence_id, latency_ms)
    ADDR_META: str = "/tribe/meta"
    # Keep-alive heartbeat
    ADDR_HEARTBEAT: str = "/tribe/heartbeat"
    # Brain state delta (for particle effects)
    ADDR_BRAIN_DELTA: str = "/tribe/brain/delta"
