"""
OSC Multi-Address Publisher
============================
Distributes ``BrainState`` objects to Unity and SuperCollider via
Open Sound Control (OSC) over UDP.

Full vertex data is chunked to stay within UDP-safe message sizes.
ROI and PCA data are sent as compact float arrays.
A heartbeat message is sent at a higher rate for keep-alive.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from .brain_state import BrainState, BrainStateDelta
from .osc_config import OSCConfig

logger = logging.getLogger(__name__)


class BrainStatePublisher:
    """Publishes BrainState to OSC targets (Unity, SuperCollider, Pure Data).

    Usage
    -----
    ::

        config = OSCConfig()
        publisher = BrainStatePublisher(config)
        publisher.start()   # starts heartbeat thread

        # In your streaming loop:
        publisher.publish(brain_state)

        publisher.stop()

    Parameters
    ----------
    config : OSCConfig
        OSC routing and transport configuration.
    """

    def __init__(self, config: Optional[OSCConfig] = None):
        self.config = config or OSCConfig()
        self._clients = {}
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False
        self._init_clients()

    def _init_clients(self):
        """Initialize python-osc UDP clients for each enabled target."""
        try:
            from pythonosc.udp_client import SimpleUDPClient
        except ImportError:
            logger.error(
                "python-osc is required: pip install python-osc"
            )
            return

        from pythonosc.udp_client import SimpleUDPClient

        if "unity" in self.config.enabled_targets:
            self._clients["unity"] = SimpleUDPClient(
                self.config.unity_ip, self.config.unity_port
            )
            logger.info(
                "OSC → Unity @ %s:%d", self.config.unity_ip, self.config.unity_port
            )

        if "sc" in self.config.enabled_targets:
            self._clients["sc"] = SimpleUDPClient(
                self.config.sc_ip, self.config.sc_port
            )
            logger.info(
                "OSC → SuperCollider @ %s:%d", self.config.sc_ip, self.config.sc_port
            )

        if "pd" in self.config.enabled_targets:
            self._clients["pd"] = SimpleUDPClient(
                self.config.pd_ip, self.config.pd_port
            )
            logger.info(
                "OSC → Pure Data @ %s:%d", self.config.pd_ip, self.config.pd_port
            )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def start(self) -> None:
        """Start the heartbeat thread."""
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="osc-heartbeat"
        )
        self._heartbeat_thread.start()
        logger.info("OSC publisher started (heartbeat @ %.1f Hz)", self.config.heartbeat_hz)

    def stop(self) -> None:
        """Stop the heartbeat thread."""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
        logger.info("OSC publisher stopped")

    def publish(self, state: BrainState) -> None:
        """Publish a complete BrainState to all enabled targets."""
        self._send_meta(state)
        self._send_roi(state)
        self._send_pca(state)

        if self.config.send_full_vertices:
            self._send_full_vertices(state)

    def publish_delta(self, delta: BrainStateDelta) -> None:
        """Publish a BrainStateDelta (for particle effects in Unity)."""
        self._send_to_target(
            "unity",
            self.config.ADDR_BRAIN_DELTA,
            [float(delta.mean_abs_change), delta.max_increase_idx, delta.max_decrease_idx],
        )

    # -------------------------------------------------------------------
    # Internal send methods
    # -------------------------------------------------------------------

    def _send_meta(self, state: BrainState) -> None:
        """Send metadata to all targets."""
        args = [
            state.timestamp,
            state.sequence_id,
            state.latency_ms,
        ]
        for target in self.config.enabled_targets:
            self._send_to_target(target, self.config.ADDR_META, args)

    def _send_roi(self, state: BrainState) -> None:
        """Send ROI averages to all targets."""
        if not state.roi_averages:
            return

        # Convert dict to flat list of floats (sorted by key for consistency)
        sorted_keys = sorted(state.roi_averages.keys())
        values = [float(state.roi_averages[k]) for k in sorted_keys]

        for target in self.config.enabled_targets:
            self._send_to_target(target, self.config.ADDR_BRAIN_ROI, values)

    def _send_pca(self, state: BrainState) -> None:
        """Send PCA components to SC and PD."""
        values = [float(v) for v in state.pca_components]
        for target in ("sc", "pd"):
            if target in self.config.enabled_targets:
                self._send_to_target(target, self.config.ADDR_BRAIN_PCA, values)

    def _send_full_vertices(self, state: BrainState) -> None:
        """Send full vertex data to Unity in chunks."""
        chunks = state.to_chunks(self.config.chunk_size)
        for i, chunk in enumerate(chunks):
            addr = f"{self.config.ADDR_BRAIN_FULL_PREFIX}/{i}"
            values = [float(v) for v in chunk]
            # Prepend sequence_id and chunk_index for reassembly
            args = [state.sequence_id, i, len(chunks)] + values
            self._send_to_target("unity", addr, args)

    def _send_to_target(self, target: str, address: str, args: list) -> None:
        """Send an OSC message to a specific target."""
        client = self._clients.get(target)
        if client is None:
            return
        try:
            client.send_message(address, args)
        except Exception as exc:
            logger.debug("OSC send failed [%s %s]: %s", target, address, exc)

    # -------------------------------------------------------------------
    # Heartbeat
    # -------------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat messages."""
        interval = 1.0 / self.config.heartbeat_hz
        while self._running:
            args = [time.time()]
            for target in self.config.enabled_targets:
                self._send_to_target(target, self.config.ADDR_HEARTBEAT, args)
            time.sleep(interval)
