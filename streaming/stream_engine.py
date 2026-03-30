"""
TribeStreamEngine — Core Streaming Inference Orchestrator
==========================================================
Converts TRIBE v2's batch ``predict()`` into a continuous, always-on
streaming loop.  Coordinates ingestors, incremental feature extraction,
model inference, and brain-state publication at ~1 Hz.

Architecture
------------
::

    ┌── Ingestor Threads ──┐
    │  VideoIngestor        │
    │  AudioIngestor        │──▶  EventAccumulator
    │  TextIngestor         │
    └───────────────────────┘
                │
                ▼
    ┌── Main Thread ────────────────────────────────┐
    │  loop @ 1 Hz:                                  │
    │    1. Read latest 40 s window from accumulator  │
    │    2. Extract features for new 1 s slice        │
    │    3. Concatenate cached features               │
    │    4. Run FmriEncoder.forward()                 │
    │    5. Compute ROI averages + PCA               │
    │    6. Emit BrainState via callback              │
    └────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Any, Callable, List, Optional

import numpy as np
import torch

from .brain_state import (
    FSAVERAGE5_TOTAL_VERTICES,
    NUM_PCA_COMPONENTS,
    BrainState,
    BrainStateDelta,
)
from .ingestors import (
    AudioIngestor,
    BaseIngestor,
    EventAccumulator,
    TextIngestor,
    VideoIngestor,
)

logger = logging.getLogger(__name__)


class ROIComputer:
    """Computes ROI averages from vertex data using the Destrieux atlas.

    Falls back to coarse parcellation (5 equal strips per hemisphere)
    if nilearn is not available.
    """

    def __init__(self):
        self._labels = None
        self._label_names = None
        self._ready = False
        self._init()

    def _init(self):
        try:
            from nilearn import datasets
            destrieux = datasets.fetch_atlas_surf_destrieux()
            # Labels for left and right hemisphere
            labels_lh = np.array(destrieux["labels_left"])
            labels_rh = np.array(destrieux["labels_right"])
            self._labels = np.concatenate([labels_lh, labels_rh])
            # Get unique label names
            self._label_names = {
                int(k): v.decode() if isinstance(v, bytes) else str(v)
                for k, v in enumerate(destrieux.get("labels", []))
            }
            self._ready = True
            logger.info("ROI computer initialized with Destrieux atlas (%d regions)", len(set(labels_lh)))
        except Exception as exc:
            logger.warning("Could not load Destrieux atlas: %s — using coarse fallback", exc)
            self._init_fallback()

    def _init_fallback(self):
        """Create coarse pseudo-ROIs: 10 equal strips per hemisphere."""
        n_per_hemi = FSAVERAGE5_TOTAL_VERTICES // 2
        n_strips = 10
        lh_labels = np.repeat(np.arange(n_strips), n_per_hemi // n_strips + 1)[:n_per_hemi]
        rh_labels = np.repeat(np.arange(n_strips, 2 * n_strips), n_per_hemi // n_strips + 1)[:n_per_hemi]
        self._labels = np.concatenate([lh_labels, rh_labels])
        self._label_names = {i: f"strip_{i}" for i in range(2 * n_strips)}
        self._ready = True

    def compute(self, vertices: np.ndarray) -> dict[str, float]:
        """Compute mean activation per ROI."""
        if not self._ready or self._labels is None:
            return {}
        result = {}
        # Ensure labels match vertex count (truncate if needed)
        labels = self._labels[:len(vertices)]
        for label_id in np.unique(labels):
            mask = labels == label_id
            name = self._label_names.get(int(label_id), f"roi_{label_id}")
            result[name] = float(np.mean(vertices[mask]))
        return result


class PCAProjector:
    """Projects vertex data to low-dimensional PCA space.

    Learns the PCA basis from the first N brain states (warm-up),
    then projects subsequent states in O(1).
    """

    def __init__(self, n_components: int = NUM_PCA_COMPONENTS, warmup_states: int = 30):
        self.n_components = n_components
        self.warmup_states = warmup_states
        self._basis: Optional[np.ndarray] = None  # (n_components, n_vertices)
        self._mean: Optional[np.ndarray] = None
        self._warmup_buffer: List[np.ndarray] = []

    def update_and_project(self, vertices: np.ndarray) -> np.ndarray:
        """Add state to warmup buffer or project using learned basis."""
        if self._basis is None:
            self._warmup_buffer.append(vertices)
            if len(self._warmup_buffer) >= self.warmup_states:
                self._fit()
            return np.zeros(self.n_components, dtype=np.float32)
        return self._project(vertices)

    def _fit(self):
        """Fit PCA basis from collected warmup states."""
        data = np.stack(self._warmup_buffer, axis=0)  # (N, V)
        self._mean = data.mean(axis=0)
        centered = data - self._mean
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            self._basis = Vt[: self.n_components]  # (K, V)
        except np.linalg.LinAlgError:
            logger.warning("SVD failed during PCA fit — using random basis")
            self._basis = np.random.randn(self.n_components, data.shape[1]).astype(np.float32)
            self._basis /= np.linalg.norm(self._basis, axis=1, keepdims=True)
        self._warmup_buffer.clear()
        logger.info("PCA basis fitted (%d components)", self.n_components)

    def _project(self, vertices: np.ndarray) -> np.ndarray:
        centered = vertices - self._mean
        return (self._basis @ centered).astype(np.float32)


class TribeStreamEngine:
    """Always-on streaming inference wrapper for TRIBE v2.

    Parameters
    ----------
    model : TribeModel
        A loaded TRIBE v2 model (from ``TribeModel.from_pretrained``).
    window_sec : float
        Sliding window duration in seconds (typically 40 s to match
        TRIBE v2's ``duration_trs``).
    stride_sec : float
        Prediction stride — how often to produce a new brain state.
        Default 1.0 s for ~1 Hz update rate.
    device : str
        Torch device for inference.
    """

    def __init__(
        self,
        model: Any,  # TribeModel
        window_sec: float = 40.0,
        stride_sec: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.device = device

        # Event accumulation
        self.accumulator = EventAccumulator(max_duration_sec=window_sec * 3)

        # Ingestors
        self._ingestors: List[BaseIngestor] = []

        # ROI + PCA postprocessors
        self._roi_computer = ROIComputer()
        self._pca_projector = PCAProjector()

        # State
        self._running = False
        self._sequence_id = 0
        self._prev_state: Optional[BrainState] = None

        # Callback for brain state consumers
        self.on_brain_state: Optional[Callable[[BrainState], None]] = None
        self.on_brain_delta: Optional[Callable[[BrainStateDelta], None]] = None

    # -------------------------------------------------------------------
    # Ingestor management
    # -------------------------------------------------------------------

    def add_video_ingestor(
        self, source: str = "webcam", fps: int = 30, **kwargs
    ) -> VideoIngestor:
        ing = VideoIngestor(
            self.accumulator, source=source, fps=fps,
            start_time_ref=time.time(), **kwargs
        )
        self._ingestors.append(ing)
        return ing

    def add_audio_ingestor(
        self, source: str = "mic", sample_rate: int = 16000, **kwargs
    ) -> AudioIngestor:
        ing = AudioIngestor(
            self.accumulator, source=source, sample_rate=sample_rate,
            start_time_ref=time.time(), **kwargs
        )
        self._ingestors.append(ing)
        return ing

    def add_text_ingestor(
        self, source: str = "asr", **kwargs
    ) -> TextIngestor:
        ing = TextIngestor(
            self.accumulator, source=source,
            start_time_ref=time.time(), **kwargs
        )
        self._ingestors.append(ing)
        return ing

    # -------------------------------------------------------------------
    # Core inference loop
    # -------------------------------------------------------------------

    def run(self, max_steps: int = 0) -> None:
        """Run the streaming inference loop.

        Blocks until interrupted (Ctrl+C) or *max_steps* is reached
        (0 = infinite).
        """
        self._running = True
        self._setup_signal_handler()

        # Start all ingestors
        for ing in self._ingestors:
            ing.start()
            logger.info("Started ingestor: %s", ing.name)

        # Wait for initial data accumulation
        warmup_sec = min(5.0, self.window_sec / 4)
        logger.info("Warming up for %.1f seconds...", warmup_sec)
        time.sleep(warmup_sec)

        step = 0
        logger.info("=== TRIBE v2 Streaming Engine LIVE ===")

        try:
            while self._running:
                t_start = time.time()

                # --- Predict ---
                brain_state = self._inference_step()

                if brain_state is not None:
                    # Compute delta
                    if self._prev_state is not None and self.on_brain_delta:
                        delta = BrainStateDelta.from_states(self._prev_state, brain_state)
                        self.on_brain_delta(delta)

                    # Publish
                    if self.on_brain_state:
                        self.on_brain_state(brain_state)

                    self._prev_state = brain_state
                    self._sequence_id += 1

                    logger.info(
                        "[step %d] latency=%.0fms  mean=%.4f  std=%.4f  seq=%d",
                        step,
                        brain_state.latency_ms,
                        brain_state.global_mean,
                        brain_state.global_std,
                        brain_state.sequence_id,
                    )

                step += 1
                if max_steps > 0 and step >= max_steps:
                    break

                # Trim old events
                trim_before = time.time() - self.window_sec * 2
                self.accumulator.trim(trim_before)

                # Pace to stride interval
                elapsed = time.time() - t_start
                sleep_time = max(0, self.stride_sec - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop all ingestors and the engine."""
        self._running = False
        for ing in self._ingestors:
            ing.stop()
        logger.info("=== TRIBE v2 Streaming Engine stopped ===")

    # -------------------------------------------------------------------
    # Single inference step
    # -------------------------------------------------------------------

    def _inference_step(self) -> Optional[BrainState]:
        """Perform one prediction step.

        1. Gather the current window of events
        2. Run TRIBE v2 inference
        3. Post-process (ROI, PCA)
        4. Package as BrainState
        """
        t_start = time.time()

        # Gather events for the current window
        events_df = self.accumulator.get_all()
        if events_df.empty or len(events_df) < 2:
            logger.debug("Not enough events yet (%d), skipping", len(events_df))
            return None

        try:
            # Use the model's predict method
            with torch.inference_mode():
                preds, segments = self.model.predict(events=events_df, verbose=False)

            if preds is None or len(preds) == 0:
                logger.debug("No predictions returned, skipping")
                return None

            # Take the last prediction (most recent time step)
            vertices = preds[-1].astype(np.float32)

            # Ensure correct shape
            if len(vertices) != FSAVERAGE5_TOTAL_VERTICES:
                logger.warning(
                    "Unexpected vertex count: %d (expected %d)",
                    len(vertices),
                    FSAVERAGE5_TOTAL_VERTICES,
                )
                # Pad or truncate
                if len(vertices) < FSAVERAGE5_TOTAL_VERTICES:
                    vertices = np.pad(
                        vertices,
                        (0, FSAVERAGE5_TOTAL_VERTICES - len(vertices)),
                    )
                else:
                    vertices = vertices[:FSAVERAGE5_TOTAL_VERTICES]

        except Exception as exc:
            logger.warning("Inference failed: %s — emitting zero state", exc)
            vertices = np.zeros(FSAVERAGE5_TOTAL_VERTICES, dtype=np.float32)

        # Post-processing
        roi_averages = self._roi_computer.compute(vertices)
        pca_components = self._pca_projector.update_and_project(vertices)

        latency_ms = (time.time() - t_start) * 1000

        return BrainState(
            timestamp=time.time(),
            vertices=vertices,
            roi_averages=roi_averages,
            pca_components=pca_components,
            sequence_id=self._sequence_id,
            latency_ms=latency_ms,
        )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _setup_signal_handler(self):
        """Handle Ctrl+C gracefully."""
        def handler(signum, frame):
            logger.info("Signal %s received, stopping...", signum)
            self._running = False

        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except (ValueError, OSError):
            pass  # Not on main thread or unsupported platform
