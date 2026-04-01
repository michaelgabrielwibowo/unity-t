"""
Stream Ingestors — Video, Audio, and Text
==========================================
Concurrent ingestor threads that capture live or quasi-live sensory
streams and maintain a rolling events DataFrame compatible with
TRIBE v2's ``model.get_events_dataframe()`` format.

Each ingestor runs in its own daemon thread and writes to a shared
``EventAccumulator``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event Accumulator  — shared sink for all ingestors
# ---------------------------------------------------------------------------

class EventAccumulator:
    """Thread-safe rolling storage for TRIBE v2-style events.

    Ingestors append events here; the streaming engine reads the current
    window via ``get_window()``.
    """

    def __init__(self, max_duration_sec: float = 120.0):
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._max_duration = max_duration_sec

    def append(self, event: Dict[str, Any]) -> None:
        """Append a single event dict."""
        with self._lock:
            self._events.append(event)

    def append_batch(self, events: List[Dict[str, Any]]) -> None:
        """Append multiple events at once."""
        with self._lock:
            self._events.extend(events)

    def get_window(self, start_time: float, end_time: float) -> pd.DataFrame:
        """Return events within the time window [start_time, end_time]."""
        with self._lock:
            if not self._events:
                return pd.DataFrame()
            df = pd.DataFrame(self._events)
            if "start" not in df.columns:
                return pd.DataFrame()
            mask = (df["start"] >= start_time) & (df["start"] <= end_time)
            return df[mask].copy()

    def get_all(self) -> pd.DataFrame:
        """Return all accumulated events."""
        with self._lock:
            if not self._events:
                return pd.DataFrame()
            return pd.DataFrame(self._events).copy()

    def trim(self, before_time: float) -> int:
        """Remove events older than *before_time*.  Returns count removed."""
        with self._lock:
            original_len = len(self._events)
            self._events = [
                e for e in self._events if e.get("start", 0) >= before_time
            ]
            return original_len - len(self._events)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._events)


# ---------------------------------------------------------------------------
# Base Ingestor
# ---------------------------------------------------------------------------

class BaseIngestor(ABC, threading.Thread):
    """Base class for all stream ingestors.

    Subclasses must implement ``_run_loop()``.  The ingestor is a daemon
    thread that writes to the shared ``EventAccumulator``.
    """

    def __init__(
        self,
        accumulator: EventAccumulator,
        name: str = "ingestor",
        start_time_ref: float | None = None,
    ):
        super().__init__(name=name, daemon=True)
        self.accumulator = accumulator
        self._stop_event = threading.Event()
        self._start_time_ref = start_time_ref or time.time()
        self._frames_ingested = 0

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    @property
    def elapsed(self) -> float:
        """Seconds since this ingestor started."""
        return time.time() - self._start_time_ref

    def run(self) -> None:
        logger.info("[%s] Ingestor started", self.name)
        try:
            self._run_loop()
        except Exception as exc:
            logger.exception("[%s] Ingestor crashed: %s", self.name, exc)
        finally:
            logger.info(
                "[%s] Ingestor stopped (%d frames ingested)",
                self.name,
                self._frames_ingested,
            )

    @abstractmethod
    def _run_loop(self) -> None: ...


# ---------------------------------------------------------------------------
# Video Ingestor
# ---------------------------------------------------------------------------

class VideoIngestor(BaseIngestor):
    """Captures video frames from a webcam or video file.

    Writes ``Video`` events to the accumulator at the configured FPS.
    Frames are stored (or referenced by path) for the feature extractor
    to process later.

    Parameters
    ----------
    source : str
        ``"webcam"`` for live capture, or a path to a video file.
    fps : int
        Target capture rate.  Frames are decimated to this rate.
    temp_dir : str
        Directory for temporary frame storage.
    """

    def __init__(
        self,
        accumulator: EventAccumulator,
        source: str = "webcam",
        fps: int = 30,
        temp_dir: str = "./cache/video_frames",
        **kwargs,
    ):
        super().__init__(accumulator, name="video-ingestor", **kwargs)
        self.source = source
        self.fps = fps
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _run_loop(self) -> None:
        try:
            import cv2
        except ImportError:
            logger.error("opencv-python is required for video ingest: pip install opencv-python")
            return

        if self.source == "webcam":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            logger.error("Cannot open video source: %s", self.source)
            return

        frame_interval = 1.0 / self.fps
        chunk_duration = 1.0  # 1-second video chunks
        chunk_start = self.elapsed
        chunk_frames = []

        try:
            while not self.is_stopped:
                ret, frame = cap.read()
                if not ret:
                    if self.source != "webcam":
                        logger.info("[video] End of file reached, looping")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                current_time = self.elapsed
                chunk_frames.append(frame)
                self._frames_ingested += 1

                # Every 1-second chunk, emit a Video event
                if current_time - chunk_start >= chunk_duration:
                    # Save chunk as a temp video file for the extractor
                    chunk_path = self.temp_dir / f"chunk_{self._frames_ingested:08d}.mp4"
                    h, w = chunk_frames[0].shape[:2]
                    writer = cv2.VideoWriter(
                        str(chunk_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.fps,
                        (w, h),
                    )
                    for f in chunk_frames:
                        writer.write(f)
                    writer.release()

                    event = {
                        "type": "Video",
                        "filepath": str(chunk_path),
                        "start": chunk_start,
                        "duration": current_time - chunk_start,
                        "timeline": "default",
                        "subject": "default",
                        "split": "all",
                    }
                    self.accumulator.append(event)
                    chunk_frames = []
                    chunk_start = current_time

                # Throttle to target FPS
                time.sleep(max(0, frame_interval - 0.001))
        finally:
            cap.release()


# ---------------------------------------------------------------------------
# Audio Ingestor
# ---------------------------------------------------------------------------

class AudioIngestor(BaseIngestor):
    """Captures audio from a microphone or audio file.

    Writes ``Audio`` events to the accumulator in 1-second chunks.

    Parameters
    ----------
    source : str
        ``"mic"`` for live capture, or a path to an audio file.
    sample_rate : int
        Audio sample rate (default 16 kHz for Wav2Vec-BERT).
    temp_dir : str
        Directory for temporary audio chunk storage.
    """

    def __init__(
        self,
        accumulator: EventAccumulator,
        source: str = "mic",
        sample_rate: int = 16000,
        temp_dir: str = "./cache/audio_chunks",
        audio_queue: Optional[queue.Queue] = None,
        **kwargs,
    ):
        super().__init__(accumulator, name="audio-ingestor", **kwargs)
        self.source = source
        self.sample_rate = sample_rate
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.audio_queue = audio_queue

    def _run_loop(self) -> None:
        if self.source == "mic":
            self._run_mic_loop()
        else:
            self._run_file_loop()

    def _run_mic_loop(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice is required for mic input: pip install sounddevice")
            return

        chunk_samples = self.sample_rate  # 1 second of audio
        chunk_idx = 0

        while not self.is_stopped:
            try:
                audio = sd.rec(
                    chunk_samples,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
            except Exception as exc:
                logger.warning("Audio capture error: %s", exc)
                time.sleep(0.5)
                continue

            current_time = self.elapsed
            chunk_path = self.temp_dir / f"chunk_{chunk_idx:08d}.wav"

            try:
                import soundfile as sf
                sf.write(str(chunk_path), audio.flatten(), self.sample_rate)
            except ImportError:
                # Fallback: save as raw numpy
                np.save(str(chunk_path).replace(".wav", ".npy"), audio)

            event = {
                "type": "Audio",
                "filepath": str(chunk_path),
                "start": current_time - 1.0,
                "duration": 1.0,
                "timeline": "default",
                "subject": "default",
                "split": "all",
            }
            self.accumulator.append(event)
            if self.audio_queue is not None and not self.audio_queue.full():
                self.audio_queue.put(chunk_path)
            self._frames_ingested += 1
            chunk_idx += 1

    def _run_file_loop(self) -> None:
        """Play back an audio file as if it were a live stream."""
        try:
            import soundfile as sf
        except ImportError:
            logger.error("soundfile is required: pip install soundfile")
            return

        audio_data, sr = sf.read(self.source)
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]  # mono

        # Resample if needed
        if sr != self.sample_rate:
            try:
                import julius
                import torch as _torch
                audio_tensor = _torch.from_numpy(audio_data).float().unsqueeze(0)
                audio_tensor = julius.resample_frac(audio_tensor, sr, self.sample_rate)
                audio_data = audio_tensor.squeeze(0).numpy()
            except ImportError:
                logger.warning("julius not available; using raw sample rate %d", sr)
                self.sample_rate = sr

        chunk_samples = self.sample_rate  # 1 second
        total_chunks = len(audio_data) // chunk_samples

        chunk_idx = 0
        while not self.is_stopped:
            idx = chunk_idx % total_chunks
            start_sample = idx * chunk_samples
            chunk = audio_data[start_sample : start_sample + chunk_samples]

            chunk_path = self.temp_dir / f"chunk_{chunk_idx:08d}.wav"
            try:
                import soundfile as sf_write
                sf_write.write(str(chunk_path), chunk, self.sample_rate)
            except Exception:
                np.save(str(chunk_path).replace(".wav", ".npy"), chunk)

            current_time = self.elapsed
            event = {
                "type": "Audio",
                "filepath": str(chunk_path),
                "start": current_time - 1.0,
                "duration": 1.0,
                "timeline": "default",
                "subject": "default",
                "split": "all",
            }
            self.accumulator.append(event)
            if self.audio_queue is not None and not self.audio_queue.full():
                self.audio_queue.put(chunk_path)
            self._frames_ingested += 1
            chunk_idx += 1
            time.sleep(1.0)  # Real-time pacing


# ---------------------------------------------------------------------------
# Text Ingestor
# ---------------------------------------------------------------------------

class TextIngestor(BaseIngestor):
    """Generates word-level events from text or ASR on the audio stream.

    Modes:
    - ``"asr"``: Run Whisper on the audio stream for real-time transcription
    - ``"file"``: Read a text file and pace word delivery to match real-time

    Parameters
    ----------
    source : str
        ``"asr"`` for live transcription, or a path to a ``.txt`` file.
    words_per_second : float
        Pacing rate when reading from a file.
    asr_model : str
        Whisper model name when *source* is ``"asr"``.
    """

    def __init__(
        self,
        accumulator: EventAccumulator,
        source: str = "asr",
        words_per_second: float = 3.0,
        asr_model: str = "openai/whisper-small",
        audio_queue: Optional[queue.Queue] = None,
        **kwargs,
    ):
        super().__init__(accumulator, name="text-ingestor", **kwargs)
        self.source = source
        self.words_per_second = words_per_second
        self.asr_model = asr_model
        self.audio_queue = audio_queue

    def _run_loop(self) -> None:
        if self.source == "asr":
            self._run_asr_loop()
        else:
            self._run_file_loop()

    def _run_asr_loop(self) -> None:
        """Placeholder for Whisper-based ASR transcription.

        In a full deployment, this would:
        1. Read audio chunks from a shared queue (fed by AudioIngestor)
        2. Run Whisper inference to get word-level timestamps
        3. Emit Word events to the accumulator
        """
        logger.info("[text/asr] ASR mode — requires Whisper integration")
        if self.audio_queue is None:
            logger.error("[text/asr] audio_queue is None! ASR disabled.")
            return

        try:
            import whisper
            asr_model = whisper.load_model("small")
        except ImportError:
            logger.warning("openai-whisper not installed; text ingestor will emit empty events")
            while not self.is_stopped:
                time.sleep(1.0)
            return

        # Poll for audio chunks and transcribe
        while not self.is_stopped:
            if self.audio_queue and not self.audio_queue.empty():
                audio_path = self.audio_queue.get()
                try:
                    result = asr_model.transcribe(str(audio_path), word_timestamps=True)
                    for segment in result.get("segments", []):
                        for word_info in segment.get("words", []):
                            event = {
                                "type": "Word",
                                "word": word_info["word"].strip(),
                                "start": self.elapsed + word_info["start"],
                                "duration": word_info["end"] - word_info["start"],
                                "timeline": "default",
                                "subject": "default",
                                "split": "all",
                            }
                            self.accumulator.append(event)
                            self._frames_ingested += 1
                except Exception as exc:
                    logger.warning("ASR transcription failed: %s", exc)
            else:
                time.sleep(0.5)

    def _run_file_loop(self) -> None:
        """Read a text file and emit Word events at a steady pace."""
        path = Path(self.source)
        if not path.exists():
            logger.error("Text file not found: %s", path)
            return

        text = path.read_text(encoding="utf-8")
        words = text.split()
        if not words:
            logger.warning("Text file is empty: %s", path)
            return

        interval = 1.0 / self.words_per_second
        word_idx = 0

        while not self.is_stopped:
            word = words[word_idx % len(words)]
            current_time = self.elapsed
            event = {
                "type": "Word",
                "word": word,
                "start": current_time,
                "duration": interval * 0.8,
                "timeline": "default",
                "subject": "default",
                "split": "all",
            }
            self.accumulator.append(event)
            self._frames_ingested += 1
            word_idx += 1
            time.sleep(interval)
