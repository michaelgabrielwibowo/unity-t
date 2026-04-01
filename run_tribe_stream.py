#!/usr/bin/env python3
"""
TRIBE v2 Streaming Simulation — Master Launcher
=================================================
Loads configuration, initializes the model with TurboQuant optimization,
starts ingestors, connects OSC publisher, and runs the streaming loop.

Usage
-----
::

    # Default (webcam + mic + ASR)
    python run_tribe_stream.py

    # With custom config
    python run_tribe_stream.py --config config/my_config.yaml

    # With video file input
    python run_tribe_stream.py --video path/to/stimulus.mp4

    # Benchmark mode (5 steps, then exit)
    python run_tribe_stream.py --benchmark --steps 5

    # CPU-only mode
    python run_tribe_stream.py --device cpu

    # iGPU / DirectML mode (Intel/AMD integrated graphics)
    python run_tribe_stream.py --device dml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Ensure stdout can print Unicode characters safely (e.g. arrows) on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from streaming.device_utils import device_info, resolve_device  # noqa: E402
from streaming.cpu_optimization import (
    patch_cuda_for_cpu,
    patch_extractors_for_cpu,
    patch_projectors
)

patch_cuda_for_cpu()

logger = logging.getLogger("tribe_stream")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    """Configure logging from config."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    fmt = "[%(asctime)s %(levelname)s] %(name)s: %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]
    log_file = log_config.get("file")
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def main():
    parser = argparse.ArgumentParser(
        description="TRIBE v2 Real-Time Streaming Brain Simulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/tribe_stream_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--video", type=str, help="Override video source")
    parser.add_argument("--audio", type=str, help="Override audio source")
    parser.add_argument("--text", type=str, help="Override text source")
    parser.add_argument("--device", type=str, help="Override device (cuda/cpu)")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run in benchmark mode"
    )
    parser.add_argument(
        "--steps", type=int, default=0, help="Max steps (0=infinite)"
    )
    parser.add_argument(
        "--no-osc", action="store_true", help="Disable OSC output"
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------------
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        logger.info("Loaded config from %s", config_path)
    else:
        logger.warning("Config not found at %s — using defaults", config_path)
        config = {}

    setup_logging(config)

    model_config = config.get("model", {})
    stream_config = config.get("streaming", {})
    ingestor_config = config.get("ingestors", {})
    osc_config_dict = config.get("osc", {})

    # Apply CLI overrides then resolve to actual torch.device
    device_str = args.device or model_config.get("device", "auto")
    device = resolve_device(device_str)
    dev_info = device_info(device)

    video_source = args.video or ingestor_config.get("video", {}).get("source", "webcam")
    audio_source = args.audio or ingestor_config.get("audio", {}).get("source", "mic")
    text_source = args.text or ingestor_config.get("text", {}).get("source", "asr")

    logger.info("=" * 60)
    logger.info("  TRIBE v2 Streaming Brain Simulation")
    logger.info("=" * 60)
    logger.info("  Device:  %s (%s)", dev_info.get('name', device), dev_info.get('device'))
    logger.info("  Video:   %s", video_source)
    logger.info("  Audio:   %s", audio_source)
    logger.info("  Text:    %s", text_source)
    logger.info("  Window:  %.0fs  |  Stride: %.1fs",
                stream_config.get("window_sec", 40),
                stream_config.get("stride_sec", 1))
    logger.info("=" * 60)

    # ---------------------------------------------------------------
    # 2. Load TRIBE v2 Model
    # ---------------------------------------------------------------
    logger.info("Loading TRIBE v2 model...")

    try:
        # Add tribev2 to path
        tribev2_path = Path(__file__).parent / "tribev2"
        if tribev2_path.exists():
            sys.path.insert(0, str(tribev2_path))

        from tribev2.demo_utils import TribeModel

        checkpoint = model_config.get("checkpoint", "facebook/tribev2")
        cache_folder = model_config.get("cache_folder", "./cache")

        model = TribeModel.from_pretrained(
            checkpoint,
            cache_folder=cache_folder,
            device=str(device),
        )
        logger.info("Model loaded successfully")
        
        # Apply CPU optimizations to avoid 8-Billion param wait time and OOMs
        if str(device.type) == "cpu" or str(device.type) == "privateuseone":
            patch_extractors_for_cpu(model)
            patch_projectors(model._model)

    except Exception as exc:
        logger.error("Failed to load TRIBE v2 model: %s", exc)
        logger.error("Ensure 'tribev2' package is installed or exists in the repository root. Exiting.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # 3. Apply TurboQuant Optimization
    # ---------------------------------------------------------------
    quant_config = model_config.get("quantization", {})

    if model is not None and model._model is not None:
        from streaming.quantization import QuantizationManager
        from streaming.cpu_optimization import apply_turboquant
        
        qm = QuantizationManager(target_vram_gb=12.0)

        # Apply Dynamic Quantization to the FmriEncoder on CPU
        if quant_config.get("turboquant_bits") or quant_config.get("turboquant", False):
            try:
                if str(device.type) in ("cpu", "privateuseone"):
                    stats = apply_turboquant(model._model)
                    logger.info("Dynamic Quantization: %d attention layers quantized (%.1f MB saved)", 
                        stats.get("layers_patched", 0), stats.get("savings_mb", 0.0))
                else:
                    logger.info("GPU TurboQuant is currently an experimental WIP, skipping CPU dynamic quant.")
            except Exception as exc:
                logger.warning("Dynamic Quantization failed: %s", exc)

        # Apply fp16 + torch.compile
        if quant_config.get("fp16_extractors", True):
            model._model = qm.optimize_model(
                model._model,
                component_name="fmri_encoder",
                use_fp16=True,
                use_compile=quant_config.get("compile_encoder", True),
            )

        logger.info("Quantization report: %s", qm.report())

    # ---------------------------------------------------------------
    # 4. Initialize Streaming Engine
    # ---------------------------------------------------------------
    from streaming.stream_engine import TribeStreamEngine

    if model is not None:
        engine = TribeStreamEngine(
            model=model,
            window_sec=stream_config.get("window_sec", 40.0),
            stride_sec=stream_config.get("stride_sec", 1.0),
            max_latency_ms=stream_config.get("max_latency_ms", 1000.0),
            device=device,
        )

    # ---------------------------------------------------------------
    # 5. Add Ingestors
    # ---------------------------------------------------------------
    video_cfg = ingestor_config.get("video", {})
    audio_cfg = ingestor_config.get("audio", {})
    text_cfg = ingestor_config.get("text", {})

    engine.add_video_ingestor(
        source=video_source,
        fps=video_cfg.get("fps", 30),
    )

    import queue
    asr_q = queue.Queue(maxsize=10) if text_source == "asr" else None

    engine.add_audio_ingestor(
        source=audio_source,
        sample_rate=audio_cfg.get("sample_rate", 16000),
        audio_queue=asr_q,
    )
    engine.add_text_ingestor(
        source=text_source,
        audio_queue=asr_q,
    )

    logger.info("Ingestors configured: video=%s, audio=%s, text=%s",
                video_source, audio_source, text_source)

    # ---------------------------------------------------------------
    # 6. Connect OSC Publisher
    # ---------------------------------------------------------------
    if not args.no_osc:
        from streaming.osc_config import OSCConfig
        from streaming.osc_publisher import BrainStatePublisher

        osc_cfg = OSCConfig(
            unity_ip=osc_config_dict.get("unity_ip", "127.0.0.1"),
            unity_port=osc_config_dict.get("unity_port", 9000),
            sc_ip=osc_config_dict.get("sc_ip", "127.0.0.1"),
            sc_port=osc_config_dict.get("sc_port", 57120),
            pd_ip=osc_config_dict.get("pd_ip", "127.0.0.1"),
            pd_port=osc_config_dict.get("pd_port", 9001),
            chunk_size=osc_config_dict.get("chunk_size", 5000),
            heartbeat_hz=osc_config_dict.get("heartbeat_hz", 10.0),
            send_full_vertices=osc_config_dict.get("send_full_vertices", True),
            enabled_targets=osc_config_dict.get("enabled_targets", ["unity", "sc"]),
        )

        publisher = BrainStatePublisher(osc_cfg)
        publisher.start()

        engine.on_brain_state = publisher.publish

        logger.info(
            "OSC publisher active → Unity:%d, SC:%d",
            osc_cfg.unity_port,
            osc_cfg.sc_port,
        )
    else:
        logger.info("OSC output disabled")

    # ---------------------------------------------------------------
    # 7. Run
    # ---------------------------------------------------------------
    max_steps = args.steps if args.benchmark else 0

    if args.benchmark:
        logger.info("BENCHMARK MODE: running %d steps", max_steps or 10)
        max_steps = max_steps or 10

    try:
        engine.run(max_steps=max_steps)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        engine.stop()
        if not args.no_osc and "publisher" in dir():
            publisher.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
