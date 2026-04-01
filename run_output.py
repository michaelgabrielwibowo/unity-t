#!/usr/bin/env python3
"""
TRIBE v2 — Output-Only Runner with TurboQuant Memory Compression
=================================================================
Runs TRIBE v2 on a stimulus file and writes brain activity predictions
directly to disk (NumPy .npy + human-readable CSV summary).

No Unity, no OSC, no visualisation — just raw fMRI predictions.

TurboQuant patches the transformer encoder's attention layers to compress
the KV cache in-place, reducing peak RAM by ~50-60% on CPU.

Usage
-----
    # Audio file
    python run_output.py --audio path/to/audio.wav

    # Video file
    python run_output.py --video path/to/video.mp4

    # Text file
    python run_output.py --text path/to/script.txt

    # Override output dir
    python run_output.py --audio clip.wav --out results/

    # Skip TurboQuant (full precision, more RAM)
    python run_output.py --audio clip.wav --no-turboquant
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tribev2"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tribe_output")


# ===================================================================
# CPU Compatibility Shims
# ===================================================================

from streaming.cpu_optimization import (
    patch_cuda_for_cpu,
    patch_extractors_for_cpu,
    patch_projectors,
    apply_turboquant
)

patch_cuda_for_cpu()


# ===================================================================
# ===================================================================
# Output helpers
# ===================================================================

def save_outputs(preds: np.ndarray, segments: list, out_dir: Path) -> None:
    """Save predictions to .npy and a human-readable summary CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # Full predictions — shape (n_segments, n_vertices)
    npy_path = out_dir / f"brain_activity_{ts}.npy"
    np.save(npy_path, preds)
    log.info("Saved full predictions -> %s  shape=%s", npy_path, preds.shape)

    # Summary CSV — per-segment stats
    rows = []
    for i, seg in enumerate(segments):
        row = {
            "segment_idx": i,
            "start_s": getattr(seg, "start", float("nan")),
            "duration_s": getattr(seg, "duration", float("nan")),
            "mean_activation": float(preds[i].mean()),
            "std_activation": float(preds[i].std()),
            "max_activation": float(preds[i].max()),
            "min_activation": float(preds[i].min()),
            "n_vertices": preds.shape[1],
        }
        rows.append(row)

    import csv
    csv_path = out_dir / f"summary_{ts}.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info("Saved summary CSV   -> %s  (%d rows)", csv_path, len(rows))

    # Print quick stats to console
    print("\n" + "=" * 54)
    print("  TRIBE v2 Brain Activity Predictions")
    print("=" * 54)
    print(f"  Segments predicted : {preds.shape[0]}")
    print(f"  Vertices per seg   : {preds.shape[1]:,}")
    print(f"  Global mean        : {preds.mean():.4f}")
    print(f"  Global std         : {preds.std():.4f}")
    print(f"  Global max         : {preds.max():.4f}")
    print(f"  Global min         : {preds.min():.4f}")
    print("=" * 54)
    print(f"  .npy  -> {npy_path}")
    print(f"  .csv  -> {csv_path}")
    print("=" * 54 + "\n")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TRIBE v2 output-only runner with TurboQuant"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", type=str, help="Path to audio file (.wav/.mp3/.flac)")
    group.add_argument("--video", type=str, help="Path to video file (.mp4/.avi/.mkv)")
    group.add_argument("--text",  type=str, help="Path to text file (.txt)")
    parser.add_argument(
        "--out", type=str, default="output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="facebook/tribev2",
        help="HuggingFace repo or local path to checkpoint"
    )
    parser.add_argument(
        "--cache", type=str, default="./cache",
        help="Feature cache directory"
    )
    parser.add_argument(
        "--audio-only", action="store_true",
        help="Skip ASR transcription (audio features only, no WhisperX needed)"
    )
    parser.add_argument(
        "--no-turboquant", action="store_true",
        help="Disable TurboQuant (full fp32, uses more RAM)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device: cpu (default)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    device = args.device

    log.info("=" * 54)
    log.info("  TRIBE v2 Output Runner")
    log.info("  Device     : %s", device)
    log.info("  TurboQuant : %s", "OFF" if args.no_turboquant else "ON (int8 attn)")
    log.info("  Checkpoint : %s", args.checkpoint)
    log.info("  Output dir : %s", out_dir)
    log.info("=" * 54)

    # -----------------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------------
    log.info("Loading TRIBE v2 model (this may take a moment)...")
    t0 = time.time()

    from tribev2.demo_utils import TribeModel

    model_wrapper = TribeModel.from_pretrained(
        args.checkpoint,
        cache_folder=args.cache,
        device=device,
    )

    load_time = time.time() - t0
    log.info("Model loaded in %.1fs", load_time)

    # Patch heavily for CPU environments
    patch_extractors_for_cpu(model_wrapper)
    patch_projectors(model_wrapper._model)

    # -----------------------------------------------------------
    # 2. Apply TurboQuant
    # -----------------------------------------------------------
    if not args.no_turboquant:
        log.info("Applying TurboQuant (int8 KV compression)...")
        inner_model = model_wrapper._model
        stats = apply_turboquant(inner_model, verbose=True)
        # Force garbage collect the original fp32 weights
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc; gc.collect()
        log.info(
            "Memory after quant: ~%.0f MB freed from attention layers",
            stats["savings_mb"]
        )
    else:
        log.info("TurboQuant disabled — running full fp32")

    # -----------------------------------------------------------
    # 3. Build events dataframe from input
    # -----------------------------------------------------------
    log.info("Building events dataframe from input...")
    t1 = time.time()

    if args.audio:
        import pandas as pd
        from neuralset.events.utils import standardize_events
        from tribev2.demo_utils import get_audio_and_text_events
        audio_only_mode = getattr(args, "audio_only", False)
        event = {
            "type": "Audio",
            "filepath": str(Path(args.audio).resolve()),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        events = get_audio_and_text_events(
            pd.DataFrame([event]),
            audio_only=audio_only_mode,
        )
    elif args.video:
        import pandas as pd
        from tribev2.demo_utils import get_audio_and_text_events
        audio_only_mode = getattr(args, "audio_only", False)
        event = {
            "type": "Video",
            "filepath": str(Path(args.video).resolve()),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        events = get_audio_and_text_events(
            pd.DataFrame([event]),
            audio_only=audio_only_mode,
        )
    else:
        events = model_wrapper.get_events_dataframe(text_path=args.text)

    log.info("Events ready in %.1fs  (%d rows)", time.time() - t1, len(events))

    # -----------------------------------------------------------
    # 4. Run inference
    # -----------------------------------------------------------
    log.info("Running inference (CPU — may take a few minutes per segment)...")
    t2 = time.time()

    with torch.inference_mode():
        preds, segments = model_wrapper.predict(events, verbose=True)

    inf_time = time.time() - t2
    log.info("Inference done in %.1fs", inf_time)

    # -----------------------------------------------------------
    # 5. Save outputs
    # -----------------------------------------------------------
    save_outputs(preds, segments, out_dir)

    total_time = time.time() - t0
    log.info("Total wall time: %.1fs", total_time)


if __name__ == "__main__":
    main()
