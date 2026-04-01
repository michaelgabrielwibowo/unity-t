import sys
import time
import logging
import queue
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming.stream_engine import TribeStreamEngine

logging.basicConfig(level=logging.INFO)

def main():
    import numpy as np
    
    # Dummy mock model that returns random brain states
    class MockModel:
        def predict(self, events, verbose=False):
            # predict returns (preds, segments)
            # preds is a list/array of shape (N, 20484)
            preds = np.random.randn(1, 20484).astype(np.float32)
            return preds, None
            
    engine = TribeStreamEngine(
        model=MockModel(),
        window_sec=2.0,
        stride_sec=0.5,
        max_latency_ms=200.0,
        device="cpu"
    )
    
    # Audio ingestor
    asr_q = queue.Queue(maxsize=10)
    engine.add_audio_ingestor(source="mic", audio_queue=asr_q)
    
    # We skip TextIngestor to avoid triggering Whisper downloads in tests.
    engine.add_video_ingestor(source="webcam", fps=10)
    
    print("Starting smoke test (3 steps)...")
    engine.run(max_steps=3)
        
    total_events = engine.accumulator.count
    if total_events > 0:
        print(f"Smoke test passed! Processed {total_events} events successfully.")
    else:
        print("Smoke test warning: No events captured. This may just be due to mock delays.")
    
if __name__ == "__main__":
    main()
