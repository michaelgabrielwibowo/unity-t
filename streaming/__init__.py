"""
TRIBE v2 Streaming Inference Engine
====================================
Real-time, always-on streaming pipeline that wraps Meta's TRIBE v2
brain-encoding model for continuous fMRI prediction from live
video, audio, and language stimuli.

Modules
-------
brain_state      – Canonical BrainState output dataclass
osc_config       – OSC transport configuration
feature_cache    – Incremental feature extraction with ring-buffer caching
ingestors        – Video / Audio / Text stream ingestors
stream_engine    – Core TribeStreamEngine orchestrator
osc_publisher    – Multi-address OSC publisher
turboquant_wrapper – TurboQuant KV-cache compression integration
quantization     – Additional quantization utilities (fp16, torch.compile)
"""

__version__ = "0.1.0"
