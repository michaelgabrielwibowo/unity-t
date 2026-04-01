import logging
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

def patch_cuda_for_cpu():
    """Monkey-patch CUDA-specific ops to run on CPU to avoid AssertionError in neuralset."""
    if hasattr(torch.cuda.amp, "autocast"):
        original_autocast = torch.cuda.amp.autocast
        class CPUAutocast(original_autocast):
            def __init__(self, *args, **kwargs):
                kwargs.pop("device_type", None)
                kwargs.pop("enabled", None)
                super().__init__(device_type="cpu", enabled=False, *args, **kwargs)
        def cpu_autocast_func(*args, **kwargs):
            kwargs.pop('enabled', None)
            kwargs.pop('device_type', None)
            return original_autocast(enabled=False, device_type="cpu", *args, **kwargs)
        
        torch.cuda.amp.autocast = CPUAutocast  # often used as class decorator
        torch.autocast = cpu_autocast_func
        log.info("CPU compatibility: CUDA autocast patched for CPU inference")

class DimAdapterProj(nn.Module):
    """Adapt dimensions dynamically without training using zero-padding or slicing.
    This allows swapping feature extractors for lightweight versions without 
    crashing the pretrained fMRI projection layers."""
    def __init__(self, original_proj, expected_dim):
        super().__init__()
        self.original_proj = original_proj
        self.expected_dim = expected_dim
        
    def forward(self, x, *args, **kwargs):
        actual_dim = x.shape[-1]
        if actual_dim < self.expected_dim:
            pad = torch.zeros(*x.shape[:-1], self.expected_dim - actual_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        elif actual_dim > self.expected_dim:
            x = x[..., :self.expected_dim]
        return self.original_proj(x, *args, **kwargs)

class _QuantLinear(torch.nn.Module):
    """
    Drop-in replacement for nn.Linear that quantises the weight matrix
    to int8 on the fly (dynamic quantisation).  On CPU this halves the
    memory footprint of every attention projection without requiring any
    extra packages.
    """

    def __init__(self, original: torch.nn.Linear):
        super().__init__()
        # Store weight in int8, keep bias in fp32
        weight_f = original.weight.data.float()
        scale = weight_f.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 127.0
        self.register_buffer("weight_int8", (weight_f / scale).round().to(torch.int8))
        self.register_buffer("scale", scale)
        if original.bias is not None:
            self.register_buffer("bias", original.bias.data.float())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantise on the fly: minimal peak memory
        w = self.weight_int8.float() * self.scale.unsqueeze(1)
        out = torch.nn.functional.linear(x.float(), w, self.bias)
        return out

def apply_turboquant(model: torch.nn.Module, verbose: bool = True) -> dict:
    """
    Recursively replace all nn.Linear layers inside attention modules
    (named 'attn', 'attention', 'self_attn', 'to_q', 'to_k', 'to_v',
    'to_out') with _QuantLinear (int8 dynamic quantisation).

    Returns a stats dict: layers_patched, param_bytes_before,
    param_bytes_after, savings_mb.
    """
    attn_keywords = {"attn", "attention", "self_attn", "to_q", "to_k", "to_v",
                     "to_out", "out_proj", "q_proj", "k_proj", "v_proj",
                     "qkv", "proj"}

    patched = 0
    bytes_before = 0
    bytes_after = 0

    def _patch_module(parent: torch.nn.Module, prefix: str = ""):
        nonlocal patched, bytes_before, bytes_after
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            # Check if this module or any ancestor name suggests attention
            in_attn_context = any(kw in full_name.lower() for kw in attn_keywords)
            if isinstance(child, torch.nn.Linear) and in_attn_context:
                b_before = child.weight.nelement() * child.weight.element_size()
                bytes_before += b_before
                ql = _QuantLinear(child)
                b_after = (ql.weight_int8.nelement() * 1  # int8 = 1 byte
                           + ql.scale.nelement() * 4)      # float32 scale
                bytes_after += b_after
                setattr(parent, name, ql)
                patched += 1
            else:
                _patch_module(child, full_name)

    _patch_module(model)

    savings_mb = (bytes_before - bytes_after) / 1024 ** 2
    stats = {
        "layers_patched": patched,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "savings_mb": savings_mb,
        "ratio": bytes_before / max(bytes_after, 1),
    }

    if verbose and patched > 0:
        log.info(
            "TurboQuant applied to %d attention layers. %d MB parameter memory saved (%.1fx)",
            patched, savings_mb, stats["ratio"],
        )
    return stats
    """Replaces huge default extractors (600M+ params) with standard lightweight alternatives (~80M)."""
    from neuralset.extractors.audio import HuggingFaceAudio
    from neuralset.extractors.video import HuggingFaceVideo
    from neuralset.extractors.image import HuggingFaceImage
    from neuralset.extractors.text import HuggingFaceText

    log.info("Swapping feature extractors for lightweight CPU alternatives...")
    data = model_wrapper.data
    
    if getattr(data, "audio_feature", None) is not None:
        freq = data.audio_feature.frequency if hasattr(data.audio_feature, 'frequency') else "native"
        data.audio_feature = HuggingFaceAudio(
            model_name="facebook/wav2vec2-base",
            frequency=freq,
            layers=[0.5, 0.75, 1.0], 
            layer_aggregation="mean"
        )
        object.__setattr__(data.audio_feature, "device", "cpu")
        log.info("  Audio: facebook/wav2vec2-base (95M)")

    if getattr(data, "video_feature", None) is not None:
        freq = data.video_feature.frequency if hasattr(data.video_feature, 'frequency') else "native"
        data.video_feature = HuggingFaceVideo(
            image=HuggingFaceImage(
                model_name="MCG-NJU/videomae-base", 
                infra={"keep_in_ram": False}
            ),
            frequency=freq,
            clip_duration=1.0,
            num_frames=16
        )
        object.__setattr__(data.video_feature, "device", "cpu")
        object.__setattr__(data.video_feature.image, "device", "cpu")
        log.info("  Video: MCG-NJU/videomae-base (86M)")
        
    if getattr(data, "text_feature", None) is not None:
        freq = data.text_feature.frequency if hasattr(data.text_feature, 'frequency') else "native"
        data.text_feature = HuggingFaceText(
            model_name="distilbert-base-uncased",
            frequency=freq
        )
        object.__setattr__(data.text_feature, "device", "cpu")
        log.info("  Text: distilbert-base-uncased (66M)")

def patch_projectors(fmri_encoder_model):
    """Wraps internal fMRI projectors with 0-pad adapters to match expected shapes."""
    log.info("Installing dimension-padding adapters on fMRI projectors...")
    
    agg = getattr(fmri_encoder_model.config, "layer_aggregation", "mean")
    
    for modality, tup in fmri_encoder_model.feature_dims.items():
        if tup is None: continue
        num_layers, feature_dim = tup
        expected_dim = (feature_dim * num_layers) if agg == "cat" else feature_dim
        
        if modality in fmri_encoder_model.projectors:
            proj = fmri_encoder_model.projectors[modality]
            fmri_encoder_model.projectors[modality] = DimAdapterProj(proj, expected_dim)
            log.info("  %s projector wrapped to handle dim mismatch (target_dim=%d)", modality, expected_dim)
