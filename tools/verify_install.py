"""Verify TRIBE v2 streaming stack installation."""
import sys

checks = []

try:
    import torch
    checks.append("PyTorch " + torch.__version__ + " (CUDA: " + str(torch.cuda.is_available()) + ")")
except Exception:
    checks.append("PyTorch: MISSING")

try:
    import cv2
    checks.append("OpenCV " + cv2.__version__)
except Exception:
    checks.append("OpenCV: MISSING")

try:
    from pythonosc.udp_client import SimpleUDPClient
    checks.append("python-osc: OK")
except Exception:
    checks.append("python-osc: MISSING")

try:
    import sounddevice
    checks.append("sounddevice: OK")
except Exception:
    checks.append("sounddevice: MISSING")

try:
    import yaml
    checks.append("pyyaml: OK")
except Exception:
    checks.append("pyyaml: MISSING")

try:
    sys.path.insert(0, "tribev2")
    import tribev2
    checks.append("tribev2: OK")
except Exception:
    checks.append("tribev2: MISSING")

for c in checks:
    print("  " + c)
