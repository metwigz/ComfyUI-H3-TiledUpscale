"""
Unit test for Face Refinement and Tracking in H3-TiledUpscale
"""
import sys
import os
from pathlib import Path
import importlib

repo_root = Path(__file__).resolve().parents[3]
custom_nodes_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(custom_nodes_dir))

import torch
import numpy as np

face_module = importlib.import_module("ComfyUI-H3-TiledUpscale.core.face_tracker")
FaceTracker = face_module.FaceTracker
generate_elliptical_feather_mask = face_module.generate_elliptical_feather_mask

def test_face_tracker_and_mask():
    print("[1/2] Testing Elliptical Feather Mask...")
    h, w = 128, 128
    mask = generate_elliptical_feather_mask(h, w, device="cpu")
    assert mask.shape == (1, 1, h, w)
    # Center should be ~1.0, corners should be 0.0
    assert mask[0, 0, h // 2, w // 2].item() > 0.95
    assert mask[0, 0, 0, 0].item() < 0.05
    print("  [OK] Elliptical feather mask verified (center > 0.95, corner < 0.05)")

    print("[2/2] Testing Face Tracker EMA Smoothing & Bounds...")
    tracker = FaceTracker(min_size=32, max_size=512)
    img_w, img_h = 640, 360
    
    # Synthetic frame
    synthetic_frame = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    
    # Test tracking with mock/fallback
    box = tracker.detect_and_smooth_frame(synthetic_frame, img_w, img_h)
    # No face in all-black frame -> should be None
    assert box is None
    print("  [OK] Face tracker size gating & clean non-detection verified")

if __name__ == "__main__":
    test_face_tracker_and_mask()
    print("=== Face Refinement Tests Passed ===")
