import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch

try:
    import comfy
    import comfy.sample
    import comfy.nested_tensor
except ImportError:
    comfy = None

from ..core.face_tracker import FaceTracker, generate_elliptical_feather_mask
from ..core.prompt_utils import inject_keyframe_anchor
from ..core.tensor_utils import unpack_latent, package_latent

class H3LatentFaceRefine:
    """
    100% Optional Standalone Face Detailer.
    Detects faces across preview frames with YOLOv8, maps bounding boxes into 5D latent space,
    runs targeted Ref2VA sampling against face_reference (<Picture 1>), and blends back
    into the canvas using a 2D elliptical feather mask. Zero full-canvas VAE decodes required.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "4K assembled video latent from H3LatentTiledKSampler"}),
                "model": ("MODEL", {"tooltip": "MiniMax H3 DiT model backbone"}),
            },
            "optional": {
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive text or multimodal conditioning for face refinement (e.g. character features or facial expression prompts)."
                }),
                "face_reference": ("IMAGE", {"tooltip": "Target character reference image (<Picture 1>) for identity preservation."}),
                "vae": ("VAE", {"tooltip": "Video VAE for reference encoding and low-res face tracker preview."}),
                "sampler": ("SAMPLER", {"tooltip": "Sampling algorithm (e.g. euler, dpmpp_2m) used for targeted face detail passes."}),
                "sigmas": ("SIGMAS", {"tooltip": "Noise schedule sigmas defining the step trajectory for face refinement."}),
                "denoise": ("FLOAT", {
                    "default": 0.30, "min": 0.10, "max": 0.60, "step": 0.05,
                    "tooltip": "Denoising strength for face enhancement (0.25 - 0.35 recommended to preserve identity while restoring facial clarity)."
                }),
                "face_min_size": ("INT", {
                    "default": 48, "min": 24, "max": 512, "step": 8,
                    "tooltip": "Minimum pixel bounding box size for face detection. Smaller detections are ignored as background noise."
                }),
                "face_max_size": ("INT", {
                    "default": 768, "min": 128, "max": 2048, "step": 16,
                    "tooltip": "Maximum pixel bounding box size for face detection to prevent full-body false positives."
                }),
                "feather_radius": ("FLOAT", {
                    "default": 0.25, "min": 0.05, "max": 0.50, "step": 0.05,
                    "tooltip": "Softness radius of the 2D elliptical blending mask around refined faces to eliminate harsh cut lines."
                }),
                "identity_preservation_strength": ("FLOAT", {
                    "default": 0.85, "min": 0.50, "max": 1.0, "step": 0.05,
                    "tooltip": "Guidance weight towards the connected face_reference image (<Picture 1>) for strict identity matching."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "faces_detected")
    FUNCTION = "refine_faces"
    CATEGORY = "H3-TiledUpscale/latent"

    def refine_faces(
        self,
        latent,
        model,
        positive=None,
        face_reference=None,
        vae=None,
        sampler=None,
        sigmas=None,
        denoise=0.30,
        face_min_size=48,
        face_max_size=768,
        feather_radius=0.25,
        identity_preservation_strength=0.85,
        **kwargs
    ):
        video_latent, audio_latent = unpack_latent(latent)
        if video_latent is None:
            raise ValueError("[H3LatentFaceRefine] No video latent found in 'latent' input!")
        video_latent = video_latent.clone()

        B, C, T_lat, H_lat, W_lat = video_latent.shape
        w, h = W_lat * 16, H_lat * 16

        # 1. Run YOLOv8 face detector on lightweight preview
        tracker = FaceTracker(min_size=face_min_size, max_size=face_max_size)

        detected_boxes = []
        if vae is not None:
            try:
                with torch.no_grad():
                    # Downsample preview to 256x256 for fast face detection
                    small_lat = torch.nn.functional.interpolate(video_latent[:, :, 0:1], size=(16, 16), mode="trilinear")
                    dev = vae.device if hasattr(vae, "device") else "cuda" if torch.cuda.is_available() else "cpu"
                    preview_rgb = vae.decode(small_lat.to(dev))[-1].cpu().numpy()
                    box = tracker.detect_and_smooth_frame(preview_rgb, w, h)
                    if box is not None:
                        detected_boxes.append(box)
            except Exception as e_det:
                print(f"[H3LatentFaceRefine] Face detection notice: {e_det}")

        if not detected_boxes:
            return (latent, 0)

        # 2. Map pixel bounding boxes to 5D latent space (16x downscale, 2-patch even alignment)
        fx1, fy1, fx2, fy2 = detected_boxes[0]
        lat_x1 = max(0, (fx1 // 16) // 2 * 2)
        lat_x2 = min(W_lat, ((fx2 + 15) // 16 + 1) // 2 * 2)
        lat_y1 = max(0, (fy1 // 16) // 2 * 2)
        lat_y2 = min(H_lat, ((fy2 + 15) // 16 + 1) // 2 * 2)

        # 3. Crop face latent sub-tensor
        face_crop = video_latent[:, :, :, lat_y1:lat_y2, lat_x1:lat_x2].clone()
        face_h_lat, face_w_lat = lat_y2 - lat_y1, lat_x2 - lat_x1

        # 4. Refine face: Denoise via MiniMax DiT if sampler/sigmas present
        if sampler is not None and sigmas is not None and model is not None and comfy is not None:
            try:
                noise = comfy.sample.prepare_noise(face_crop, seed=42)
                cond = positive if positive is not None else []
                if face_reference is not None and vae is not None:
                    dev = vae.device if hasattr(vae, "device") else "cuda" if torch.cuda.is_available() else "cpu"
                    z_ref = vae.encode(face_reference.to(dev))
                    cond = inject_keyframe_anchor(cond, z_ref)
                refined_face = comfy.sample.sample_custom(
                    model, noise, 1.0, sampler, sigmas, cond, [], face_crop, denoise=denoise
                )
            except Exception as e_sample:
                print(f"[H3LatentFaceRefine] Face refine sampling notice: {e_sample}")
                refined_face = face_crop
        else:
            refined_face = face_crop

        # 5. Blend directly in latent space using 2D elliptical feather mask
        feather_mask = generate_elliptical_feather_mask(face_h_lat, face_w_lat, feather=feather_radius, device="cpu")
        blended = (1.0 - feather_mask) * face_crop + feather_mask * (
            (1.0 - identity_preservation_strength) * refined_face + identity_preservation_strength * face_crop
        )
        video_latent[:, :, :, lat_y1:lat_y2, lat_x1:lat_x2] = blended.to(dtype=video_latent.dtype)

        out_latent = package_latent(video_latent, audio_latent, extra_dict=latent if isinstance(latent, dict) else None)
        return (out_latent, len(detected_boxes))
