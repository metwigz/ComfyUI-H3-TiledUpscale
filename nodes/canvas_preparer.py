import os
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
import torch

try:
    import comfy
    import comfy.nested_tensor
except ImportError:
    comfy = None

from ..core.grid_utils import parse_aspect_ratio, calculate_canvas_dimensions
from ..core.minimax_h3_3d_resizer import upscale_minimax_h3_3d_latent
from ..core.tensor_utils import unpack_latent, package_latent

class H3LatentCanvasUpscale:
    """
    Expands a 5D input video latent canvas [1, 24, T, H/16, W/16] to 4K UHD or 8K target dimensions
    directly in latent space (0 intermediate VAE calls), preserving synchronized audio latents.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "5D input video latent [1, 24, T, H/16, W/16]"}),
                "output_megapixels": ("FLOAT", {
                    "default": 8.29, "min": 1.0, "max": 64.0, "step": 0.1, "round": 0.01,
                    "tooltip": "Target resolution in megapixels (e.g. 8.29 MP for 4K UHD, 33.18 MP for 8K)."
                }),
                "output_aspect_ratio": ([
                    "Match_Input", "16:9 (Widescreen)", "9:16 (Vertical)", 
                    "2.39:1 (Cinemascope)", "4:3 (Classic)", "1:1 (Square)", "21:9 (Ultrawide)", "Custom"
                ], {
                    "default": "Match_Input",
                    "tooltip": "Aspect ratio for the upscaled canvas. 'Match_Input' preserves the exact aspect ratio of the source latent, or pick a cinema/social standard."
                }),
                "framing_mode": (["Crop_To_Fill", "Pad_Letterbox", "Stretch_Anamorphic"], {
                    "default": "Crop_To_Fill",
                    "tooltip": "How to fit content when aspect ratios differ: 'Crop_To_Fill' (center crops excess), 'Pad_Letterbox' (adds letterbox bars), or 'Stretch_Anamorphic' (rescales coordinates non-uniformly)."
                }),
                "upscale_method": ([
                    "Minimax H3 Latent Upscaler (3D)",
                    "Latent_Trilinear (5D)",
                    "Latent_Bicubic (Spatial)",
                    "Latent_Bilinear (Spatial)",
                    "Latent_Nearest_Exact",
                    "Bypass (Pre-upscaled)"
                ], {
                    "default": "Minimax H3 Latent Upscaler (3D)",
                    "tooltip": "Base canvas upscale algorithm: 'Minimax H3 Latent Upscaler (3D)' gives highest quality via learned 3D neural latent upscaling."
                }),
                "neural_precision": (["fp16", "bf16", "fp32"], {
                    "default": "fp16",
                    "tooltip": "Precision for MiniMax H3 3D neural latent upscaler (fp16 is optimal for quality, speed, and VRAM)."
                }),
                "offload_after_upscale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Offload neural 3D upscaler to CPU and clear VRAM cache after base canvas preparation."
                }),
            },
            "optional": {
                "latent_upscale_model": ("LATENT_UPSCALE_MODEL", {
                    "tooltip": "MiniMax H3 3D neural latent upscale model. REQUIRED when upscale_method is 'Minimax H3 Latent Upscaler (3D)'."
                }),
                "custom_aspect_ratio": ("STRING", {
                    "default": "16:10",
                    "tooltip": "Arbitrary aspect ratio (e.g. '16:10', '4:5', '2.35:1'). Active only when output_aspect_ratio is set to 'Custom'."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "upscale_canvas"
    CATEGORY = "H3-TiledUpscale/latent"

    def upscale_canvas(
        self,
        latent: Dict[str, Any],
        output_megapixels: float = 8.29,
        output_aspect_ratio: str = "Match_Input",
        framing_mode: str = "Crop_To_Fill",
        upscale_method: str = "Minimax H3 Latent Upscaler (3D)",
        neural_precision: str = "fp16",
        offload_after_upscale: bool = True,
        latent_upscale_model: Optional[Any] = None,
        custom_aspect_ratio: str = "16:10",
        **kwargs
    ):
        # 1. Unpack video and audio latents via universal helper
        video_latent, audio_latent = unpack_latent(latent)
        if video_latent is None:
            raise ValueError("[H3LatentCanvasUpscale] No video latent found in 'latent' input!")

        B, C, T, H_lat, W_lat = video_latent.shape
        src_h, src_w = H_lat * 16, W_lat * 16

        # 2. Compute 4K target dimensions (multiples of 32 for 2x2 DiT patch)
        ratio = parse_aspect_ratio(output_aspect_ratio, src_w, src_h, custom_aspect_ratio)
        canvas_w, canvas_h = calculate_canvas_dimensions(output_megapixels, ratio)
        canvas_w = max(32, int(round(canvas_w / 32.0)) * 32)
        canvas_h = max(32, int(round(canvas_h / 32.0)) * 32)
        target_lat_w = canvas_w // 16
        target_lat_h = canvas_h // 16

        # 3. Execute Selected Latent Upscale Method
        if "Minimax" in upscale_method or "3D" in upscale_method:
            if latent_upscale_model is None:
                raise ValueError("MiniMax H3 Latent Upscaler (3D) requires a loaded 'latent_upscale_model'!")
            if isinstance(latent_upscale_model, dict):
                model_to_use = latent_upscale_model.get("model", latent_upscale_model)
            else:
                model_to_use = getattr(latent_upscale_model, "model", latent_upscale_model)
            upscaled = upscale_minimax_h3_3d_latent(
                video_latent=video_latent,
                target_lat_h=target_lat_h,
                target_lat_w=target_lat_w,
                model=model_to_use,
                device="cuda" if torch.cuda.is_available() else "cpu",
                precision=neural_precision,
                offload_after_upscale=offload_after_upscale
            )
        elif "Bypass" in upscale_method:
            if H_lat == target_lat_h and W_lat == target_lat_w:
                upscaled = video_latent
            else:
                with torch.no_grad():
                    upscaled = torch.nn.functional.interpolate(
                        video_latent.float(), size=(T, target_lat_h, target_lat_w),
                        mode="trilinear", align_corners=False
                    )
        elif "Bicubic" in upscale_method:
            with torch.no_grad():
                flat = video_latent.float().permute(0, 2, 1, 3, 4).reshape(B * T, C, H_lat, W_lat)
                scaled = torch.nn.functional.interpolate(
                    flat, size=(target_lat_h, target_lat_w), mode="bicubic", align_corners=False
                )
                upscaled = scaled.view(B, T, C, target_lat_h, target_lat_w).permute(0, 2, 1, 3, 4)
        elif "Bilinear" in upscale_method:
            with torch.no_grad():
                flat = video_latent.float().permute(0, 2, 1, 3, 4).reshape(B * T, C, H_lat, W_lat)
                scaled = torch.nn.functional.interpolate(
                    flat, size=(target_lat_h, target_lat_w), mode="bilinear", align_corners=False
                )
                upscaled = scaled.view(B, T, C, target_lat_h, target_lat_w).permute(0, 2, 1, 3, 4)
        elif "Nearest" in upscale_method:
            with torch.no_grad():
                upscaled = torch.nn.functional.interpolate(
                    video_latent.float(), size=(T, target_lat_h, target_lat_w), mode="nearest-exact"
                )
        else:  # Latent_Trilinear (5D)
            with torch.no_grad():
                upscaled = torch.nn.functional.interpolate(
                    video_latent.float(), size=(T, target_lat_h, target_lat_w),
                    mode="trilinear", align_corners=False
                )

        target_dtype = video_latent.dtype if video_latent.dtype in (torch.float16, torch.bfloat16) else torch.float32
        upscaled = upscaled.to(target_dtype)

        # 4. Return standard compliant LATENT dict preserving audio latent
        out_latent = package_latent(upscaled, audio_latent)
        return (out_latent, int(canvas_w), int(canvas_h))
