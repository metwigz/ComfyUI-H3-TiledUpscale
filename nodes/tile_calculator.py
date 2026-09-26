import math
from typing import Optional, Dict, Any, Tuple
import torch

from ..core.grid_utils import (
    parse_aspect_ratio,
    calculate_canvas_dimensions,
    calculate_optimal_tile_dimensions,
    compute_tile_intervals,
    compute_temporal_chunks,
    minimax_latents_to_frames
)
from ..core.tensor_utils import unpack_latent

class H3TileCalculator:
    """
    Independent helper node to calculate, inspect, and preview spatial tile grids,
    overlap percentages, memory estimates, and temporal chunking before running sampling.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_megapixels": ("FLOAT", {
                    "default": 0.92, "min": -1.0, "max": 10.0, "step": 0.01, "round": 0.01,
                    "tooltip": "Target tile resolution in megapixels. Setting <= 0 disables spatial tiling (single-tile full-frame mode)."
                }),
                "overlap_percent": ("FLOAT", {
                    "default": 0.25, "min": 0.05, "max": 0.50, "step": 0.01, "round": 0.01,
                    "tooltip": "Target fractional overlap between adjacent spatial tiles (e.g. 0.25 = 25%)."
                }),
                "tight_tile_overlap": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Minimizes tile dimensions to satisfy overlap_percent without redundant overlap inflation while strictly preserving canvas aspect ratio."
                }),
                "min_tile_megapixels": ("FLOAT", {
                    "default": 0.25, "min": 0.05, "max": 2.0, "step": 0.01, "round": 0.01,
                    "tooltip": "Semantic safety floor below which tiles will not shrink when tight_tile_overlap is enabled."
                }),
                "canvas_megapixels": ("FLOAT", {
                    "default": 8.29, "min": 0.1, "max": 64.0, "step": 0.01, "round": 0.01,
                    "tooltip": "Target canvas resolution (e.g. 8.29 MP for 4K UHD, 33.18 MP for 8K). Overridden if latent or image is connected."
                }),
                "aspect_ratio": ([
                    "Match_Input", "16:9 (Widescreen)", "9:16 (Vertical)",
                    "2.39:1 (Cinemascope)", "4:3 (Classic)", "1:1 (Square)", "21:9 (Ultrawide)", "Custom"
                ], {
                    "default": "Match_Input",
                    "tooltip": "Canvas aspect ratio to calculate grid against. 'Match_Input' uses connected latent/image dimensions; otherwise calculates based on the selected standard."
                }),
                "custom_aspect_ratio": ("STRING", {
                    "default": "16:9",
                    "tooltip": "Custom aspect ratio string (e.g. '16:9', '21:9', '4:3', '1.85:1') used when aspect_ratio is set to 'Custom'."
                }),
                "temporal_chunk_frames": ("INT", {
                    "default": 124, "min": -1, "max": 1000, "step": 1,
                    "tooltip": "Frame count per temporal chunk. Setting to -1 processes the whole video in one chunk."
                }),
                "temporal_blend_frames": ("INT", {
                    "default": 17, "min": 0, "max": 68, "step": 1,
                    "tooltip": "Overlap blend transition frames between adjacent temporal chunks (snaps to multiples of 17)."
                }),
                "total_frames": ("INT", {
                    "default": 124, "min": 1, "max": 10000, "step": 1,
                    "tooltip": "Total frames in the video. Overridden if latent or image is connected."
                }),
            },
            "optional": {
                "latent": ("LATENT", {"tooltip": "Optional video latent to auto-detect canvas dimensions and total frames."}),
                "image": ("IMAGE", {"tooltip": "Optional image tensor to auto-detect canvas dimensions and total frames."}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "INT", "INT", "INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = (
        "info_text", "tile_width", "tile_height", "num_tiles_w", "num_tiles_h",
        "total_spatial_tiles", "total_sampling_jobs", "tile_megapixels", "canvas_width", "canvas_height"
    )
    FUNCTION = "calculate_grid"
    OUTPUT_NODE = True
    CATEGORY = "H3-TiledUpscale/utilities"

    def calculate_grid(
        self,
        tile_megapixels=0.92,
        overlap_percent=0.25,
        tight_tile_overlap=False,
        min_tile_megapixels=0.25,
        canvas_megapixels=8.29,
        aspect_ratio="Match_Input",
        custom_aspect_ratio="16:9",
        temporal_chunk_frames=124,
        temporal_blend_frames=17,
        total_frames=124,
        latent=None,
        image=None,
        **kwargs
    ):
        # 1. Determine canvas dimensions and total frames
        src_w, src_h = 3840, 2160
        detected_source = "Manual Inputs"

        if latent is not None:
            try:
                samples, _ = unpack_latent(latent)
                if samples.ndim == 5:
                    # 5D latent [B, C, T, H/16, W/16]
                    _, _, t_lat, h_lat, w_lat = samples.shape
                    src_w, src_h = w_lat * 16, h_lat * 16
                    total_frames = minimax_latents_to_frames(t_lat)
                    detected_source = f"Auto-detected from LATENT ({src_w}x{src_h}, {total_frames} frames)"
            except Exception:
                pass
        elif image is not None and isinstance(image, torch.Tensor):
            # [B, H, W, C]
            total_frames = image.shape[0]
            src_h, src_w = image.shape[1], image.shape[2]
            detected_source = f"Auto-detected from IMAGE ({src_w}x{src_h}, {total_frames} frames)"

        if latent is None and image is None:
            ratio = parse_aspect_ratio(aspect_ratio, src_w, src_h, custom_aspect_ratio)
            canvas_w, canvas_h = calculate_canvas_dimensions(canvas_megapixels, ratio)
        else:
            canvas_w, canvas_h = src_w, src_h

        canvas_w = max(32, int(round(canvas_w / 32.0)) * 32)
        canvas_h = max(32, int(round(canvas_h / 32.0)) * 32)
        canvas_mp = (canvas_w * canvas_h) / 1e6
        canvas_ratio = canvas_w / canvas_h

        # 2. Compute Spatial Tile Dimensions
        if tile_megapixels <= 0:
            tile_w, tile_h = canvas_w, canvas_h
            num_tiles_w, num_tiles_h = 1, 1
            x_intervals = [(0, canvas_w)]
            y_intervals = [(0, canvas_h)]
        else:
            tile_w, tile_h, num_tiles_w, num_tiles_h = calculate_optimal_tile_dimensions(
                canvas_w, canvas_h, tile_megapixels, overlap_percent,
                tight_tile_overlap=tight_tile_overlap,
                min_megapixels=min_tile_megapixels
            )
            x_intervals = compute_tile_intervals(canvas_w, tile_w, overlap_percent)
            y_intervals = compute_tile_intervals(canvas_h, tile_h, overlap_percent)

        actual_tile_mp = (tile_w * tile_h) / 1e6
        total_spatial_tiles = len(x_intervals) * len(y_intervals)

        # 3. Compute Actual Overlaps
        actual_step_w = (canvas_w - tile_w) / max(1, num_tiles_w - 1) if num_tiles_w > 1 else 0
        actual_step_h = (canvas_h - tile_h) / max(1, num_tiles_h - 1) if num_tiles_h > 1 else 0
        actual_ov_w = (1.0 - (actual_step_w / tile_w)) * 100.0 if num_tiles_w > 1 else 0.0
        actual_ov_h = (1.0 - (actual_step_h / tile_h)) * 100.0 if num_tiles_h > 1 else 0.0

        # 4. Compute Temporal Chunks
        chunks = compute_temporal_chunks(
            total_frames,
            chunk_frames=temporal_chunk_frames,
            blend_frames=temporal_blend_frames
        )
        total_chunks = len(chunks)
        total_sampling_jobs = total_chunks * total_spatial_tiles

        # 5. Determine VRAM & Hardware Advice
        if actual_tile_mp <= 0.80:
            vram_tier = "12 GB - 16 GB VRAM (Mid-Range: RTX 4070 / 4070 Ti / 4080)"
        elif actual_tile_mp <= 1.25:
            vram_tier = "16 GB - 24 GB VRAM (High-End: RTX 3090 / 4090)"
        else:
            vram_tier = "24 GB - 32 GB+ VRAM (Workstation: RTX 5090 / 6000 Ada / A5000)"

        # 6. Format Comprehensive Report
        lines = [
            "==================================================================",
            "           MiniMax H3 Tiled Super-Resolution Preview              ",
            "==================================================================",
            f"Source Context:        {detected_source}",
            f"Canvas Dimensions:     {canvas_w} x {canvas_h} ({canvas_mp:.2f} Megapixels)",
            f"Canvas Aspect Ratio:   {canvas_ratio:.3f} : 1 ({canvas_w//16} x {canvas_h//16} latent tokens)",
            f"Video Duration:        {total_frames} frames (~{total_frames/24.0:.2f}s @ 24 fps)",
            "------------------------------------------------------------------",
            f"Target Tile Megapixels:{tile_megapixels:.2f} MP (Target Overlap: {overlap_percent*100.0:.1f}%)",
            f"Tight Tile Overlap:    {tight_tile_overlap} (Semantic Floor: {min_tile_megapixels:.2f} MP)",
            f"Calculated Tile Size:  {tile_w} x {tile_h} ({actual_tile_mp:.2f} MP)",
            f"Latent Tile Shape:     [1, 24, T_chunk, {tile_h//16}, {tile_w//16}]",
            f"Tile Aspect Ratio:     {tile_w/tile_h:.3f} : 1 (Preserved: {abs(tile_w/tile_h - canvas_ratio) < 0.05})",
            "------------------------------------------------------------------",
            f"Spatial Grid:          {num_tiles_w} columns x {num_tiles_h} rows = {total_spatial_tiles} tiles / chunk",
            f"Actual Overlap W:      {actual_ov_w:.1f}% (Step: {actual_step_w:.0f} px)",
            f"Actual Overlap H:      {actual_ov_h:.1f}% (Step: {actual_step_h:.0f} px)",
            "------------------------------------------------------------------",
            f"Temporal Chunking:     {total_chunks} chunk(s) (chunk_size: {temporal_chunk_frames}f, blend: {temporal_blend_frames}f)",
            f"Total Sampling Passes: {total_sampling_jobs} DiT inference jobs ({total_chunks} chunks x {total_spatial_tiles} tiles)",
            f"Hardware VRAM Tier:    {vram_tier}",
            "=================================================================="
        ]
        info_text = "\n".join(lines)

        return {
            "ui": {"text": [info_text]},
            "result": (
                info_text,
                int(tile_w),
                int(tile_h),
                int(num_tiles_w),
                int(num_tiles_h),
                int(total_spatial_tiles),
                int(total_sampling_jobs),
                float(actual_tile_mp),
                int(canvas_w),
                int(canvas_h)
            )
        }
