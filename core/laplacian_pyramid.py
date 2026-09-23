import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class LaplacianPyramidBlender:
    def __init__(self, num_levels: int = 3, device: str = "cuda"):
        self.levels = num_levels
        self.device = torch.device(device if (torch.cuda.is_available() and device == "cuda") else "cpu")
        
        # 1D Binomial Kernel [1, 4, 6, 4, 1] / 16
        kernel_1d = torch.tensor([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=torch.float32, device=self.device)
        kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).unsqueeze(0).unsqueeze(0)
        self.kernel_rgb = kernel_2d.repeat(3, 1, 1, 1)  # 3 channels (RGB)
        self.kernel_1ch = kernel_2d.clone()              # 1 channel (Mask / Weight)

    def _conv_gauss(self, img: torch.Tensor) -> torch.Tensor:
        channels = img.shape[1]
        kernel = self.kernel_rgb if channels == 3 else self.kernel_1ch
        if kernel.device != img.device:
            kernel = kernel.to(img.device)
        return F.conv2d(img, kernel, padding=2, groups=channels)

    def _downsample(self, img: torch.Tensor) -> torch.Tensor:
        blurred = self._conv_gauss(img)
        return blurred[:, :, ::2, ::2]

    def _upsample(self, img: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        upsampled = F.interpolate(img, size=target_shape, mode="bilinear", align_corners=False)
        return self._conv_gauss(upsampled)

    def build_pyramid(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Builds Laplacian pyramid from an input tensor [B, C, H, W]."""
        current = img
        pyr = []
        for _ in range(self.levels):
            down = self._downsample(current)
            up = self._upsample(down, (current.shape[2], current.shape[3]))
            lap = current - up
            pyr.append(lap)
            current = down
        pyr.append(current)  # Final low-frequency residual
        return pyr

    def build_gaussian_pyramid(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Builds Gaussian pyramid for mask/weight tensors [B, C, H, W]."""
        current = img
        pyr = [current]
        for _ in range(self.levels):
            down = self._downsample(current)
            pyr.append(down)
            current = down
        return pyr

    def reconstruct(self, pyr: List[torch.Tensor]) -> torch.Tensor:
        """Reconstructs image from Laplacian pyramid."""
        current = pyr[-1]
        for lap in reversed(pyr[:-1]):
            up = self._upsample(current, (lap.shape[2], lap.shape[3]))
            current = up + lap
        return torch.clamp(current, 0.0, 1.0)

    def blend_two_images(self, img_a: torch.Tensor, img_b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Multiscale blend between img_a and img_b using weight mask in range [0, 1].
        All inputs are [B, C, H, W].
        """
        pyr_a = self.build_pyramid(img_a)
        pyr_b = self.build_pyramid(img_b)
        pyr_m = self.build_gaussian_pyramid(mask)

        blended_pyr = []
        for i in range(len(pyr_a)):
            m = pyr_m[i]
            if m.shape[2:] != pyr_a[i].shape[2:]:
                m = F.interpolate(m, size=pyr_a[i].shape[2:], mode="bilinear", align_corners=False)
            blended = pyr_a[i] * (1.0 - m) + pyr_b[i] * m
            blended_pyr.append(blended)

        return self.reconstruct(blended_pyr)

    def apply_frequency_decoupling(self, base_canvas: torch.Tensor, high_detail_canvas: torch.Tensor) -> torch.Tensor:
        """
        Extracts high frequencies from high_detail_canvas and injects them onto base_canvas,
        guaranteeing zero color/luminance drift.
        """
        base_pyr = self.build_pyramid(base_canvas)
        detail_pyr = self.build_pyramid(high_detail_canvas)

        # Retain base canvas low-frequency residual
        fused_pyr = [detail_pyr[i] for i in range(self.levels)] + [base_pyr[-1]]
        return self.reconstruct(fused_pyr)


def generate_hann_weight_2d(
    h: int,
    w: int,
    overlap_y: Optional[int] = None,
    overlap_x: Optional[int] = None,
    fade_top: bool = True,
    fade_bottom: bool = True,
    fade_left: bool = True,
    fade_right: bool = True,
    ov_top: Optional[int] = None,
    ov_bottom: Optional[int] = None,
    ov_left: Optional[int] = None,
    ov_right: Optional[int] = None,
    device: str = "cpu",
    blend_mode: str = "Multiscale_Laplacian"
) -> torch.Tensor:
    """
    Generates a 2D raised-cosine (Hann) weight window [1, 1, 1, H, W] for a tile.
    Supports exact partition-of-unity overlap blending across directional seams
    (smoothly fading to 0.0 at internal boundaries with 1.0 at canvas edges),
    as well as full-tile symmetric raised cosine window fallback.
    """
    # 1. Directional Overlap Partition-of-Unity Mode
    has_directional_ov = any(v is not None for v in (ov_top, ov_bottom, ov_left, ov_right))
    has_uniform_ov = (overlap_y is not None and overlap_x is not None)

    if has_directional_ov or has_uniform_ov:
        top_tokens = ov_top if ov_top is not None else (overlap_y if (overlap_y and fade_top) else 0)
        bottom_tokens = ov_bottom if ov_bottom is not None else (overlap_y if (overlap_y and fade_bottom) else 0)
        left_tokens = ov_left if ov_left is not None else (overlap_x if (overlap_x and fade_left) else 0)
        right_tokens = ov_right if ov_right is not None else (overlap_x if (overlap_x and fade_right) else 0)

        if not fade_top: top_tokens = 0
        if not fade_bottom: bottom_tokens = 0
        if not fade_left: left_tokens = 0
        if not fade_right: right_tokens = 0

        def make_1d_ramp(length: int, ov_start: int, ov_end: int) -> torch.Tensor:
            ramp = torch.ones(length, dtype=torch.float32, device=device)
            if ov_start > 0:
                act_start = min(ov_start, length)
                t_in = torch.linspace(0, torch.pi, act_start, device=device)
                ramp[:act_start] = 0.5 - 0.5 * torch.cos(t_in)
            if ov_end > 0:
                act_end = min(ov_end, length)
                t_out = torch.linspace(0, torch.pi, act_end, device=device)
                ramp[-act_end:] = 0.5 + 0.5 * torch.cos(t_out)
            return ramp

        ramp_y = make_1d_ramp(h, top_tokens, bottom_tokens)
        ramp_x = make_1d_ramp(w, left_tokens, right_tokens)
        weight_2d = (ramp_y[:, None] * ramp_x[None, :]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return weight_2d.clamp(min=0.0)

    # 2. Standalone Full-Tile Symmetric Window Fallback
    if blend_mode == "Linear_Feather":
        ty = torch.linspace(0.0, 1.0, (h + 1) // 2, device=device)
        ry = torch.cat([ty, ty.flip(0)[:h // 2]])
        tx = torch.linspace(0.0, 1.0, (w + 1) // 2, device=device)
        rx = torch.cat([tx, tx.flip(0)[:w // 2]])
    else:
        # Full symmetric 2*pi Hann cosine window (rises to 1.0 in center, smoothly drops to 0 at both edges)
        t_y = torch.linspace(0, 2.0 * torch.pi, h + 2, device=device)[1:-1]
        ry = 0.5 - 0.5 * torch.cos(t_y)
        t_x = torch.linspace(0, 2.0 * torch.pi, w + 2, device=device)[1:-1]
        rx = 0.5 - 0.5 * torch.cos(t_x)

    weight_2d = (ry[:, None] * rx[None, :]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return weight_2d.clamp(min=1e-4)
