"""
MiniMax H3 3D Neural Latent Upscaler Core Implementation
Replicates the learned 3D latent upscaling architecture, auto-downloader,
and 24-channel normalization from 'comfyui-minimax-h3-latent-upscaler' (LBH-123-AI / xmarre).
"""
import os
import glob
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Dict, Any, Tuple

try:
    import folder_paths
except ImportError:
    folder_paths = None

# ==========================================
# MiniMax H3 24-Channel Latent Normalization
# ==========================================
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]

def make_norm_tensors(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std

# ==========================================
# 3D Neural Architecture Components
# ==========================================
def normalization(channels: int) -> nn.Module:
    return nn.GroupNorm(32, channels)

def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module

class AttnBlock3D(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = rearrange(self.q(h), "b c t h w -> b 1 (t h w) c")
        k = rearrange(self.k(h), "b c t h w -> b 1 (t h w) c")
        v = rearrange(self.v(h), "b c t h w -> b 1 (t h w) c")
        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "b 1 (t h w) c -> b c t h w", t=x.shape[2], h=x.shape[3], w=x.shape[4])
        return x + self.proj_out(h)

class ResBlockEmb3D(nn.Module):
    def __init__(self, channels: int, emb_channels: int, dropout: float = 0.0, out_channels: Optional[int] = None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h

class TemporalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h

class LatentResizer3D(nn.Module):
    """3D Latent Resizer neural network matching MiniMax H3 3D upscaler."""
    def __init__(self, in_channels: int = 24, in_blocks: int = 12, out_blocks: int = 12,
                 channels: int = 512, dropout: float = 0.1, attn: bool = False,
                 temporal_every: int = 2, temporal_kernel: int = 5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(AttnBlock3D(channels))
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(AttnBlock3D(channels))
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, scale: Optional[float] = None, target_size: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device
        ).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        # Trilinear feature-space interpolation
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x

# ==========================================
# Known Hugging Face Models & Auto-Download
# ==========================================
KNOWN_HF_MODELS = {
    "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors": {
        "repo_id": "LBH-123-AI/Minimax_h3_latent_Upscaler",
        "subfolder": "minimax_h3_latent_upscaler_3d_conv_v1",
        "filename": "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
    },
    "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors": {
        "repo_id": "LBH-123-AI/Minimax_h3_latent_Upscaler",
        "subfolder": "minimax_h3_latent_upscaler_3d_conv_v1",
        "filename": "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
    },
}

# Register latent_upscale_models folder in folder_paths if not present
if folder_paths and "latent_upscale_models" not in folder_paths.folder_names_and_paths:
    try:
        folder_paths.add_model_folder_path(
            "latent_upscale_models",
            os.path.join(folder_paths.models_dir, "latent_upscale_models")
        )
    except Exception:
        pass

def get_latent_upscale_models_dir() -> str:
    if folder_paths and "latent_upscale_models" in folder_paths.folder_names_and_paths:
        dirs = folder_paths.get_folder_paths("latent_upscale_models")
        if dirs:
            return dirs[0]
    return os.path.join(os.getcwd(), "ComfyUI", "models", "latent_upscale_models")

def resolve_latent_upscale_model_path(name_or_path: str) -> Optional[str]:
    """Resolves model filename or path across all ComfyUI search locations."""
    if not name_or_path:
        return None
    if os.path.isfile(name_or_path):
        return os.path.abspath(name_or_path)

    if folder_paths and "latent_upscale_models" in folder_paths.folder_names_and_paths:
        resolved = folder_paths.get_full_path("latent_upscale_models", name_or_path)
        if resolved and os.path.isfile(resolved):
            return resolved

        for d in folder_paths.get_folder_paths("latent_upscale_models"):
            cand = os.path.join(d, name_or_path)
            if os.path.isfile(cand):
                return cand
            cand_base = os.path.join(d, os.path.basename(name_or_path))
            if os.path.isfile(cand_base):
                return cand_base
            if os.path.isdir(d):
                for root, _, fnames in os.walk(d):
                    if os.path.basename(name_or_path) in fnames:
                        match = os.path.join(root, os.path.basename(name_or_path))
                        if os.path.isfile(match):
                            return match

    return None


# ==========================================
# Model Loading, Caching & Scanning
# ==========================================
MODEL_CACHE: Dict[str, LatentResizer3D] = {}
_PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def scan_latent_upscale_models() -> list:
    """Returns list of available model files found on disk in latent_upscale_models folder(s)."""
    files = []
    if folder_paths and "latent_upscale_models" in folder_paths.folder_names_and_paths:
        try:
            flist = folder_paths.get_filename_list("latent_upscale_models")
            if flist:
                files.extend(flist)
        except Exception:
            pass

        for d in folder_paths.get_folder_paths("latent_upscale_models"):
            if os.path.exists(d):
                for ext in ("*.pth", "*.safetensors", "*.pt", "*.bin"):
                    files.extend([os.path.basename(f) for f in glob.glob(os.path.join(d, ext))])
                for sub in glob.glob(os.path.join(d, "*", "*")):
                    if sub.endswith((".safetensors", ".pth", ".pt", ".bin")):
                        files.append(os.path.relpath(sub, d))

    names = sorted(set(f for f in files if f != "put_latent_upscale_models_here"))
    if not names:
        names = ["minimax_h3_latent_upscaler_3d_fp16.safetensors"]
    return names

def _detect_arch(sd: dict) -> dict:
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
        "dropout": 0.1, "attn": False, "temporal_every": 2, "temporal_kernel": 5,
    }
    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids, out_ids = set(), set()
    temporal_in_indices, temporal_out_indices = set(), set()
    for k in sd.keys():
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m: in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m: out_ids.add(int(m.group(1)))
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_in_indices.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_out_indices.add(int(m.group(1)))

    if in_ids: cfg["in_blocks"] = len(in_ids)
    if out_ids: cfg["out_blocks"] = len(out_ids)

    if temporal_in_indices or temporal_out_indices:
        cfg["temporal_every"] = 2
        for k in sd.keys():
            if 'dwconv.weight' in k and k.endswith('dwconv.weight'):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0

    cfg["attn"] = False  # Forced off at inference for speed/stability
    return cfg

def _load_raw_sd(path: str, device: torch.device, dtype: torch.dtype) -> dict:
    if path.endswith('.safetensors'):
        from safetensors import safe_open
        with safe_open(path, framework='pt', device=str(device)) as f:
            keys = list(f.keys())
            has_prefix = any(k.startswith("upscaler.") for k in keys)
            sd = {}
            for k in keys:
                if has_prefix and not k.startswith("upscaler."):
                    continue
                out_key = k[len("upscaler."):] if has_prefix else k
                tensor = f.get_tensor(k)
                if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.dtype != dtype:
                    tensor = tensor.to(dtype=dtype)
                sd[out_key] = tensor
        return sd

    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return {k: (v.to(dtype=dtype) if torch.is_tensor(v) and v.is_floating_point() and v.dtype != dtype else v) for k, v in sd.items()}

def load_minimax_h3_3d_model(name_or_path: str, device: torch.device, precision: str = "fp16", **kwargs) -> LatentResizer3D:
    """Loads and caches a MiniMax H3 3D latent upscaler model strictly from local storage."""
    cache_key = f"{name_or_path}::{device}::{precision}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key].to(device)

    # Locate model file strictly from disk
    path = resolve_latent_upscale_model_path(name_or_path)
    if not path or not os.path.isfile(path):
        models_dir = get_latent_upscale_models_dir()
        search_dirs = [models_dir]
        if folder_paths and "latent_upscale_models" in folder_paths.folder_names_and_paths:
            search_dirs = folder_paths.get_folder_paths("latent_upscale_models")
        raise FileNotFoundError(
            f"[H3-TiledUpscale] MiniMax H3 latent upscaler model not found: '{name_or_path}'.\n"
            f"Searched directories: {search_dirs}.\n"
            f"Please ensure your model file exists in 'ComfyUI/models/latent_upscale_models/'."
        )

    dtype = _PRECISION_DTYPES.get(precision, torch.float16)
    sd = _load_raw_sd(path, device, dtype)
    if "conv_in.weight" not in sd and "conv_in.bias" not in sd:
        raise ValueError(
            f"[H3-TiledUpscale] Selected model '{name_or_path}' is not a MiniMax H3 latent upscaler!\n"
            f"Expected MiniMax H3 latent upscaler weights (e.g. 'minimax_h3_latent_upscaler_3d_fp16.safetensors')."
        )

    cfg = _detect_arch(sd)
    with torch.device("meta"):
        model = LatentResizer3D(
            in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
            channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
            temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"]
        )
    model.load_state_dict(sd, strict=True, assign=True)
    model.eval().requires_grad_(False)
    del sd

    MODEL_CACHE[cache_key] = model
    print(f"[H3-TiledUpscale] Loaded MiniMax H3 3D latent upscaler: {os.path.basename(path)} ({precision}, {device})")
    return model

def upscale_minimax_h3_3d_latent(
    video_latent: torch.Tensor,
    target_lat_h: int,
    target_lat_w: int,
    model_name: Optional[str] = None,
    model: Optional[LatentResizer3D] = None,
    device: str = "cuda",
    precision: str = "fp16",
    offload_after_upscale: bool = True,
    **kwargs
) -> torch.Tensor:
    """
    Executes learned 3D neural latent upscaling on 5D video latents [B, 24, T, H, W].
    """
    if video_latent.ndim != 5:
        raise ValueError(f"Expected 5D video latent [B, C, T, H, W], got shape {video_latent.shape}")
    B, C, T, src_h, src_w = video_latent.shape
    if C != 24:
        raise ValueError(f"Expected 24 channels for MiniMax H3 video latent, got {C}")

    if target_lat_h == src_h and target_lat_w == src_w:
        return video_latent.clone()

    exec_device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    dtype = _PRECISION_DTYPES.get(precision, torch.float16)
    orig_dtype = video_latent.dtype

    if isinstance(model, dict):
        if "model" in model and model["model"] is not None:
            model = model["model"]
        elif "path" in model and model["path"]:
            model = load_minimax_h3_3d_model(model["path"], exec_device, precision)
        elif "state_dict" in model and model["state_dict"] is not None:
            sd = model["state_dict"]
            cfg = _detect_arch(sd)
            with torch.device("meta"):
                m = LatentResizer3D(
                    in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
                    channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
                    temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"]
                )
            m.load_state_dict(sd, strict=True, assign=True)
            m.eval().requires_grad_(False)
            model = m

    if model is None:
        if not model_name or model_name in ("None", ""):
            avail = scan_latent_upscale_models()
            model_name = avail[0] if avail else "minimax_h3_latent_upscaler_3d_fp16.safetensors"
        model = load_minimax_h3_3d_model(model_name, exec_device, precision)
    elif hasattr(model, "to"):
        model = model.to(device=exec_device, dtype=dtype)

    scale = ((target_lat_h / src_h) + (target_lat_w / src_w)) / 2.0
    work = video_latent.to(device=exec_device, dtype=dtype, copy=True)
    mean, std = make_norm_tensors(exec_device, dtype)

    with torch.inference_mode():
        work.sub_(mean).div_(std)
        output = model(work, scale=scale, target_size=(T, target_lat_h, target_lat_w))
        del work
        output.mul_(std).add_(mean)

    output = output.to(device=video_latent.device, dtype=orig_dtype)

    if exec_device.type == "cuda":
        if offload_after_upscale and hasattr(model, "to"):
            model.to("cpu")
            torch.cuda.empty_cache()

    return output
