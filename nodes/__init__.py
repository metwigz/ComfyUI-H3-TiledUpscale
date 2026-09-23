"""
Node class exports for ComfyUI-H3-TiledUpscale.
"""

from .canvas_preparer import H3LatentCanvasUpscale
from .reference_bundle import H3ReferenceAssetBundle
from .chunk_describer import H3ChunkVideoDescriber
from .latent_tiled_ksampler import H3LatentTiledKSampler
from .face_refine import H3LatentFaceRefine
from .video_save import H3VideoSave
from .audio_decode import H3AudioVAEDecode
from .tile_calculator import H3TileCalculator
from .latent_loader import H3LoadLatentFromPath, H3SaveLatent
from .vlm_loader import H3VLMModelLoader
from .latent_upscale_loader import H3LatentUpscaleModelLoader, LatentUpscaleModelLoaderOverride

__all__ = [
    "H3LatentCanvasUpscale",
    "H3ReferenceAssetBundle",
    "H3ChunkVideoDescriber",
    "H3LatentTiledKSampler",
    "H3LatentFaceRefine",
    "H3VideoSave",
    "H3AudioVAEDecode",
    "H3TileCalculator",
    "H3LoadLatentFromPath",
    "H3SaveLatent",
    "H3VLMModelLoader",
    "H3LatentUpscaleModelLoader",
    "LatentUpscaleModelLoaderOverride",
]
