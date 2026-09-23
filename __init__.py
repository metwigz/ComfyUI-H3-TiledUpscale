"""
ComfyUI-H3-TiledUpscale
Native 5D Latent Tiled Video Super-Resolution using MiniMax H3 Ref2VA under 16GB-24GB VRAM.
"""

from .nodes.canvas_preparer import H3LatentCanvasUpscale
from .nodes.reference_bundle import H3ReferenceAssetBundle
from .nodes.chunk_describer import H3ChunkVideoDescriber
from .nodes.latent_tiled_ksampler import H3LatentTiledKSampler
from .nodes.face_refine import H3LatentFaceRefine
from .nodes.video_save import H3VideoSave
from .nodes.tile_calculator import H3TileCalculator
from .nodes.audio_decode import H3AudioVAEDecode
from .nodes.latent_loader import H3LoadLatentFromPath, H3SaveLatent
from .nodes.vlm_loader import H3VLMModelLoader
from .nodes.latent_upscale_loader import H3LatentUpscaleModelLoader, LatentUpscaleModelLoaderOverride, load_latent_upscale_model_any

NODE_CLASS_MAPPINGS = {
    # Core Redesigned Pipeline (100% Standard Types)
    "H3LatentCanvasUpscale": H3LatentCanvasUpscale,
    "H3ReferenceAssetBundle": H3ReferenceAssetBundle,
    "H3ChunkVideoDescriber": H3ChunkVideoDescriber,
    "H3LatentTiledKSampler": H3LatentTiledKSampler,
    "H3LatentFaceRefine": H3LatentFaceRefine,
    "H3VideoSave": H3VideoSave,

    # Independent Preview & Auxiliary Utilities
    "H3TileCalculator": H3TileCalculator,
    "H3AudioVAEDecode": H3AudioVAEDecode,
    "H3LoadLatentFromPath": H3LoadLatentFromPath,
    "H3SaveLatent": H3SaveLatent,
    "H3VLMModelLoader": H3VLMModelLoader,
    "H3LatentUpscaleModelLoader": H3LatentUpscaleModelLoader,
    "LatentUpscaleModelLoader": LatentUpscaleModelLoaderOverride,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LatentCanvasUpscale": "H3 Latent Canvas Upscale (4K)",
    "H3ReferenceAssetBundle": "H3 Reference Asset Bundle",
    "H3ChunkVideoDescriber": "H3 Chunk Video Describer (VLM)",
    "H3LatentTiledKSampler": "H3 Latent Tiled KSampler",
    "H3LatentFaceRefine": "H3 Latent Face Refine",
    "H3VideoSave": "H3 Video Save & Multiplexer",
    "H3TileCalculator": "H3 Tile Calculator & Grid Preview",
    "H3AudioVAEDecode": "H3 Audio VAE Decode",
    "H3LoadLatentFromPath": "H3 Load Latent From Path",
    "H3SaveLatent": "H3 Save Latent",
    "H3VLMModelLoader": "H3 VLM Model Loader",
    "H3LatentUpscaleModelLoader": "Load Latent Upscale Model (H3 / Hunyuan / LTX)",
}

WEB_DIRECTORY = "./js"

def _patch_latent_upscale_model_loader():
    try:
        import sys
        import nodes

        targets = set()
        if hasattr(nodes, "NODE_CLASS_MAPPINGS") and "LatentUpscaleModelLoader" in nodes.NODE_CLASS_MAPPINGS:
            targets.add(nodes.NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"])
        for mod_name in ("nodes_hunyuan", "comfy_extras.nodes_hunyuan"):
            if mod_name in sys.modules:
                mod = sys.modules[mod_name]
                if hasattr(mod, "LatentUpscaleModelLoader"):
                    targets.add(getattr(mod, "LatentUpscaleModelLoader"))

        for target_cls in targets:
            orig_execute = getattr(target_cls, "execute", None)

            @classmethod
            def safe_execute(cls, *args, **kwargs):
                model_name = kwargs.get("model_name")
                if not model_name and args:
                    for a in args:
                        if isinstance(a, str):
                            model_name = a
                            break

                try:
                    if orig_execute:
                        return orig_execute(*args, **kwargs)
                except UnboundLocalError:
                    pass
                except Exception as e:
                    if "unassociated" in str(e) or "local variable 'model'" in str(e):
                        pass
                    else:
                        raise

                model = load_latent_upscale_model_any(model_name)
                try:
                    from comfy_api.latest import io
                    return io.NodeOutput(model)
                except Exception:
                    return (model,)

            target_cls.execute = safe_execute

        if hasattr(nodes, "NODE_CLASS_MAPPINGS"):
            nodes.NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"] = LatentUpscaleModelLoaderOverride

    except Exception as e:
        import logging
        logging.warning(f"[H3-TiledUpscale] Failed to patch LatentUpscaleModelLoader: {e}")

_patch_latent_upscale_model_loader()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
