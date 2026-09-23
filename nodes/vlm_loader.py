import os
from pathlib import Path
import folder_paths
from ..core.vlm_engine import load_local_vlm
from ..core.schemas import VLMContainer

def get_available_vlm_models():
    defaults = [
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "microsoft/Florence-2-large",
        "microsoft/Florence-2-base",
        "Custom/Local_Path"
    ]
    vlm_dir = os.path.join(folder_paths.models_dir, "vlm")
    if os.path.exists(vlm_dir):
        try:
            for item in os.listdir(vlm_dir):
                full_p = os.path.join(vlm_dir, item)
                if os.path.isdir(full_p) and item not in defaults:
                    defaults.insert(0, item)
        except Exception:
            pass
    return defaults

class H3VLMModelLoader:
    """
    Loads a local Vision-Language Model (e.g. Qwen2.5-VL or Florence-2)
    and passes the loaded model container into downstream chunk analysis nodes.
    """
    @classmethod
    def INPUT_TYPES(cls):
        models = get_available_vlm_models()
        return {
            "required": {
                "model_name": (models, {
                    "default": models[0] if "Qwen" in models[0] else "Qwen/Qwen2.5-VL-3B-Instruct",
                    "tooltip": "Select a local Vision-Language Model to describe video chunks. Checks models/vlm, models/LLM, or local HuggingFace cache."
                }),
                "precision": (["bf16", "fp16", "fp8_e4m3fn", "fp32"], {
                    "default": "bf16",
                    "tooltip": "Weight precision: 'bf16' (recommended for Ampere/Ada/Blackwell), 'fp16' (standard GPU), or 'fp8_e4m3fn' for lowest VRAM."
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device destination for VLM inference: 'cuda' (fast GPU inference) or 'cpu' (0 GB GPU VRAM overhead)."
                }),
            },
            "optional": {
                "local_path_override": ("STRING", {
                    "default": "",
                    "tooltip": "Optional absolute path or folder name in models/vlm to load local custom weights directly."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "H3-TiledUpscale"

    def load_model(
        self,
        model_name: str,
        precision: str,
        device: str,
        local_path_override: str = ""
    ):
        vlm_container = load_local_vlm(
            model_name=model_name,
            local_path_override=local_path_override,
            precision=precision,
            device=device
        )
        return (vlm_container,)
