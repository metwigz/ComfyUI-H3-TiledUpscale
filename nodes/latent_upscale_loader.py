import json
import os
import torch
import folder_paths
import comfy.utils
import comfy.model_management
import comfy.model_patcher
import comfy.ops

try:
    from comfy_api.latest import io
    USE_NEW_API = True
except ImportError:
    USE_NEW_API = False
    class io:
        class ComfyNode: pass
        class Schema: pass
        class NodeOutput:
            def __init__(self, *args, **kwargs):
                self.args = args

try:
    from comfy.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel
except Exception:
    HunyuanVideo15SRModel = None

try:
    from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
except Exception:
    LatentUpsampler = None

from ..core.minimax_h3_3d_resizer import (
    load_minimax_h3_3d_model,
    resolve_latent_upscale_model_path,
    LatentResizer3D,
    _detect_arch
)


def load_latent_upscale_model_any(model_name: str):
    """
    Safely loads any latent upscale model (Hunyuan, LTX, MiniMax H3, or other custom architecture).
    Never raises UnboundLocalError.
    """
    model_path = resolve_latent_upscale_model_path(model_name)
    if not model_path and folder_paths:
        model_path = folder_paths.get_full_path("latent_upscale_models", model_name)
    if not model_path:
        model_path = folder_paths.get_full_path_or_raise("latent_upscale_models", model_name)

    sd, metadata = comfy.utils.load_torch_file(model_path, safe_load=True, return_metadata=True)

    # Branch 1: HunyuanVideo 1.5 720p
    if "blocks.0.block.0.conv.weight" in sd and HunyuanVideo15SRModel is not None:
        config = {
            "in_channels": sd["in_conv.conv.weight"].shape[1],
            "out_channels": sd["out_conv.conv.weight"].shape[0],
            "hidden_channels": sd["in_conv.conv.weight"].shape[0],
            "num_blocks": len([k for k in sd.keys() if k.startswith("blocks.") and k.endswith(".block.0.conv.weight")]),
            "global_residual": False,
        }
        model_type = "720p"
        model = HunyuanVideo15SRModel(model_type, config)
        model.load_sd(sd)
        return model

    # Branch 2: HunyuanVideo 1.5 1080p
    elif "up.0.block.0.conv1.conv.weight" in sd and HunyuanVideo15SRModel is not None:
        sd = {key.replace("nin_shortcut", "nin_shortcut.conv", 1): value for key, value in sd.items()}
        config = {
            "z_channels": sd["conv_in.conv.weight"].shape[1],
            "out_channels": sd["conv_out.conv.weight"].shape[0],
            "block_out_channels": tuple(sd[f"up.{i}.block.0.conv1.conv.weight"].shape[0] for i in range(len([k for k in sd.keys() if k.startswith("up.") and k.endswith(".block.0.conv1.conv.weight")]))),
        }
        model_type = "1080p"
        model = HunyuanVideo15SRModel(model_type, config)
        model.load_sd(sd)
        return model

    # Branch 3: LTX-Video Latent Upsampler
    elif "post_upsample_res_blocks.0.conv2.bias" in sd and LatentUpsampler is not None:
        config = json.loads(metadata["config"]) if metadata and "config" in metadata else {}
        model = LatentUpsampler.from_config(config, operations=comfy.ops.disable_weight_init).to(dtype=comfy.model_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float32]))
        comfy.model_management.archive_model_dtypes(model)
        model_patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.unet_offload_device())
        model.load_state_dict(sd, assign=model_patcher.is_dynamic())
        return model_patcher

    # Branch 4: MiniMax H3 3D Latent Upscaler
    elif "conv_in.weight" in sd or any(k.startswith("upscaler.") for k in sd) or "minimax" in model_name.lower():
        device = comfy.model_management.get_torch_device()
        try:
            model_instance = load_minimax_h3_3d_model(model_path, device=device, precision="fp16")
            if model_instance is not None:
                return model_instance
        except Exception:
            pass

        try:
            cfg = _detect_arch(sd)
            with torch.device("meta"):
                m = LatentResizer3D(
                    in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
                    channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
                    temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"]
                )
            m.load_state_dict(sd, strict=True, assign=True)
            m.eval().requires_grad_(False)
            return m
        except Exception:
            return {
                "model": None,
                "state_dict": sd,
                "path": model_path,
                "metadata": metadata,
                "name": model_name
            }

    # Branch 5: Generic fallback for any other latent upscaler
    else:
        try:
            cfg = _detect_arch(sd)
            with torch.device("meta"):
                m = LatentResizer3D(
                    in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
                    channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
                    temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"]
                )
            m.load_state_dict(sd, strict=True, assign=True)
            m.eval().requires_grad_(False)
            return m
        except Exception:
            return {
                "state_dict": sd,
                "path": model_path,
                "metadata": metadata,
                "name": model_name
            }


class H3LatentUpscaleModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LatentUpscaleModelLoader",
            display_name="Load Latent Upscale Model (H3 Compatible)",
            category="model/loaders",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("latent_upscale_models"),
                    tooltip="Select a neural latent upscale model from models/latent_upscale_models (e.g. HunyuanVideo 1.5, LTX-Video, or MiniMax H3 3D latent upscaler)."
                ),
            ],
            outputs=[
                io.LatentUpscaleModel.Output(),
            ],
        )

    @classmethod
    def execute(cls, model_name: str) -> io.NodeOutput:
        model = load_latent_upscale_model_any(model_name)
        return io.NodeOutput(model)


class LatentUpscaleModelLoaderOverride(H3LatentUpscaleModelLoader):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentUpscaleModelLoader",
            display_name="Load Latent Upscale Model",
            category="model/loaders",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("latent_upscale_models"),
                    tooltip="Select a neural latent upscale model from models/latent_upscale_models (e.g. HunyuanVideo 1.5, LTX-Video, or MiniMax H3 3D latent upscaler)."
                ),
            ],
            outputs=[
                io.LatentUpscaleModel.Output(),
            ],
        )

