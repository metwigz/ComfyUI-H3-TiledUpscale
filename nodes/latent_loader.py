import os
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import safetensors.torch

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import comfy.nested_tensor
    NestedTensor = comfy.nested_tensor.NestedTensor
except ImportError:
    class NestedTensor:
        def __init__(self, tensors):
            self.tensors = tensors

from ..core.tensor_utils import unpack_latent, package_latent

class H3LoadLatentFromPath:
    """
    Loads a ComfyUI LATENT dictionary from an arbitrary file path or filename.
    Supports absolute paths, paths relative to ComfyUI root, input directory, or output directory.
    Seamlessly unpacks joint MiniMax H3 Video + Audio latents into NestedTensor([video, audio]).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "File path or filename of the .latent file to load. Can be an absolute path (e.g. C:/.../file.latent), relative to ComfyUI root, input, or output directory."
                }),
            },
            "optional": {
                "audio_latent_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional path to separate audio .latent file if audio was saved separately from the video latent (e.g. via ComfyUI default SaveLatent)."
                }),
                "load_device": (["cpu", "cuda"], {
                    "default": "cpu",
                    "tooltip": "Device to load the latent tensor into ('cpu' preserves GPU VRAM; 'cuda' loads directly onto GPU)."
                })
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "latent_info")
    FUNCTION = "load_latent"
    CATEGORY = "H3-TiledUpscale"

    @classmethod
    def IS_CHANGED(cls, latent_path: str, audio_latent_path: str = "", load_device: str = "cpu"):
        resolved_v = cls._resolve_path(latent_path)
        resolved_a = cls._resolve_path(audio_latent_path) if audio_latent_path else None
        tag_v = f"{resolved_v.stat().st_mtime}_{resolved_v.stat().st_size}" if (resolved_v and resolved_v.exists()) else latent_path
        tag_a = f"_{resolved_a.stat().st_mtime}_{resolved_a.stat().st_size}" if (resolved_a and resolved_a.exists()) else ""
        return f"{tag_v}{tag_a}"

    @classmethod
    def _resolve_path(cls, path_str: str) -> Path:
        if not path_str or not path_str.strip():
            return None
        
        cleaned = path_str.strip().strip('"\'')
        candidates = []

        # 1. Direct path (absolute or relative to CWD)
        p_direct = Path(cleaned)
        candidates.append(p_direct)
        if not cleaned.endswith(".latent"):
            candidates.append(Path(cleaned + ".latent"))

        # 2. Check relative to ComfyUI base path
        if folder_paths and hasattr(folder_paths, "base_path"):
            try:
                base_dir = Path(folder_paths.base_path)
                candidates.append(base_dir / cleaned)
                if not cleaned.endswith(".latent"):
                    candidates.append(base_dir / (cleaned + ".latent"))
            except Exception:
                pass

        # 3. Check in ComfyUI output directory (and strip leading 'output/' if user typed output/sys/...)
        if folder_paths:
            try:
                out_dir = Path(folder_paths.get_output_directory())
                candidates.append(out_dir / cleaned)
                if not cleaned.endswith(".latent"):
                    candidates.append(out_dir / (cleaned + ".latent"))
                
                # If path starts with 'output/' or 'output\', also check stripped subpath
                norm_c = cleaned.replace("\\", "/")
                if norm_c.startswith("output/"):
                    sub_c = norm_c[7:]
                    candidates.append(out_dir / sub_c)
                    if not sub_c.endswith(".latent"):
                        candidates.append(out_dir / (sub_c + ".latent"))
            except Exception:
                pass

        # 4. Check in ComfyUI input directory
        if folder_paths:
            try:
                inp_dir = Path(folder_paths.get_input_directory())
                candidates.append(inp_dir / cleaned)
                if not cleaned.endswith(".latent"):
                    candidates.append(inp_dir / (cleaned + ".latent"))

                norm_c = cleaned.replace("\\", "/")
                if norm_c.startswith("input/"):
                    sub_c = norm_c[6:]
                    candidates.append(inp_dir / sub_c)
                    if not sub_c.endswith(".latent"):
                        candidates.append(inp_dir / (sub_c + ".latent"))
            except Exception:
                pass

        # 5. Check in ComfyUI temp directory
        if folder_paths:
            try:
                tmp_dir = Path(folder_paths.get_temp_directory())
                candidates.append(tmp_dir / cleaned)
                if not cleaned.endswith(".latent"):
                    candidates.append(tmp_dir / (cleaned + ".latent"))
            except Exception:
                pass

        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()

        return p_direct

    def load_latent(self, latent_path: str, audio_latent_path: str = "", load_device: str = "cpu") -> Tuple[Dict[str, Any], str]:
        if not latent_path or not latent_path.strip():
            raise ValueError("[H3LoadLatentFromPath] latent_path is empty. Please provide a valid file path or filename.")

        resolved = self._resolve_path(latent_path)
        if not resolved or not resolved.exists():
            searched_hints = [
                f"- Absolute / Direct: {latent_path}",
                f"- Output folder: {Path(folder_paths.get_output_directory()) / latent_path if folder_paths else 'N/A'}",
                f"- Input folder: {Path(folder_paths.get_input_directory()) / latent_path if folder_paths else 'N/A'}",
            ]
            raise FileNotFoundError(
                f"[H3LoadLatentFromPath] Latent file not found: '{latent_path}'. Checked locations:\n" + "\n".join(searched_hints)
            )

        device = load_device if torch.cuda.is_available() and load_device == "cuda" else "cpu"
        
        # Load safetensors file
        try:
            loaded = safetensors.torch.load_file(str(resolved), device=device)
        except Exception as e:
            # Fallback to torch.load if not safetensors
            try:
                loaded = torch.load(str(resolved), map_location=device)
            except Exception:
                raise RuntimeError(f"[H3LoadLatentFromPath] Failed to load latent file '{resolved}': {e}")

        # Extract Video Latent
        video_tensor = None
        if "latent_tensor" in loaded:
            video_tensor = loaded["latent_tensor"]
        elif "samples" in loaded:
            video_tensor = loaded["samples"]
        else:
            for k, v in loaded.items():
                if isinstance(v, torch.Tensor) and v.ndim in (4, 5):
                    video_tensor = v
                    break

        if video_tensor is None:
            raise ValueError(f"[H3LoadLatentFromPath] No valid latent tensor found in '{resolved}'. Keys found: {list(loaded.keys())}")

        video_tensor = video_tensor.float()

        # Handle SD 4D latent scaling if version tag is absent
        if video_tensor.ndim == 4 and "latent_format_version_0" not in loaded:
            video_tensor = video_tensor * (1.0 / 0.18215)

        # Extract Audio Latent (for MiniMax H3 joint latents)
        audio_tensor = None
        audio_source_desc = "None"
        if "audio_latent_tensor" in loaded and isinstance(loaded["audio_latent_tensor"], torch.Tensor):
            audio_tensor = loaded["audio_latent_tensor"].float()
            audio_source_desc = "Embedded in file"
        elif "audio_samples" in loaded and isinstance(loaded["audio_samples"], torch.Tensor):
            audio_tensor = loaded["audio_samples"].float()
            audio_source_desc = "Embedded in file"

        # If audio is not inside the primary file, check explicit audio_latent_path or auto-pairing
        if audio_tensor is None:
            audio_path_resolved = None
            if audio_latent_path and audio_latent_path.strip():
                audio_path_resolved = self._resolve_path(audio_latent_path)
                if audio_path_resolved and audio_path_resolved.exists():
                    audio_source_desc = f"From explicit path: {audio_path_resolved.name}"
            else:
                # Auto-pairing detection in same directory
                parent = resolved.parent
                stem = resolved.stem
                suffix = resolved.suffix
                candidates = [
                    parent / f"{stem}_audio{suffix}",
                    parent / f"{stem}.audio{suffix}",
                ]
                if "video" in stem.lower():
                    # Replace 'video' with 'audio' while preserving case
                    stem_sub = stem.replace("video", "audio").replace("Video", "Audio").replace("VIDEO", "AUDIO")
                    candidates.append(parent / f"{stem_sub}{suffix}")
                
                for cand in candidates:
                    if cand.exists() and cand.is_file():
                        audio_path_resolved = cand
                        audio_source_desc = f"Auto-paired: {cand.name}"
                        break

            if audio_path_resolved and audio_path_resolved.exists():
                try:
                    loaded_a = safetensors.torch.load_file(str(audio_path_resolved), device=device)
                except Exception:
                    try:
                        loaded_a = torch.load(str(audio_path_resolved), map_location=device)
                    except Exception:
                        loaded_a = {}
                
                if "audio_latent_tensor" in loaded_a and isinstance(loaded_a["audio_latent_tensor"], torch.Tensor):
                    audio_tensor = loaded_a["audio_latent_tensor"].float()
                elif "audio_samples" in loaded_a and isinstance(loaded_a["audio_samples"], torch.Tensor):
                    audio_tensor = loaded_a["audio_samples"].float()
                elif "latent_tensor" in loaded_a and isinstance(loaded_a["latent_tensor"], torch.Tensor):
                    audio_tensor = loaded_a["latent_tensor"].float()
                elif "samples" in loaded_a and isinstance(loaded_a["samples"], torch.Tensor):
                    audio_tensor = loaded_a["samples"].float()

        # Bundle samples via standard package_latent
        latent_dict = package_latent(video_tensor, audio_tensor)

        # Build informative metadata string
        info_lines = [f"File: {resolved.name}"]
        if video_tensor.ndim == 5:
            b, c, t, h, w = video_tensor.shape
            derived_w = w * 16
            derived_h = h * 16
            derived_frames = (t - 1) * 4 + 1 if t > 1 else 1
            info_lines.append(f"Video Latent: 5D {list(video_tensor.shape)} (equiv. {derived_w}x{derived_h}, {derived_frames} frames)")
        elif video_tensor.ndim == 4:
            b, c, h, w = video_tensor.shape
            info_lines.append(f"Latent: 4D {list(video_tensor.shape)} (equiv. {w*8}x{h*8})")
        else:
            info_lines.append(f"Tensor Shape: {list(video_tensor.shape)}")

        if audio_tensor is not None:
            info_lines.append(f"Audio Latent: Joint Stereo {list(audio_tensor.shape)} ({audio_source_desc})")
        else:
            info_lines.append("Audio Latent: None")

        latent_info = " | ".join(info_lines)
        print(f"[H3LoadLatentFromPath] {latent_info}")

        return (latent_dict, latent_info)


class H3SaveLatent:
    """
    Losslessly saves ComfyUI LATENT dictionaries containing single tensors or
    joint MiniMax H3 NestedTensor([video, audio]) into a single .latent file.
    Solves the crash in ComfyUI default SaveLatent when handling NestedTensor combos.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory() if folder_paths else "ComfyUI/output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {
                    "tooltip": "Latent dictionary containing video samples or joint video + audio latents to serialize to disk."
                }),
                "filename_prefix": ("STRING", {
                    "default": "latents/H3_Latent",
                    "tooltip": "Subfolder and filename prefix for saving. Saves to ComfyUI/output/<filename_prefix>_00001_.latent"
                }),
            },
            "optional": {
                "save_separately": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When True and samples contains joint Video + Audio latents, saves two distinct files (<prefix>_video_... and <prefix>_audio_...) so each can be loaded by ComfyUI's default LoadLatent node."
                }),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "H3-TiledUpscale"

    def save(self, samples: Dict[str, Any], filename_prefix: str = "latents/H3_Latent", save_separately: bool = False, prompt=None, extra_pnginfo=None):
        import json

        video_tensor, audio_tensor = unpack_latent(samples)

        metadata = {}
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for k, v in extra_pnginfo.items():
                metadata[k] = json.dumps(v)

        has_audio = audio_tensor is not None

        if save_separately and has_audio:
            # Save separate video and audio files
            results = []

            # 1. Video
            prefix_v = f"{filename_prefix}_video"
            if folder_paths:
                full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(prefix_v, self.output_dir)
                file_v = f"{filename}_{counter:05}_.latent"
                path_v = os.path.join(full_folder, file_v)
                results.append({"filename": file_v, "subfolder": subfolder, "type": "output"})
            else:
                out_p = Path(self.output_dir) / f"{prefix_v}.latent"
                out_p.parent.mkdir(parents=True, exist_ok=True)
                path_v = str(out_p)
                results.append({"filename": out_p.name, "subfolder": "", "type": "output"})

            out_v = {
                "latent_tensor": video_tensor.contiguous().cpu(),
                "latent_format_version_0": torch.tensor([])
            }
            safetensors.torch.save_file(out_v, path_v, metadata=metadata)

            # 2. Audio (save under latent_tensor for default LoadLatent and audio_latent_tensor)
            prefix_a = f"{filename_prefix}_audio"
            if folder_paths:
                full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(prefix_a, self.output_dir)
                file_a = f"{filename}_{counter:05}_.latent"
                path_a = os.path.join(full_folder, file_a)
                results.append({"filename": file_a, "subfolder": subfolder, "type": "output"})
            else:
                out_p = Path(self.output_dir) / f"{prefix_a}.latent"
                out_p.parent.mkdir(parents=True, exist_ok=True)
                path_a = str(out_p)
                results.append({"filename": out_p.name, "subfolder": "", "type": "output"})

            out_a = {
                "latent_tensor": audio_tensor.contiguous().cpu(),
                "latent_format_version_0": torch.tensor([])
            }
            safetensors.torch.save_file(out_a, path_a, metadata=metadata)
            print(f"[H3SaveLatent] Latents saved separately: Video -> {path_v}, Audio -> {path_a}")

            out_latent = package_latent(video_tensor, audio_tensor, extra_dict=samples if isinstance(samples, dict) else None)
            return {"ui": {"latents": results}, "result": (out_latent,)}

        # Unified single file save
        if folder_paths:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
            file = f"{filename}_{counter:05}_.latent"
            full_path = os.path.join(full_output_folder, file)
            results = [{
                "filename": file,
                "subfolder": subfolder,
                "type": "output"
            }]
        else:
            out_p = Path(self.output_dir) / f"{filename_prefix}.latent"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            full_path = str(out_p)
            results = [{"filename": out_p.name, "subfolder": "", "type": "output"}]

        output = {
            "latent_tensor": video_tensor.contiguous().cpu(),
            "latent_format_version_0": torch.tensor([])
        }
        if audio_tensor is not None:
            output["audio_latent_tensor"] = audio_tensor.contiguous().cpu()
            metadata["has_audio_latent"] = "true"

        safetensors.torch.save_file(output, full_path, metadata=metadata)
        print(f"[H3SaveLatent] Latent saved successfully: {full_path} (Keys: {list(output.keys())})")

        out_latent = package_latent(video_tensor, audio_tensor, extra_dict=samples if isinstance(samples, dict) else None)
        return {"ui": {"latents": results}, "result": (out_latent,)}
