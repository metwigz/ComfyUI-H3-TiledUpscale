import os
import re
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import safetensors.torch

try:
    import folder_paths
except ImportError:
    folder_paths = None

from ..core.ffmpeg_pipe import get_ffmpeg_binary, spawn_frame_writer, mux_audio_track
from ..core.tensor_utils import tensor_frame_to_rgb24_bytes, save_waveform_to_wav, unpack_latent


def resolve_dynamic_path(
    base_dir: str,
    prefix_pattern: str,
    ext: str,
    meta: Dict[str, Any],
    overwrite: bool = False
) -> Tuple[Path, str]:
    """
    Parses dynamic formatting tokens (%date%, %time%, %counter%, %resolution%, etc.)
    and resolves absolute destination directory and filename.
    """
    now = datetime.datetime.now()
    resolved_dir_str = base_dir.strip().replace("\\", "/") if base_dir else ""
    
    # 1. Resolve base directory (absolute vs ComfyUI output)
    if not resolved_dir_str:
        out_root = Path(folder_paths.get_output_directory() if folder_paths else "output")
    elif Path(resolved_dir_str).is_absolute():
        out_root = Path(resolved_dir_str)
    else:
        comfy_out = Path(folder_paths.get_output_directory() if folder_paths else "output")
        norm = resolved_dir_str.replace("\\", "/")
        if norm.startswith("output/"):
            resolved_dir_str = norm[7:]
        out_root = comfy_out / resolved_dir_str

    # 2. Token dictionary
    tokens = {
        "%date%": now.strftime("%Y-%m-%d"),
        "%time%": now.strftime("%H%M%S"),
        "%resolution%": f"{meta.get('width', 3840)}x{meta.get('height', 2160)}",
        "%width%": str(meta.get("width", 3840)),
        "%height%": str(meta.get("height", 2160)),
        "%megapixels%": f"{meta.get('megapixels', 8.29):.2f}MP",
        "%fps%": f"{int(round(meta.get('fps', 24)))}fps",
        "%frames%": f"{meta.get('frames', 124)}f",
        "%seed%": f"seed_{meta.get('seed', 0)}",
        "%session%": str(meta.get("session_id", "h3"))[:8]
    }

    # Substitute tokens in directory and prefix
    for k, v in tokens.items():
        if k in str(out_root):
            out_root = Path(str(out_root).replace(k, v))
        prefix_pattern = prefix_pattern.replace(k, v)

    out_root.mkdir(parents=True, exist_ok=True)

    # 3. Handle sequential counter
    counter_match = re.search(r"%counter(?::(\d+))?%", prefix_pattern)
    pad_len = int(counter_match.group(1)) if (counter_match and counter_match.group(1)) else 5

    if overwrite:
        clean_prefix = re.sub(r"%counter(?::\d+)?%", "00001", prefix_pattern)
        filename = f"{clean_prefix}.{ext.lstrip('.')}"
        return out_root / filename, filename

    # Search existing files to find highest counter
    search_prefix = re.sub(r"%counter(?::\d+)?%", "*", prefix_pattern)
    existing = list(out_root.glob(f"{search_prefix}.{ext.lstrip('.')}"))
    max_counter = 0
    for f in existing:
        m = re.search(r"(\d+)", f.stem)
        if m:
            val = int(m.group(1))
            if val > max_counter:
                max_counter = val

    next_counter = max_counter + 1
    counter_str = str(next_counter).zfill(pad_len)

    if counter_match:
        final_stem = prefix_pattern.replace(counter_match.group(0), counter_str)
    else:
        final_stem = f"{prefix_pattern}_{counter_str}"

    filename = f"{final_stem}.{ext.lstrip('.')}"
    return out_root / filename, filename


class H3VideoSave:
    """
    Dedicated, user-friendly video exporter supporting custom output directories,
    arbitrary filenames, dynamic placeholder tokens, and multiple video codecs.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "4K canvas latent containing video and audio latents (decoded via tiled streaming)"}),
                "vae": ("VAE", {"tooltip": "MiniMax Video VAE for tiled streaming decode"}),
                "output_dir": ("STRING", {
                    "default": "output/4K_Upscales",
                    "tooltip": "Destination folder. Supports absolute paths (e.g. D:/Renders/) or relative paths inside ComfyUI/output/."
                }),
                "filename_prefix": ("STRING", {
                    "default": "H3_%date%_%resolution%_%counter%",
                    "tooltip": "Filename pattern. Supports %date%, %time%, %counter%, %resolution%, %fps%, %seed%."
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "Video framerate in frames per second (e.g. 24.0 for standard cinematic film, 30.0, 60.0)."
                }),
                "video_codec": ([
                    "h264_nvenc (NVIDIA GPU - Recommended)",
                    "hevc_nvenc (NVIDIA GPU - 10-bit)",
                    "Apple ProRes 422 HQ (MOV - VFX Master)",
                    "H.264 (CPU libx264 - Universal)",
                    "AV1 (MKV - Modern Open Source)"
                ], {
                    "default": "h264_nvenc (NVIDIA GPU - Recommended)",
                    "tooltip": "Output video compression codec: NVIDIA NVENC for fast hardware export, ProRes 422 HQ for post-production editing, or libx264/AV1 for universal playback."
                }),
                "bitrate_mbps": ("INT", {
                    "default": 40, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Video bitrate in Mbps for GPU/CPU encoders."
                }),
                "save_latent_copy": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When True, saves companion .latent file(s) and dynamically reveals latent_output_dir and save_latents_separately."
                }),
                "latent_output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Optional destination folder for .latent files. If left empty, saves to output_dir. Dynamically hidden when save_latent_copy is False."
                }),
                "save_latents_separately": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When True (and save_latent_copy is active), saves separate *_video.latent and *_audio.latent files for standard ComfyUI LoadLatent compatibility. Dynamically hidden when save_latent_copy is False."
                }),
                "overwrite_existing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When True, overwrites destination file instead of creating sequential _00001_ copies."
                }),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "MiniMax Audio VAE to decode embedded audio latent directly to PCM"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("saved_video_path", "preview_frame")
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "H3-TiledUpscale/export"

    def save_video(
        self,
        latent,
        vae,
        output_dir="output/4K_Upscales",
        filename_prefix="H3_%date%_%resolution%_%counter%",
        fps=24.0,
        video_codec="h264_nvenc",
        bitrate_mbps=40,
        save_latent_copy=False,
        latent_output_dir="",
        save_latents_separately=False,
        overwrite_existing=False,
        audio_vae=None,
        **kwargs
    ):
        video_lat, audio_lat = unpack_latent(latent)
        if video_lat is None:
            raise ValueError("[H3VideoSave] No video latent found in 'latent' input!")
        b, c, t, h_lat, w_lat = video_lat.shape
        w, h = w_lat * 16, h_lat * 16
        ext = "mov" if "ProRes" in video_codec else ("mkv" if "AV1" in video_codec else "mp4")
        
        meta = {"width": w, "height": h, "fps": fps, "frames": (t - 1) * 4 + 1 if t > 1 else 1, "megapixels": (w * h) / 1e6}
        final_path, filename = resolve_dynamic_path(output_dir, filename_prefix, ext, meta, overwrite=overwrite_existing)

        # 1. Decode & encode video frames via VAE streaming to FFmpeg
        temp_video_path = final_path.with_name(f"temp_stream_{final_path.name}")
        writer = spawn_frame_writer(str(temp_video_path), w, h, fps, codec=video_codec, bitrate_mbps=bitrate_mbps)
        preview_frame = None
        dev = next(vae.first_stage_model.parameters()).device if hasattr(vae, "first_stage_model") else (vae.device if hasattr(vae, "device") else "cuda" if torch.cuda.is_available() else "cpu")

        try:
            with torch.no_grad():
                dec = vae.decode(video_lat.to(dev))
            if dec.ndim == 5:
                dec = dec.reshape(-1, dec.shape[-3], dec.shape[-2], dec.shape[-1])
            for f_idx in range(dec.shape[0]):
                frame_t = dec[f_idx]
                writer.stdin.write(tensor_frame_to_rgb24_bytes(frame_t))
                if preview_frame is None or f_idx == dec.shape[0] // 2:
                    preview_frame = frame_t.unsqueeze(0).cpu()
        finally:
            writer.stdin.close()
            writer.wait()

        if preview_frame is None:
            preview_frame = torch.zeros((1, h, w, 3), dtype=torch.float32)

        # 2. Decode audio via audio_vae & multiplex into container
        if audio_vae is not None and audio_lat is not None:
            try:
                aud_dev = audio_vae.device if hasattr(audio_vae, "device") else dev
                with torch.no_grad():
                    aud_pcm = audio_vae.decode(audio_lat.to(aud_dev))
                if hasattr(aud_pcm, "detach"):
                    aud_pcm = aud_pcm.detach().cpu()
                temp_wav = final_path.with_suffix(".temp.wav")
                sr = getattr(audio_vae, "audio_sample_rate", getattr(audio_vae, "audio_sample_rate_output", 32000))
                if save_waveform_to_wav(aud_pcm, temp_wav, sample_rate=sr):
                    if mux_audio_track(str(temp_video_path), str(temp_wav), str(final_path)):
                        temp_video_path.unlink(missing_ok=True)
                        temp_wav.unlink(missing_ok=True)
                    else:
                        if temp_video_path.exists():
                            temp_video_path.replace(final_path)
                else:
                    if temp_video_path.exists():
                        temp_video_path.replace(final_path)
            except Exception as e_mux:
                print(f"[H3VideoSave] Audio multiplex warning: {e_mux}")
                if temp_video_path.exists():
                    temp_video_path.replace(final_path)
        else:
            if temp_video_path.exists():
                temp_video_path.replace(final_path)

        # 3. Companion Latent Export (Single or Separate)
        if save_latent_copy:
            target_latent_dir = latent_output_dir.strip() if latent_output_dir and latent_output_dir.strip() else output_dir
            latent_target_path, _ = resolve_dynamic_path(target_latent_dir, filename_prefix, "latent", meta, overwrite=overwrite_existing)

            if save_latents_separately and audio_lat is not None:
                # Save separate video and audio .latent files for standard ComfyUI LoadLatent compatibility
                safetensors.torch.save_file({"latent_tensor": video_lat.contiguous().cpu(), "latent_format_version_0": torch.tensor([])}, str(latent_target_path.with_name(f"{latent_target_path.stem}_video.latent")))
                safetensors.torch.save_file({"latent_tensor": audio_lat.contiguous().cpu(), "latent_format_version_0": torch.tensor([])}, str(latent_target_path.with_name(f"{latent_target_path.stem}_audio.latent")))
            else:
                # Save single combined .latent file compatible with both standard ComfyUI and custom nodes
                payload = {"latent_tensor": video_lat.contiguous().cpu(), "latent_format_version_0": torch.tensor([])}
                if audio_lat is not None:
                    payload["audio_latent_tensor"] = audio_lat.contiguous().cpu()
                safetensors.torch.save_file(payload, str(latent_target_path))

        print(f"[H3VideoSave] Successfully saved 4K video: {final_path} ({w}x{h} @ {fps} fps)")
        return {"ui": {"videos": [{"filename": filename, "subfolder": output_dir, "type": "output"}]}, "result": (str(final_path), preview_frame)}
