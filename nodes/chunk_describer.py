import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
import torch

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import comfy.model_management
except ImportError:
    pass

from ..core.grid_utils import compute_temporal_chunks
from ..core.vlm_engine import (
    extract_chunk_keyframes,
    caption_chunk_with_vlm,
    safe_to_cpu,
    safe_to_device
)
from ..core.prompt_utils import (
    parse_h3_prompt,
    format_cut_timestamp,
    strip_shot_headers,
    parse_shot_ranges,
    match_shots_to_chunk
)
from ..core.prompt_serializer import format_human_readable_prompts
from ..core.schemas import VLMContainer
from ..core.tensor_utils import unpack_latent


class H3ChunkVideoDescriber:
    """
    Automates video chunk captioning and canonical MiniMax H3 6-section prompt construction.
    Samples keyframes across each temporal chunk, extracts micro-texture descriptions via VLM/LLM,
    preserves user subject definitions, camera dynamics, and audio sections, and formats canonical
    per-chunk prompts and combined master prompt.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_style_prompt": ("STRING", {
                    "multiline": True,
                    "default": "cinematic photorealistic, 8k resolution, crisp textures",
                    "tooltip": "User's original prompt. Preserves character descriptions and narrative while injecting VLM micro-textures."
                }),
                "temporal_chunk_frames": ("INT", {
                    "default": 124, "min": -1, "max": 1000, "step": 1,
                    "tooltip": "Number of video frames per temporal chunk (e.g. 124 frames). Set to -1 to describe the entire video in a single chunk."
                }),
                "temporal_blend_frames": ("INT", {
                    "default": 17, "min": 0, "max": 68, "step": 1,
                    "tooltip": "Number of overlapping transition frames between chunks to ensure smooth prompt continuity and prevent seam cuts."
                }),
                "frames_per_chunk": ("INT", {
                    "default": 4, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Number of keyframes sampled from each chunk for the VLM to inspect when building chunk-specific micro-descriptions."
                }),
                "detail_focus": ([
                    "Balanced_Micro_Detail", "Fabrics_And_Costumes", 
                    "Faces_And_Skin_Textures", "Environments_And_Reflections", "Sharpening_And_Edges"
                ], {
                    "default": "Balanced_Micro_Detail",
                    "tooltip": "Aesthetic focus guiding the VLM prompt generation (e.g. emphasize skin pore texture, fabric weave, reflections, or overall detail)."
                }),
                "include_action_and_motion": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Instructs the VLM to describe camera dynamics (pan, zoom, orbit) and character kinetics to maintain motion realism."
                }),
                "save_prompts_to_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When True, saves prompt manifests to disk and dynamically reveals prompt_output_path."
                }),
                "prompt_output_path": ("STRING", {
                    "default": "output/h3_chunk_prompts",
                    "tooltip": "Destination directory for prompt manifests. Dynamically hidden when save_prompts_to_output is False."
                }),
                "unload_model_after_captioning": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Purges VLM weights from GPU VRAM immediately after captioning (0 GB VRAM leak)."
                }),
            },
            "optional": {
                "model": ("MODEL,QWENMODEL", {"tooltip": "VLM model weights (loaded via QwenLoader, Florence-2, Qwen-2.5-VL, or H3VLMModelLoader)"}),
                "latent": ("LATENT", {"tooltip": "Upscaled 5D canvas latent from H3LatentCanvasUpscale"}),
                "reference_bundle": ("REFERENCE_BUNDLE,H3_REFERENCE_BUNDLE", {"tooltip": "Multimodal references from H3ReferenceAssetBundle"}),
                "vae": ("VAE", {"tooltip": "Optional Video VAE to decode lightweight 1-frame keyframe thumbnails directly from 5D latents"}),
                "frames": ("IMAGE", {"tooltip": "Optional source video frames for direct visual inspection without latent decoding"}),
                "custom_prompt_guidance": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional guidance injected into the VLM prompt (e.g. 'Emphasize rain mist on dark leather')."
                }),
            }
        }

    RETURN_TYPES = ("CHUNK_PROMPTS", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("chunk_prompts", "master_prompt", "saved_prompt_path", "temporal_chunk_frames", "temporal_blend_frames")
    FUNCTION = "describe_chunks"
    CATEGORY = "H3-TiledUpscale"

    def describe_chunks(
        self,
        base_style_prompt="cinematic photorealistic, 8k resolution, crisp textures",
        temporal_chunk_frames=124,
        temporal_blend_frames=17,
        frames_per_chunk=4,
        detail_focus="Balanced_Micro_Detail",
        include_action_and_motion=True,
        save_prompts_to_output=True,
        prompt_output_path="output/h3_chunk_prompts",
        unload_model_after_captioning=True,
        model=None,
        latent=None,
        reference_bundle=None,
        vae=None,
        frames=None,
        custom_prompt_guidance="",
        **kwargs
    ):
        # 1. Prepare active VLM model
        active_model = model
        if active_model is not None and not isinstance(active_model, VLMContainer):
            inner_m = getattr(active_model, "model", active_model)
            processor = getattr(active_model, "processor", getattr(inner_m, "processor", None))
            if processor is not None:
                active_model = VLMContainer(
                    model=inner_m,
                    processor=processor,
                    model_name=getattr(active_model, "model_name", getattr(inner_m, "__class__", type("")).__name__),
                    model_type="florence2" if "florence" in str(type(inner_m)).lower() else "qwen2_5_vl",
                    precision="bf16",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )

        # 2. Unpack video latent & temporal dimensions
        video_lat = None
        if latent is not None:
            try:
                video_lat, _ = unpack_latent(latent)
            except Exception:
                video_lat = None

        if video_lat is not None and hasattr(video_lat, "shape") and video_lat.ndim == 5:
            T_lat = video_lat.shape[2]
            total_frames = (T_lat - 1) * 4 + 1 if T_lat > 1 else 1
        elif frames is not None and hasattr(frames, "shape"):
            total_frames = frames.shape[0]
            T_lat = (total_frames - 1) // 4 + 1
        else:
            total_frames = max(1, temporal_chunk_frames if temporal_chunk_frames > 0 else 124)
            T_lat = (total_frames - 1) // 4 + 1

        fps = 24.0
        temporal_chunks = compute_temporal_chunks(
            total_frames,
            chunk_frames=temporal_chunk_frames,
            blend_frames=temporal_blend_frames
        )
        num_chunks = len(temporal_chunks)
        print(f"[H3ChunkVideoDescriber] Beginning analysis across {num_chunks} temporal chunks ({frames_per_chunk} keyframes/chunk)...")

        # 3. Parse original user prompt into standard MiniMax H3 sections
        parsed_user = parse_h3_prompt(base_style_prompt)
        orig_sections = parsed_user.get("sections", {}) if parsed_user.get("is_structured") else {}

        chunk_results: Dict[int, Dict[str, Any]] = {}
        shot_lines: List[str] = []
        chunk_shot_bodies: Dict[int, str] = {}

        # 4. Iterate temporal chunks and analyze keyframes
        for chunk in temporal_chunks:
            k = chunk.chunk_idx
            s_f = chunk.start_frame
            e_f = chunk.end_frame
            s_sec = s_f / fps
            e_sec = e_f / fps
            total_chunk_frames = max(1, e_f - s_f)
            sample_count = max(1, min(frames_per_chunk, total_chunk_frames))

            keyframe_images: List[Image.Image] = []

            # Option A: In-memory raw video frames tensor [N, H, W, 3]
            if frames is not None and isinstance(frames, torch.Tensor):
                keyframe_images = extract_chunk_keyframes(
                    video_path="",
                    start_frame=s_f,
                    end_frame=e_f,
                    num_samples=sample_count,
                    video_tensor=frames
                )

            # Option B: Decode keyframes from 5D latent using VAE
            elif vae is not None and video_lat is not None:
                try:
                    indices = [
                        int(round(s_f + i * (total_chunk_frames - 1) / max(1, sample_count - 1)))
                        for i in range(sample_count)
                    ]
                    dev = vae.device if hasattr(vae, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
                    for f_idx in indices:
                        lat_idx = min(f_idx // 4, video_lat.shape[2] - 1)
                        z_slice = video_lat[:, :, lat_idx:lat_idx+1, :, :]
                        with torch.no_grad():
                            dec_th = vae.decode(z_slice.to(dev))
                        if isinstance(dec_th, torch.Tensor):
                            dec_f = dec_th.detach().cpu()
                            while dec_f.ndim > 3:
                                dec_f = dec_f[0]
                            if dec_f.ndim == 3 and dec_f.shape[0] == 3:
                                dec_f = dec_f.permute(1, 2, 0)
                            dec_np = (dec_f.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
                            keyframe_images.append(Image.fromarray(dec_np))
                except Exception as e_dec:
                    print(f"[H3ChunkVideoDescriber] VAE decode notice for chunk {k}: {e_dec}")

            # Option C: Include reference pictures from reference_bundle if available
            if reference_bundle:
                pics = reference_bundle.get("pictures", []) if isinstance(reference_bundle, dict) else getattr(reference_bundle, "pictures", [])
                for ref_pic in pics[:2]:
                    if isinstance(ref_pic, torch.Tensor):
                        ref_t = ref_pic.detach().cpu()
                        while ref_t.ndim > 3:
                            ref_t = ref_t[0]
                        if ref_t.ndim == 3 and ref_t.shape[0] == 3:
                            ref_t = ref_t.permute(1, 2, 0)
                        ref_np = (ref_t.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
                        keyframe_images.append(Image.fromarray(ref_np))
                    elif isinstance(ref_pic, Image.Image):
                        keyframe_images.append(ref_pic)

            # Fallback placeholder if no keyframes resolved
            if not keyframe_images:
                keyframe_images = [Image.new("RGB", (512, 512), (128, 128, 128))]

            # 5. Formulate canonical MiniMax H3 shot header
            if k == 0:
                shot_header = "[Shot 1]"
            else:
                cut_str = format_cut_timestamp(s_sec)
                shot_header = f"[Shot {k + 1}] At {cut_str}, the shot transitions to"

            # 6. Extract user narrative context strictly within this chunk's timeframe
            chunk_scene_context = ""
            if orig_sections.get("detailed_description"):
                orig_desc_text = orig_sections["detailed_description"]
                parsed_shots = parse_shot_ranges(orig_desc_text)
                if len(parsed_shots) > 1:
                    matched = match_shots_to_chunk(parsed_shots, s_sec, e_sec)
                    if matched:
                        chunk_scene_context = " ".join(strip_shot_headers(p[2]) for p in matched if strip_shot_headers(p[2])).strip()
                    else:
                        chunk_scene_context = ""
                else:
                    if s_sec < 5.0 or not active_model:
                        chunk_scene_context = strip_shot_headers(orig_desc_text).strip()
                    else:
                        chunk_scene_context = ""
            elif base_style_prompt.strip():
                if s_sec < 5.0 or not active_model:
                    chunk_scene_context = strip_shot_headers(base_style_prompt).strip()
                else:
                    chunk_scene_context = ""

            # 7. Generate VLM / LLM micro-detail description
            guidance = custom_prompt_guidance.strip()
            if chunk_scene_context:
                guidance = f"{chunk_scene_context}. {guidance}".strip() if guidance else chunk_scene_context

            desc = ""
            if active_model is not None:
                try:
                    desc = caption_chunk_with_vlm(
                        vlm=active_model,
                        images=keyframe_images,
                        detail_focus=detail_focus,
                        custom_guidance=guidance,
                        include_action=include_action_and_motion
                    )
                except Exception as e_vlm:
                    print(f"[H3ChunkVideoDescriber] VLM captioning notice for chunk {k}: {e_vlm}")
                    desc = ""

            clean_desc = desc.strip().rstrip(".")
            clean_vlm = f"Observable micro-textures feature {clean_desc}." if clean_desc else ""

            if chunk_scene_context:
                clean_ctx = chunk_scene_context.rstrip(".")
                if clean_vlm:
                    shot_body = f"{clean_ctx}. {clean_vlm}"
                else:
                    shot_body = f"{clean_ctx}."
            else:
                shot_body = clean_vlm if clean_vlm else "Preserving continuous motion and elevated micro-detail."

            shot_lines.append(f"{shot_header} {shot_body}".strip())
            chunk_shot_bodies[k] = shot_body
            chunk_results[k] = {
                "start_frame": s_f,
                "end_frame": e_f,
                "description": desc or clean_ctx
            }

        # 8. Assemble canonical 6-section MiniMax H3 Master Prompt
        shots_block = "\n".join(shot_lines)

        task_prefix = "[video editing + reference generation + audio reuse]"
        if parsed_user.get("is_structured"):
            # 100% preserve user's original subjects, summary, retention, and soundscapes
            subj_defs = orig_sections.get("subject_definitions") or f"<Subject 1> is the primary subject in <Video 1>.\n<Picture 1> is the scene anchor frame."
            orig_sum = orig_sections.get("summary", "").strip()
            if orig_sum:
                clean_sum = re.sub(r'^\[.*?\]\s*', '', orig_sum).strip()
                summary_txt = f"{task_prefix} {clean_sum}"
            else:
                summary_txt = f"{task_prefix} Super-resolution detail refinement elevating physical micro-textures while adhering to motion."
            retention_txt = orig_sections.get("retention_analysis") or "- Fully preserve camera velocity, trajectories, object silhouettes, and lighting from <Video 1>.\n- Elevate micro-detail: fabric weaves, skin pores, hair strands, specular reflections."
            sound_txt = orig_sections.get("overall_soundscape") or "Synchronized ambient sound matching the motion in <Video 1>."
            music_txt = orig_sections.get("non_diegetic_music") or "None."
        else:
            subj_defs = f"<Subject 1> is the primary subject, micro-textures, and fine material features visible across the footage ({base_style_prompt.strip()}).\n<Picture 1> is the full-frame scene establishing global spatial position, environment, and context.\n<Video 1> is the source motion and temporal progression."
            summary_txt = f"{task_prefix} Super-resolution detail refinement elevating physical micro-textures, surface lighting, and fine edges while strictly preserving original motion and geometry: {base_style_prompt.strip()}."
            retention_txt = "- Fully preserve camera velocity, trajectories, object silhouettes, and lighting from <Video 1>.\n- Forbid any structural alteration, camera shifts, or trajectory drifts.\n- Elevate micro-detail: fabric weaves, skin pores, hair strands, specular reflections, and surface roughness."
            sound_txt = "Synchronized ambient sound matching the motion in <Video 1>."
            music_txt = "None."

        master_h3_prompt = f"""subject_definitions:
{subj_defs.strip()}

summary:
{summary_txt.strip()}

retention_analysis:
{retention_txt.strip()}

detailed_description:
{shots_block}

overall_soundscape:
{sound_txt.strip()}

non_diegetic_music:
{music_txt.strip()}"""

        # 9. Build tailored per-chunk 6-section prompts dictionary
        chunk_prompts = {}
        for chunk in temporal_chunks:
            k = chunk.chunk_idx
            c_body = chunk_shot_bodies.get(k, "Preserving continuous motion and elevated micro-detail.")
            chunk_prompt_text = f"""subject_definitions:
{subj_defs.strip()}

summary:
{summary_txt.strip()}

retention_analysis:
{retention_txt.strip()}

detailed_description:
[Shot 1] {c_body}

overall_soundscape:
{sound_txt.strip()}

non_diegetic_music:
{music_txt.strip()}"""
            chunk_prompts[k] = chunk_prompt_text
            chunk_prompts[str(k)] = chunk_prompt_text

        # 10. Save prompt manifest to disk if requested
        saved_path = ""
        if save_prompts_to_output:
            try:
                out_root = Path(folder_paths.get_output_directory() if folder_paths else "output")
                out_dir = Path(prompt_output_path) if Path(prompt_output_path).is_absolute() else out_root / prompt_output_path
                out_dir.mkdir(parents=True, exist_ok=True)

                session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                txt_file = out_dir / f"h3_prompts_{session_id}.txt"
                json_file = out_dir / f"h3_prompts_{session_id}.json"

                txt_content = format_human_readable_prompts(
                    session_id=session_id,
                    master_prompt=master_h3_prompt,
                    chunk_prompts=chunk_results,
                    fps=fps
                )
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(txt_content)

                json_data = {
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "fps": fps,
                    "total_chunks": len(chunk_results),
                    "master_prompt": master_h3_prompt,
                    "chunks": [
                        {
                            "chunk_idx": k,
                            "start_frame": chunk_results[k].get("start_frame", 0),
                            "end_frame": chunk_results[k].get("end_frame", 0),
                            "start_sec": chunk_results[k].get("start_frame", 0) / fps,
                            "end_sec": chunk_results[k].get("end_frame", 0) / fps,
                            "description": chunk_results[k].get("description", "")
                        }
                        for k in sorted(chunk_results.keys())
                    ]
                }
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                saved_path = str(txt_file)
                print(f"[H3ChunkVideoDescriber] Saved prompts manifest to {txt_file}")
            except Exception as e_save:
                print(f"[H3ChunkVideoDescriber] Notice: Could not write prompt manifest: {e_save}")

        # 11. Evict VLM from GPU memory if requested
        if unload_model_after_captioning and model is not None:
            safe_to_cpu(model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                comfy.model_management.soft_empty_cache()
            except Exception:
                pass

        if temporal_chunk_frames is not None and temporal_chunk_frames <= 0:
            out_chunk_frames = int(total_frames)
            out_blend_frames = 0
        else:
            out_chunk_frames = int(temporal_chunk_frames if temporal_chunk_frames is not None else 124)
            out_blend_frames = int(temporal_blend_frames if temporal_blend_frames is not None else 17)

        return (chunk_prompts, master_h3_prompt, saved_path, out_chunk_frames, out_blend_frames)
