import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import torch
import safetensors.torch

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import comfy
    import comfy.sample
    import comfy.nested_tensor
    import comfy.model_management
    import comfy.model_base
    import comfy.utils
except ImportError:
    comfy = None

from ..core.grid_utils import (
    compute_tile_intervals,
    compute_temporal_chunks,
    calculate_optimal_tile_dimensions
)
from ..core.laplacian_pyramid import generate_hann_weight_2d
from ..core.prompt_utils import (
    build_tile_ref2va_prompt,
    inject_reference_bundle,
    inject_keyframe_anchor
)
from ..core.disk_streaming import dump_tile_to_disk_stream, load_tile
from ..core.schemas import slice_audio_latent_for_video_chunk
from ..core.tensor_utils import unpack_latent, package_latent


def export_debug_chunk_mp4(
    chunk_latent: torch.Tensor,
    chunk_idx: int,
    output_dir: str,
    vae: Any,
    audio_vae: Optional[Any] = None,
    full_latent: Optional[Dict[str, Any]] = None,
    fps: float = 24.0
) -> Path:
    """
    Decodes an assembled temporal latent chunk and writes preview MP4 to output_dir.
    Multiplexes synchronized PCM audio if audio_vae and audio_samples are present.
    """
    from ..core.ffmpeg_pipe import spawn_frame_writer, mux_audio_track
    from ..core.tensor_utils import tensor_frame_to_rgb24_bytes, save_waveform_to_wav

    out_root = Path(folder_paths.get_output_directory() if folder_paths else "output")
    resolved_dir = Path(output_dir) if Path(output_dir).is_absolute() else out_root / output_dir
    resolved_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = resolved_dir / f"chunk_{chunk_idx:02d}.mp4"

    # 1. Decode video frames
    dev = vae.device if hasattr(vae, "device") else "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        dec_frames = vae.decode(chunk_latent.to(dev))
    if dec_frames.ndim == 5:
        dec_frames = dec_frames.reshape(-1, dec_frames.shape[-3], dec_frames.shape[-2], dec_frames.shape[-1])

    num_f, h, w = dec_frames.shape[0], dec_frames.shape[1], dec_frames.shape[2]
    writer = spawn_frame_writer(str(out_mp4), w, h, fps)
    for f in range(num_f):
        writer.stdin.write(tensor_frame_to_rgb24_bytes(dec_frames[f]))
    writer.stdin.close()
    writer.wait()

    # 2. Multiplex synchronized audio if available
    if audio_vae is not None and full_latent:
        try:
            full_v_lat, full_a_lat = unpack_latent(full_latent)
            if full_a_lat is not None:
                total_v_tokens = full_v_lat.shape[2]
                c_start_token = (chunk_idx * 124 // 17) * 5
                c_end_token = c_start_token + chunk_latent.shape[2]
                aud_slice = slice_audio_latent_for_video_chunk(full_a_lat, c_start_token, c_end_token, total_v_tokens)
                if aud_slice is not None:
                    aud_dev = audio_vae.device if hasattr(audio_vae, "device") else dev
                    aud_pcm = audio_vae.decode(aud_slice.to(aud_dev))
                    temp_wav = resolved_dir / f"temp_chunk_{chunk_idx:02d}.wav"
                    sr = getattr(audio_vae, "audio_sample_rate", 32000)
                    if save_waveform_to_wav(aud_pcm, temp_wav, sample_rate=sr):
                        final_muxed = resolved_dir / f"chunk_{chunk_idx:02d}_muxed.mp4"
                        if mux_audio_track(str(out_mp4), str(temp_wav), str(final_muxed)):
                            out_mp4.unlink(missing_ok=True)
                            temp_wav.unlink(missing_ok=True)
                            final_muxed.rename(out_mp4)
        except Exception as e_aud:
            print(f"[H3LatentTiledKSampler] Audio mux notice for chunk {chunk_idx}: {e_aud}")

    print(f"[H3LatentTiledKSampler] Saved preview chunk: {out_mp4.name} ({w}x{h} @ {fps} fps)")
    return out_mp4


class H3LatentTiledKSampler:
    """
    Unified 5D Latent Tiled KSampler.
    Drop-in replacement for standard KSampler that slices spatial tiles & temporal chunks directly
    in 5D latent space, performs DiT sampling with Ref2VA conditioning, and accumulates the 4K canvas
    with O(1) host RAM footprint and zero intermediate full-frame VAE roundtrips.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MiniMax H3 DiT model backbone"}),
                "latent_image": ("LATENT", {
                    "tooltip": "Pre-scaled 5D canvas latent from H3LatentCanvasUpscale (standard ComfyUI naming)."
                }),
                "sampler": ("SAMPLER", {"tooltip": "Sampling algorithm (e.g. euler, dpmpp_2m, res_multistep) used to denoise each spatial tile."}),
                "sigmas": ("SIGMAS", {"tooltip": "Noise schedule sigmas defining the step trajectory for tile denoising."}),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05,
                    "tooltip": "Classifier-Free Guidance scale. 1.0 to 2.0 is recommended for H3 Ref2VA upscaling to avoid saturation artifacts."
                }),
                "tile_megapixels": ("FLOAT", {
                    "default": 0.92, "min": -1.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Target tile budget in megapixels (default 0.92 MP ~ 1280x720) fitting within 16GB-24GB VRAM. Set <= 0 to disable spatial tiling."
                }),
                "overlap_percent": ("FLOAT", {
                    "default": 0.25, "min": 0.10, "max": 0.50, "step": 0.01,
                    "tooltip": "Fractional spatial overlap between adjacent tiles (e.g. 0.25 = 25% overlap) to prevent boundary seams."
                }),
                "tight_tile_overlap": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Minimizes tile dimensions to satisfy overlap_percent without redundant overlap inflation while strictly preserving canvas aspect ratio."
                }),
                "spatial_blend_mode": (["Multiscale_Laplacian", "Wavelet_Frequency_Decouple", "Linear_Feather"], {
                    "default": "Multiscale_Laplacian",
                    "tooltip": "Tile blending algorithm: 'Multiscale_Laplacian' (seamless frequency decomposition, recommended), 'Wavelet_Frequency_Decouple', or 'Linear_Feather'."
                }),
                "temporal_chunk_frames": ("INT", {
                    "default": 124, "min": -1, "max": 1000,
                    "tooltip": "Number of video frames per temporal chunk (default 124 frames). Set to -1 to process entire video length if VRAM allows."
                }),
                "temporal_blend_frames": ("INT", {
                    "default": 17, "min": 0, "max": 68,
                    "tooltip": "Number of transition frames between adjacent temporal chunks (e.g. 17 frames = 5 latent tokens) for smooth continuity."
                }),
                "temporal_stabilization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enforces inter-chunk temporal anchoring and latents stabilization to eliminate flickering across chunk seams."
                }),
                "tile_storage_strategy": (["temp_disk_stream", "in_memory"], {
                    "default": "temp_disk_stream",
                    "tooltip": "Memory management for tiles: 'temp_disk_stream' streams tiles to disk keeping host RAM usage at O(1); 'in_memory' caches all tiles in RAM."
                }),
                "debug_decode_chunks": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When True, decodes each assembled temporal chunk to preview MP4 videos with synchronized audio."
                }),
                "debug_chunk_output_dir": ("STRING", {
                    "default": "output/h3_debug_chunks",
                    "tooltip": "Folder for preview chunk MP4s. Dynamically hidden when debug_decode_chunks is False."
                }),
            },
            "optional": {
                "chunk_prompts": ("CHUNK_PROMPTS", {
                    "tooltip": "Per-chunk prompt manifest from H3ChunkVideoDescriber, providing context-aware prompts for each temporal chunk."
                }),
                "reference_bundle": ("H3_REFERENCE_BUNDLE", {
                    "tooltip": "Multimodal reference bundle (pictures, videos, audios) from H3ReferenceAssetBundle for H3 Ref2VA conditioning."
                }),
                "clip": ("CLIP", {
                    "tooltip": "CLIP / Text Encoder used to encode chunk prompts into conditioning tensors if not already embedded."
                }),
                "vae": ("VAE", {"tooltip": "Video VAE for keyframe anchor propagation and debug chunk decoding"}),
                "audio_vae": ("VAE", {"tooltip": "Audio VAE to decode and multiplex audio into debug chunk videos"}),
                "propagate_keyframes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Carries boundary keyframe latents forward between chunks as conditioning anchors for strict temporal continuity."
                }),
                "color_lock_to_base": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Locks output color distribution and luminance to the base canvas latent. Default False to protect 3D VAE temporal cadence."
                }),
                "enable_intersection_center_patches": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Generates minimal-MP patches centered on 4-way tile intersections simultaneously from the starting canvas with natural Hann blending, providing an organic neural bridge across seams."
                }),
                "offload_device": (["cpu", "none"], {
                    "default": "cpu",
                    "tooltip": "Device to offload intermediate models and tensors to between sampling passes ('cpu' saves VRAM; 'none' keeps on GPU for maximum speed)."
                }),
                "positive": ("CONDITIONING", {
                    "tooltip": "Optional external conditioning (e.g. from CLIPTextEncode or external conditioning pipeline)."
                }),
                "precache_tile_conditioning": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Pre-encodes reference bundle latents and all spatial tile CLIP prompts upfront, unloading VAE and CLIP once before sampling begins. Eliminates redundant VRAM model swapping on every tile."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for spatially coherent canvas noise across all tiles and chunks. Overlapping tile regions share identical noise, eliminating motion seam artifacts."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample_tiles"
    CATEGORY = "H3-TiledUpscale/sampling"

    def sample_tiles(
        self,
        model,
        sampler,
        sigmas,
        latent_image,
        cfg=1.0,
        tile_megapixels=0.92,
        overlap_percent=0.25,
        tight_tile_overlap=False,
        spatial_blend_mode="Multiscale_Laplacian",
        temporal_chunk_frames=124,
        temporal_blend_frames=17,
        temporal_stabilization=True,
        tile_storage_strategy="temp_disk_stream",
        debug_decode_chunks=False,
        debug_chunk_output_dir="output/h3_debug_chunks",
        chunk_prompts=None,
        reference_bundle=None,
        clip=None,
        vae=None,
        audio_vae=None,
        propagate_keyframes=True,
        color_lock_to_base=False,
        enable_intersection_center_patches=False,
        offload_device="cpu",
        positive=None,
        precache_tile_conditioning=True,
        seed=0,
        **kwargs
    ):
        if latent_image is None:
            raise ValueError("[H3LatentTiledKSampler] No latent input provided! Connect 'latent_image'.")

        canvas_latent, input_audio_latent = unpack_latent(latent_image)
        if canvas_latent is None:
            raise ValueError("[H3LatentTiledKSampler] No video latent found in 'latent_image'!")

        B, C, T_lat, H_lat, W_lat = canvas_latent.shape
        target_w, target_h = W_lat * 16, H_lat * 16
        total_frames = (T_lat - 1) * 4 + 1 if T_lat > 1 else 1

        temporal_chunk_frames = int(temporal_chunk_frames) if temporal_chunk_frames is not None else 124
        temporal_blend_frames = int(temporal_blend_frames) if temporal_blend_frames is not None else 17

        # 1. Compute Spatial Grid Intervals & Temporal Chunks (respecting aspect ratio & tight overlap)
        tile_w, tile_h, _, _ = calculate_optimal_tile_dimensions(
            target_w, target_h, tile_megapixels, overlap_percent, tight_tile_overlap=tight_tile_overlap
        )
        x_intervals = compute_tile_intervals(target_w, tile_w, overlap_percent)
        y_intervals = compute_tile_intervals(target_h, tile_h, overlap_percent)
        temporal_chunks = compute_temporal_chunks(total_frames, temporal_chunk_frames, blend_frames=temporal_blend_frames)

        # Construct spatial tile list: primary tiles + optional intersection-centered patches
        spatial_tiles = []
        for r, (y1, y2) in enumerate(y_intervals):
            for c, (x1, x2) in enumerate(x_intervals):
                spatial_tiles.append({
                    "coords": (x1, x2, y1, y2),
                    "is_center_patch": False,
                    "grid_pos": (r, c),
                    "desc": f"(r{r+1},c{c+1})"
                })

        if enable_intersection_center_patches and len(x_intervals) > 1 and len(y_intervals) > 1:
            aspect = float(tile_w) / float(tile_h)
            max_seam_w = max(x_intervals[c][1] - x_intervals[c + 1][0] for c in range(len(x_intervals) - 1))
            max_seam_h = max(y_intervals[r][1] - y_intervals[r + 1][0] for r in range(len(y_intervals) - 1))

            # Ensure patch covers seam with margin for smooth roll-off (at least 1.5x seam or 0.5x tile)
            # and strictly locks to native tile aspect ratio in multiples of 32
            min_w = max(max_seam_w * 1.5, tile_w * 0.5, 256)
            min_h = max(max_seam_h * 1.5, tile_h * 0.5, 192)
            if aspect >= 1.0:
                patch_w = max(256, int(round(min_w / 32.0)) * 32)
                patch_h = max(192, int(round((patch_w / aspect) / 32.0)) * 32)
            else:
                patch_h = max(256, int(round(min_h / 32.0)) * 32)
                patch_w = max(192, int(round((patch_h * aspect) / 32.0)) * 32)
            patch_w = min(patch_w, target_w)
            patch_h = min(patch_h, target_h)

            p_idx = 0
            for r in range(len(y_intervals) - 1):
                y_seam_start = y_intervals[r + 1][0]
                y_seam_end = y_intervals[r][1]
                y_center = (y_seam_start + y_seam_end) // 2
                y1_p = max(0, min(target_h - patch_h, int(round(y_center / 32.0)) * 32 - patch_h // 2))
                y2_p = y1_p + patch_h

                for c in range(len(x_intervals) - 1):
                    x_seam_start = x_intervals[c + 1][0]
                    x_seam_end = x_intervals[c][1]
                    x_center = (x_seam_start + x_seam_end) // 2
                    x1_p = max(0, min(target_w - patch_w, int(round(x_center / 32.0)) * 32 - patch_w // 2))
                    x2_p = x1_p + patch_w
                    p_idx += 1
                    spatial_tiles.append({
                        "coords": (x1_p, x2_p, y1_p, y2_p),
                        "seam_box": (x_seam_start, x_seam_end, y_seam_start, y_seam_end),
                        "is_center_patch": True,
                        "grid_pos": (r, c),
                        "desc": f"center_patch_{p_idx}(r{r+1}-{r+2},c{c+1}-{c+2})"
                    })

        actual_mp = (tile_w * tile_h) / 1e6
        center_patches_count = len(spatial_tiles) - (len(x_intervals) * len(y_intervals))
        print(f"[H3LatentTiledKSampler] Grid: {len(x_intervals)}x{len(y_intervals)} ({len(x_intervals)*len(y_intervals)} tiles + {center_patches_count} center patches) | Tile: {tile_w}x{tile_h} ({actual_mp:.2f} MP) | Tight overlap: {tight_tile_overlap}")

        # 2. Allocate In-Memory Canvas Accumulators (Only 57.5 MB for 4K!)
        delta_acc = torch.zeros_like(canvas_latent, device="cpu", dtype=torch.float32)
        weight_acc = torch.zeros((1, 1, T_lat, H_lat, W_lat), dtype=torch.float32, device="cpu")
        cached_anchor_rgb = None

        is_av_model = False
        if hasattr(model, "is_av_model") and callable(model.is_av_model):
            is_av_model = model.is_av_model()
        elif hasattr(model, "model"):
            m_cls_name = model.model.__class__.__name__
            m_type = getattr(model.model, "model_type", None)
            flow_av = getattr(comfy.model_base.ModelType, "FLOW_AV", None) if (comfy is not None and hasattr(comfy, "model_base")) else None
            if "MiniMax" in m_cls_name or (flow_av is not None and m_type == flow_av) or "MiniMax" in str(type(model.model)):
                is_av_model = True
        elif audio_vae is not None:
            is_av_model = True

        # Setup overall ComfyUI progress bar across all temporal chunks and spatial tiles
        total_spatial_tiles = len(spatial_tiles)
        total_tiles = len(temporal_chunks) * total_spatial_tiles
        steps_per_tile = max(1, len(sigmas) - 1) if (sigmas is not None and hasattr(sigmas, "__len__")) else 20
        total_progress_steps = total_tiles * steps_per_tile

        pbar = comfy.utils.ProgressBar(total_progress_steps) if (comfy is not None and hasattr(comfy, "utils")) else None

        cmd_pbar = None
        try:
            from tqdm.auto import tqdm
            cmd_pbar = tqdm(
                total=total_progress_steps,
                desc="[H3LatentTiledKSampler]",
                unit="step",
                dynamic_ncols=True,
                leave=True
            )
        except Exception:
            cmd_pbar = None

        previewer = None
        try:
            import latent_preview
            load_dev = getattr(model, "load_device", "cpu")
            latent_fmt = getattr(getattr(model, "model", None), "latent_format", None)
            if latent_fmt is not None:
                previewer = latent_preview.get_previewer(load_dev, latent_fmt)
        except Exception:
            previewer = None

        completed_steps = [0]

        # Pre-encode reference bundle once before chunk/tile loops if precaching enabled
        base_ref_items = []
        base_ref_blocks = []
        has_video_ref = False
        if reference_bundle is not None:
            if isinstance(reference_bundle, dict):
                has_video_ref = len(reference_bundle.get("videos", [])) > 0
            elif hasattr(reference_bundle, "videos"):
                has_video_ref = len(getattr(reference_bundle, "videos", [])) > 0

        if precache_tile_conditioning and reference_bundle is not None:
            try:
                dummy_cond = [[torch.zeros((1, 1, 2048), dtype=torch.float32), {}]]
                injected = inject_reference_bundle(dummy_cond, reference_bundle, vae, width=tile_w, height=tile_h)
                if injected and len(injected) > 0:
                    base_ref_items = list(injected[0][1].get("minimax_ref_items", []))
                    base_ref_blocks = list(injected[0][1].get("minimax_refs", injected[0][1].get("minimax_ref_blocks", [])))
            except Exception as e_bnd:
                print(f"[H3LatentTiledKSampler] Reference bundle pre-encode notice: {e_bnd}")

        # 3. Outer Loop: Temporal Chunks (k = 0, 1, 2...)
        for chunk in temporal_chunks:
            k = chunk.chunk_idx
            c_lat_start = (chunk.start_frame // 17) * 5
            c_lat_end = min(T_lat, ((chunk.end_frame + 16) // 17) * 5) if chunk.end_frame < total_frames else T_lat

            # Prepare globally coherent canvas noise for temporal chunk k
            # Deriving tile noise from this shared canvas ensures overlapping regions have 100% identical noise,
            # eliminating motion tearing and tile seam divergence.
            base_seed = int(seed) if seed is not None else 0
            chunk_seed = (base_seed + k * 10007) & 0xffffffffffffffff
            chunk_canvas_latent = canvas_latent[:, :, c_lat_start:c_lat_end]
            if comfy is not None and hasattr(comfy, "sample") and hasattr(comfy.sample, "prepare_noise"):
                global_chunk_noise = comfy.sample.prepare_noise(chunk_canvas_latent, chunk_seed)
            else:
                torch.manual_seed(chunk_seed)
                global_chunk_noise = torch.randn_like(chunk_canvas_latent)

            base_prompt_text = ""
            if chunk_prompts:
                if isinstance(chunk_prompts, dict):
                    base_prompt_text = chunk_prompts.get(k, chunk_prompts.get(str(k), ""))
                elif isinstance(chunk_prompts, str):
                    base_prompt_text = chunk_prompts

            # Pre-encode all spatial tile conditionings for chunk k upfront if precaching enabled
            precached_tile_conds = {}
            if precache_tile_conditioning:
                chunk_ref_items = list(base_ref_items)
                chunk_ref_blocks = list(base_ref_blocks)
                anchor_prompt_addition = ""
                if propagate_keyframes and k > 0 and cached_anchor_rgb is not None and vae is not None:
                    try:
                        prop_rgb = cached_anchor_rgb
                        kh, kw = prop_rgb.shape[1], prop_rgb.shape[2]
                        ptw = max(32, round(kw / 32) * 32)
                        pth = max(32, round(kh / 32) * 32)
                        if (ptw != kw or pth != kh) and comfy is not None and hasattr(comfy, "utils"):
                            prop_rgb = comfy.utils.common_upscale(
                                prop_rgb.movedim(-1, 1), ptw, pth, "lanczos", "disabled"
                            ).movedim(1, -1)
                        dev = vae.device if hasattr(vae, "device") else "cuda" if torch.cuda.is_available() else "cpu"
                        with torch.no_grad():
                            z_prop = vae.encode(prop_rgb.to(dev))
                        if isinstance(z_prop, torch.Tensor):
                            z_prop = z_prop.detach().cpu()
                            if z_prop.shape[-2] % 2 != 0 or z_prop.shape[-1] % 2 != 0:
                                z_prop = torch.nn.functional.pad(z_prop, (0, z_prop.shape[-1] % 2, 0, z_prop.shape[-2] % 2), mode="replicate")
                            next_pic_idx = sum(1 for b in chunk_ref_blocks if b.get("kind") == "image") + 1
                            chunk_ref_items.append({"type": "image", "data": prop_rgb.cpu()})
                            chunk_ref_blocks.append({
                                "kind": "image",
                                "latent": z_prop,
                                "latent_h": z_prop.shape[-2],
                                "latent_w": z_prop.shape[-1],
                                "index": next_pic_idx,
                                "is_keyframe_anchor": True
                            })
                            anchor_prompt_addition = f"\n<Picture {next_pic_idx}> is the preceding scene anchor from the previous temporal shot, preserving continuous appearance and elevated micro-detail."
                    except Exception as e_prop:
                        print(f"[H3LatentTiledKSampler] Anchor injection notice: {e_prop}")

                tile_prompt = build_tile_ref2va_prompt(
                    tile_idx=0,
                    row=0, col=0,
                    total_rows=len(y_intervals), total_cols=len(x_intervals),
                    user_prompt=(base_prompt_text + anchor_prompt_addition),
                    has_video_ref=has_video_ref
                )
                p_cond = []
                if clip is not None:
                    try:
                        try:
                            tokens = clip.tokenize(tile_prompt, minimax_ref_items=chunk_ref_items if chunk_ref_items else None)
                        except TypeError:
                            tokens = clip.tokenize(tile_prompt)
                        if hasattr(clip, "encode_from_tokens_scheduled"):
                            p_cond = clip.encode_from_tokens_scheduled(tokens)
                        else:
                            cond_out = clip.encode_from_tokens(tokens)
                            if isinstance(cond_out, torch.Tensor):
                                p_cond = [[cond_out, {}]]
                            elif isinstance(cond_out, (list, tuple)):
                                p_cond = list(cond_out)
                            else:
                                p_cond = [[cond_out, {}]]
                    except Exception as e_clip:
                        print(f"[H3LatentTiledKSampler] CLIP encode notice: {e_clip}")
                elif positive is not None:
                    p_cond = [list(ci) if isinstance(ci, (list, tuple)) else [ci, {}] for ci in positive]

                if not p_cond:
                    p_cond = [[torch.zeros((1, 1, 2048), dtype=torch.float32), {}]]

                if chunk_ref_blocks:
                    try:
                        import node_helpers
                        p_cond = node_helpers.conditioning_set_values(p_cond, {"minimax_refs": chunk_ref_blocks})
                    except Exception:
                        for c_item in p_cond:
                            c_item[1]["minimax_refs"] = chunk_ref_blocks
                            c_item[1]["minimax_ref_blocks"] = chunk_ref_blocks

                for s_idx in range(len(spatial_tiles)):
                    precached_tile_conds[s_idx] = p_cond

                # Offload CLIP and VAE once before spatial tile sampling begins
                if offload_device == "cpu" and comfy is not None and hasattr(comfy, "model_management"):
                    try:
                        if clip is not None:
                            if hasattr(clip, "patcher") and clip.patcher is not None:
                                comfy.model_management.unload_model_and_clones(clip.patcher)
                            elif hasattr(clip, "cond_stage_model"):
                                clip.cond_stage_model.to("cpu")
                        if vae is not None:
                            if hasattr(vae, "patcher") and vae.patcher is not None:
                                comfy.model_management.unload_model_and_clones(vae.patcher)
                            elif hasattr(vae, "first_stage_model"):
                                vae.first_stage_model.to("cpu")
                        comfy.model_management.soft_empty_cache()
                    except Exception:
                        pass

            # Inner Loop: Spatial Tiles
            for s_idx, tile_meta in enumerate(spatial_tiles):
                x1, x2, y1, y2 = tile_meta["coords"]
                is_center_patch = tile_meta["is_center_patch"]
                r, c = tile_meta["grid_pos"]
                tile_desc = tile_meta["desc"]
                tile_index = k * total_spatial_tiles + s_idx
                if cmd_pbar is not None:
                    cmd_pbar.set_description(
                        f"[H3LatentTiledKSampler] Chunk {k+1}/{len(temporal_chunks)} | Tile {tile_index+1}/{total_tiles} {tile_desc}"
                    )
                lat_y1, lat_y2 = y1 // 16, y2 // 16
                lat_x1, lat_x2 = x1 // 16, x2 // 16

                # Extract 5D sub-tensor crop (zero VAE calls)
                tile_crop = canvas_latent[:, :, c_lat_start:c_lat_end, lat_y1:lat_y2, lat_x1:lat_x2].clone()
                tile_noise_crop = global_chunk_noise[:, :, :, lat_y1:lat_y2, lat_x1:lat_x2].clone()

                # MiniMax DiT requires 2x2 spatial latent patches (even spatial dimensions)
                pad_h = 0
                pad_w = 0
                if tile_crop.shape[-2] % 2 != 0:
                    pad_h = 1
                if tile_crop.shape[-1] % 2 != 0:
                    pad_w = 1
                if pad_h > 0 or pad_w > 0:
                    tile_crop = torch.nn.functional.pad(tile_crop, (0, pad_w, 0, pad_h), mode="replicate")
                    tile_noise_crop = torch.nn.functional.pad(tile_noise_crop, (0, pad_w, 0, pad_h), mode="replicate")

                if precache_tile_conditioning:
                    positive_cond = precached_tile_conds.get(s_idx, precached_tile_conds.get((r, c), [[torch.zeros((1, 1, 2048), dtype=torch.float32), {}]]))
                else:
                    # Extract reference items and blocks from bundle (if available)
                    ref_items = []
                    ref_blocks = []
                    if reference_bundle is not None:
                        dummy_cond = [[torch.zeros((1, 1, 2048), dtype=torch.float32), {}]]
                        injected = inject_reference_bundle(dummy_cond, reference_bundle, vae, width=x2 - x1, height=y2 - y1)
                        if injected and len(injected) > 0:
                            ref_items = list(injected[0][1].get("minimax_ref_items", []))
                            ref_blocks = list(injected[0][1].get("minimax_refs", injected[0][1].get("minimax_ref_blocks", [])))

                        # Auto-Regressive Keyframe Propagation for Temporal Chunks k > 0
                        anchor_prompt_addition = ""
                        if propagate_keyframes and k > 0 and cached_anchor_rgb is not None and vae is not None:
                            try:
                                prop_rgb = cached_anchor_rgb
                                kh, kw = prop_rgb.shape[1], prop_rgb.shape[2]
                                ptw = max(32, round(kw / 32) * 32)
                                pth = max(32, round(kh / 32) * 32)
                                if ptw != kw or pth != kh and comfy is not None and hasattr(comfy, "utils"):
                                    prop_rgb = comfy.utils.common_upscale(
                                        prop_rgb.movedim(-1, 1), ptw, pth, "lanczos", "disabled"
                                    ).movedim(1, -1)
                                dev = vae.device if hasattr(vae, "device") else "cuda" if torch.cuda.is_available() else "cpu"
                                with torch.no_grad():
                                    z_prop = vae.encode(prop_rgb.to(dev))
                                if isinstance(z_prop, torch.Tensor):
                                    z_prop = z_prop.detach().cpu()
                                    if z_prop.shape[-2] % 2 != 0 or z_prop.shape[-1] % 2 != 0:
                                        z_prop = torch.nn.functional.pad(z_prop, (0, z_prop.shape[-1] % 2, 0, z_prop.shape[-2] % 2), mode="replicate")
                                    next_pic_idx = sum(1 for b in ref_blocks if b.get("kind") == "image") + 1
                                    ref_items.append({"type": "image", "data": prop_rgb.cpu()})
                                    ref_blocks.append({
                                        "kind": "image",
                                        "latent": z_prop,
                                        "latent_h": z_prop.shape[-2],
                                        "latent_w": z_prop.shape[-1],
                                        "index": next_pic_idx,
                                        "is_keyframe_anchor": True
                                    })
                                    anchor_prompt_addition = f"\n<Picture {next_pic_idx}> is the preceding scene anchor from the previous temporal shot, preserving continuous appearance and elevated micro-detail."
                            except Exception as e_prop:
                                print(f"[H3LatentTiledKSampler] Anchor injection notice: {e_prop}")

                        # Build conditioning dynamically via clip
                        tile_prompt = build_tile_ref2va_prompt(
                            tile_idx=s_idx,
                            row=r, col=c,
                            total_rows=len(y_intervals), total_cols=len(x_intervals),
                            user_prompt=(base_prompt_text + anchor_prompt_addition),
                            has_video_ref=has_video_ref
                        )

                        positive_cond = []
                        if clip is not None:
                            try:
                                try:
                                    tokens = clip.tokenize(tile_prompt, minimax_ref_items=ref_items if ref_items else None)
                                except TypeError:
                                    tokens = clip.tokenize(tile_prompt)
                                if hasattr(clip, "encode_from_tokens_scheduled"):
                                    positive_cond = clip.encode_from_tokens_scheduled(tokens)
                                else:
                                    cond_out = clip.encode_from_tokens(tokens)
                                    if isinstance(cond_out, torch.Tensor):
                                        positive_cond = [[cond_out, {}]]
                                    elif isinstance(cond_out, (list, tuple)):
                                        positive_cond = list(cond_out)
                                    else:
                                        positive_cond = [[cond_out, {}]]
                            except Exception as e_clip:
                                print(f"[H3LatentTiledKSampler] CLIP encode notice: {e_clip}")
                        elif positive is not None:
                            positive_cond = [list(ci) if isinstance(ci, (list, tuple)) else [ci, {}] for ci in positive]

                        if not positive_cond:
                            positive_cond = [[torch.zeros((1, 1, 2048), dtype=torch.float32), {}]]

                        if ref_blocks:
                            try:
                                import node_helpers
                                positive_cond = node_helpers.conditioning_set_values(positive_cond, {"minimax_refs": ref_blocks})
                            except Exception:
                                for c_item in positive_cond:
                                    c_item[1]["minimax_refs"] = ref_blocks
                                    c_item[1]["minimax_ref_blocks"] = ref_blocks

                # Package NestedTensor if AV model
                sampling_latent = tile_crop
                sampling_noise = tile_noise_crop
                if is_av_model and comfy is not None and hasattr(comfy, "nested_tensor"):
                    c_tokens = c_lat_end - c_lat_start
                    chunk_frames = (c_tokens - 1) * 4 + 1 if c_tokens > 1 else 1
                    fps_val = 24.0
                    duration = chunk_frames / fps_val
                    audio_t = max(1, round(duration * 40))

                    audio_latent = None
                    if isinstance(latent_image, dict) and "audio_samples" in latent_image:
                        c_aud = latent_image["audio_samples"]
                        if isinstance(c_aud, torch.Tensor) and c_aud.ndim >= 4:
                            chunk_start_f = chunk.start_frame
                            a_start = max(0, round((chunk_start_f / fps_val) * 40))
                            a_end = a_start + audio_t
                            if a_end <= c_aud.shape[-1]:
                                audio_latent = c_aud[:, :, :, a_start:a_end].clone()
                            else:
                                a_slice = c_aud[:, :, :, a_start:]
                                pad_amt = audio_t - a_slice.shape[-1]
                                audio_latent = torch.nn.functional.pad(a_slice, (0, max(0, pad_amt)))

                    if audio_latent is None:
                        audio_latent = torch.zeros([tile_crop.shape[0], 32, 2, audio_t], device=tile_crop.device, dtype=tile_crop.dtype)
                    elif audio_latent.shape[-1] != audio_t:
                        if audio_latent.shape[-1] > audio_t:
                            audio_latent = audio_latent[..., :audio_t]
                        else:
                            pad_amt = audio_t - audio_latent.shape[-1]
                            audio_latent = torch.nn.functional.pad(audio_latent, (0, pad_amt))
                    audio_latent = audio_latent.to(device=tile_crop.device, dtype=tile_crop.dtype)

                    sampling_latent = comfy.nested_tensor.NestedTensor((tile_crop, audio_latent))
                    if comfy is not None and hasattr(comfy, "sample") and hasattr(comfy.sample, "prepare_noise"):
                        audio_noise = comfy.sample.prepare_noise(audio_latent, chunk_seed)
                    else:
                        torch.manual_seed(chunk_seed)
                        audio_noise = torch.randn_like(audio_latent)
                    sampling_noise = comfy.nested_tensor.NestedTensor((tile_noise_crop, audio_noise))

                # Offload CLIP and VAE from GPU before DiT sampling if requested (legacy non-precached mode)
                if not precache_tile_conditioning and offload_device == "cpu" and comfy is not None and hasattr(comfy, "model_management"):
                    try:
                        if clip is not None:
                            if hasattr(clip, "patcher") and clip.patcher is not None:
                                comfy.model_management.unload_model_and_clones(clip.patcher)
                            elif hasattr(clip, "cond_stage_model"):
                                clip.cond_stage_model.to("cpu")
                        if vae is not None:
                            if hasattr(vae, "patcher") and vae.patcher is not None:
                                comfy.model_management.unload_model_and_clones(vae.patcher)
                            elif hasattr(vae, "first_stage_model"):
                                vae.first_stage_model.to("cpu")
                        comfy.model_management.soft_empty_cache()
                    except Exception:
                        pass

                # Execute DiT Diffusion Sampling on tile sub-tensor with coherent canvas noise
                def tile_callback(step, x0, x, total_steps):
                    completed_steps[0] += 1
                    if cmd_pbar is not None:
                        cmd_pbar.update(1)
                    preview_bytes = None
                    if previewer is not None and x0 is not None:
                        try:
                            x0_p = x0.unbind()[0] if getattr(x0, "is_nested", False) else x0
                            if hasattr(x0_p, "tensors"):
                                x0_p = x0_p.tensors[0]
                            preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0_p)
                        except Exception:
                            preview_bytes = None
                    if pbar is not None:
                        pbar.update_absolute(completed_steps[0], total_progress_steps, preview=preview_bytes)

                enhanced_tile = tile_crop
                if comfy is not None and hasattr(comfy, "sample") and hasattr(comfy.sample, "sample_custom"):
                    samples = comfy.sample.sample_custom(
                        model, sampling_noise, cfg, sampler, sigmas, positive_cond, [], sampling_latent,
                        callback=tile_callback,
                        disable_pbar=True,
                        seed=chunk_seed
                    )
                    enhanced_tile = samples.unbind()[0] if getattr(samples, "is_nested", False) else samples

                    # Ensure progress bar accounts for full tile completion
                    tile_end_steps = (tile_index + 1) * steps_per_tile
                    if completed_steps[0] < tile_end_steps:
                        step_diff = tile_end_steps - completed_steps[0]
                        completed_steps[0] = tile_end_steps
                        if cmd_pbar is not None:
                            cmd_pbar.update(step_diff)
                        if pbar is not None:
                            pbar.update_absolute(completed_steps[0], total_progress_steps)
                elif hasattr(model, "sample"):
                    enhanced = model.sample(sampling_latent, positive_cond)
                    enhanced_tile = enhanced.unbind()[0] if getattr(enhanced, "is_nested", False) else enhanced
                    tile_end_steps = (tile_index + 1) * steps_per_tile
                    step_diff = tile_end_steps - completed_steps[0]
                    completed_steps[0] = tile_end_steps
                    if cmd_pbar is not None:
                        cmd_pbar.update(step_diff)
                    if pbar is not None:
                        pbar.update_absolute(completed_steps[0], total_progress_steps)

                # Crop back any spatial padding if added for 2x2 DiT patch
                if pad_h > 0 or pad_w > 0:
                    end_h = enhanced_tile.shape[-2] - pad_h
                    end_w = enhanced_tile.shape[-1] - pad_w
                    enhanced_tile = enhanced_tile[..., :end_h, :end_w]

                # Offload enhanced tile immediately if using temp_disk_stream
                if tile_storage_strategy == "temp_disk_stream":
                    dump_r = 1000 + r if is_center_patch else r
                    tile_ref = dump_tile_to_disk_stream(enhanced_tile, k, dump_r, c)
                    enhanced_tile = tile_ref

                chunk_tokens = c_lat_end - c_lat_start
                temp_weight = torch.ones((1, 1, chunk_tokens, 1, 1), dtype=torch.float32)

                # Seamless raised-cosine fade-in from chunk k-1
                if k > 0:
                    prev_chunk = temporal_chunks[k - 1]
                    prev_c_lat_end = min(T_lat, ((prev_chunk.end_frame + 16) // 17) * 5) if prev_chunk.end_frame < total_frames else T_lat
                    head_blend_tokens = min(chunk_tokens // 2, max(0, prev_c_lat_end - c_lat_start))
                    if head_blend_tokens > 0:
                        fade_in = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, head_blend_tokens))
                        temp_weight[:, :, :head_blend_tokens, :, :] = fade_in.view(1, 1, -1, 1, 1)

                # Seamless raised-cosine fade-out into chunk k+1
                if k < len(temporal_chunks) - 1:
                    next_chunk = temporal_chunks[k + 1]
                    next_c_lat_start = (next_chunk.start_frame // 17) * 5
                    tail_blend_tokens = min(chunk_tokens // 2, max(0, c_lat_end - next_c_lat_start))
                    if tail_blend_tokens > 0:
                        fade_out = 0.5 + 0.5 * torch.cos(torch.linspace(0, torch.pi, tail_blend_tokens))
                        temp_weight[:, :, -tail_blend_tokens:, :, :] = fade_out.view(1, 1, -1, 1, 1)

                tile_data = load_tile(enhanced_tile).cpu()
                tile_base = canvas_latent[:, :, c_lat_start:c_lat_end, lat_y1:lat_y2, lat_x1:lat_x2].cpu()

                if not is_center_patch:
                    ov_top = (y_intervals[r - 1][1] - y1) // 16 if r > 0 else 0
                    ov_bottom = (y2 - y_intervals[r + 1][0]) // 16 if r < len(y_intervals) - 1 else 0
                    ov_left = (x_intervals[c - 1][1] - x1) // 16 if c > 0 else 0
                    ov_right = (x2 - x_intervals[c + 1][0]) // 16 if c < len(x_intervals) - 1 else 0

                    tile_weight = generate_hann_weight_2d(
                        h=lat_y2 - lat_y1,
                        w=lat_x2 - lat_x1,
                        fade_top=(r > 0),
                        fade_bottom=(r < len(y_intervals) - 1),
                        fade_left=(c > 0),
                        fade_right=(c < len(x_intervals) - 1),
                        ov_top=ov_top,
                        ov_bottom=ov_bottom,
                        ov_left=ov_left,
                        ov_right=ov_right,
                        blend_mode=spatial_blend_mode
                    )
                    full_weight = tile_weight * temp_weight
                    tile_delta = tile_data - tile_base
                    delta_acc[:, :, c_lat_start:c_lat_end, lat_y1:lat_y2, lat_x1:lat_x2] += tile_delta * full_weight
                    weight_acc[:, :, c_lat_start:c_lat_end, lat_y1:lat_y2, lat_x1:lat_x2] += full_weight
                else:
                    # Clean Partition-of-Unity Override Blending for Center Patch
                    # Overrides the 4-tile seam junction with 100% single-tile crispness and smooth Hann margin roll-off
                    seam_box = tile_meta.get("seam_box")
                    lat_w = lat_x2 - lat_x1
                    lat_h = lat_y2 - lat_y1

                    if seam_box is not None:
                        x_seam_s, x_seam_e, y_seam_s, y_seam_e = seam_box
                        lat_sx1 = max(0, min(lat_w, x_seam_s // 16 - lat_x1))
                        lat_sx2 = max(lat_sx1, min(lat_w, x_seam_e // 16 - lat_x1))
                        lat_sy1 = max(0, min(lat_h, y_seam_s // 16 - lat_y1))
                        lat_sy2 = max(lat_sy1, min(lat_h, y_seam_e // 16 - lat_y1))
                    else:
                        lat_sx1, lat_sx2 = lat_w // 4, 3 * lat_w // 4
                        lat_sy1, lat_sy2 = lat_h // 4, 3 * lat_h // 4

                    def make_flat_hann_ramp(length: int, start_idx: int, end_idx: int) -> torch.Tensor:
                        ramp = torch.ones(length, dtype=torch.float32)
                        if start_idx > 0:
                            t_in = torch.linspace(0, torch.pi, start_idx)
                            ramp[:start_idx] = 0.5 - 0.5 * torch.cos(t_in)
                        if end_idx < length:
                            m_out = length - end_idx
                            t_out = torch.linspace(0, torch.pi, m_out)
                            ramp[end_idx:] = 0.5 + 0.5 * torch.cos(t_out)
                        return ramp

                    rx = make_flat_hann_ramp(lat_w, lat_sx1, lat_sx2)
                    ry = make_flat_hann_ramp(lat_h, lat_sy1, lat_sy2)
                    patch_weight = (ry[:, None] * rx[None, :]).unsqueeze(0).unsqueeze(0).unsqueeze(0)

                    # Extract primary consensus already accumulated in this patch crop
                    w_crop = weight_acc[:, :, c_lat_start:c_lat_end, lat_y1:lat_y2, lat_x1:lat_x2].clamp_min(1e-5)
                    primary_norm = delta_acc[:, :, c_lat_start:c_lat_end, lat_y1:lat_y2, lat_x1:lat_x2] / w_crop

                    patch_delta = tile_data - tile_base

                    # DC offset alignment in transition zone to guarantee zero luminance steps
                    trans_mask = (patch_weight > 0.05) & (patch_weight < 0.95)
                    if trans_mask.any():
                        mask_2d = trans_mask[0, 0, 0]
                        diff = primary_norm - patch_delta
                        dc_diff = diff[:, :, :, mask_2d].mean(dim=-1, keepdim=True).unsqueeze(-1)
                        patch_delta = patch_delta + dc_diff

                    # Exact partition of unity override: (1 - Wp) * primary + Wp * patch
                    patch_blend_mask = patch_weight * temp_weight
                    blended_delta = (1.0 - patch_blend_mask) * primary_norm + patch_blend_mask * patch_delta
                    delta_acc[:, :, c_lat_start:c_lat_end, lat_y1:lat_y2, lat_x1:lat_x2] = blended_delta * w_crop

            def compute_normalized_canvas(t_start: int, t_end: int) -> torch.Tensor:
                w_slice = weight_acc[:, :, t_start:t_end].clamp_min(1e-5)
                norm_delta = delta_acc[:, :, t_start:t_end] / w_slice
                base_slice = canvas_latent[:, :, t_start:t_end].cpu()
                return (base_slice + norm_delta).to(dtype=canvas_latent.dtype)

            # Step 1: Trailing Anchor Capture (End of Chunk k)
            if propagate_keyframes and vae is not None:
                try:
                    clip_tokens = min(c_lat_end - c_lat_start, 5)
                    chunk_trailing_lat = compute_normalized_canvas(c_lat_end - clip_tokens, c_lat_end)
                    dev = vae.device if hasattr(vae, "device") else "cuda" if torch.cuda.is_available() else "cpu"
                    with torch.no_grad():
                        dec_clip = vae.decode(chunk_trailing_lat.to(dev))
                    if isinstance(dec_clip, torch.Tensor):
                        dec_clip = dec_clip.detach().cpu()
                        if dec_clip.ndim == 5:
                            dec_clip = dec_clip.reshape(-1, dec_clip.shape[-3], dec_clip.shape[-2], dec_clip.shape[-1])
                        # dec_clip is [T, H, W, C]; take the last single frame [1, H, W, 3]
                        cached_anchor_rgb = dec_clip[-1:, :, :, :3].clamp(0.0, 1.0).float()
                    if offload_device == "cpu" and hasattr(vae, "to"):
                        vae.to("cpu")
                except Exception as e_anc:
                    print(f"[H3LatentTiledKSampler] Anchor capture notice: {e_anc}")
                    cached_anchor_rgb = None

            # Optional Debug Decode of Assembled Temporal Chunk
            if debug_decode_chunks and vae is not None:
                try:
                    chunk_lat = compute_normalized_canvas(c_lat_start, c_lat_end)
                    export_debug_chunk_mp4(chunk_lat, k, debug_chunk_output_dir, vae, audio_vae, latent_image)
                except Exception as e_dbg:
                    print(f"[H3LatentTiledKSampler] Debug export notice: {e_dbg}")

        if cmd_pbar is not None:
            cmd_pbar.close()

        # 4. Final Normalization
        final_video_latent = compute_normalized_canvas(0, T_lat)

        # Package as standard ComfyUI audio-video NestedTensor so default VAEDecode and VAEDecodeAudio work out-of-the-box
        out_latent = package_latent(
            final_video_latent,
            input_audio_latent,
            extra_dict=latent_image if isinstance(latent_image, dict) else None
        )

        return (out_latent,)
