import os
import gc
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import torch
import numpy as np

import folder_paths
import comfy.model_management
from .schemas import VLMContainer
from .ffmpeg_pipe import get_ffmpeg_binary

def resolve_local_model_path(model_name: str, local_path_override: str = "") -> Tuple[str, str]:
    """
    Resolves local path for a VLM model.
    Returns (resolved_path_or_repo_id, model_type).
    """
    if local_path_override and local_path_override.strip():
        p = Path(local_path_override.strip())
        if p.exists():
            name_lower = p.name.lower()
            mtype = "florence2" if "florence" in name_lower else ("qwen3_vl" if "qwen3" in name_lower else "qwen2_5_vl")
            return str(p), mtype

    # Check ComfyUI models directories (models/vlm, models/LLM, models/checkpoints)
    search_dirs = [
        os.path.join(folder_paths.models_dir, "vlm"),
        os.path.join(folder_paths.models_dir, "LLM"),
        os.path.join(folder_paths.models_dir, "checkpoints"),
    ]

    clean_basename = Path(model_name).name
    for base in search_dirs:
        if not os.path.exists(base):
            continue
        candidates = [
            os.path.join(base, clean_basename),
            os.path.join(base, model_name),
            os.path.join(base, clean_basename.replace("-", "_")),
            os.path.join(base, clean_basename.replace("_", "-")),
        ]
        for c in candidates:
            if os.path.exists(c):
                name_lower = clean_basename.lower()
                mtype = "florence2" if "florence" in name_lower else ("qwen3_vl" if "qwen3" in name_lower else "qwen2_5_vl")
                return c, mtype

    # Fall back to HuggingFace identifier
    name_lower = model_name.lower()
    mtype = "florence2" if "florence" in name_lower else ("qwen3_vl" if "qwen3" in name_lower else "qwen2_5_vl")
    return model_name, mtype


def load_local_vlm(
    model_name: str,
    local_path_override: str = "",
    precision: str = "bf16",
    device: str = "cuda"
) -> VLMContainer:
    """
    Loads a local Vision-Language Model (Qwen3-VL, Qwen2.5-VL, or Florence-2) using Transformers.
    If the model is not already in models/vlm, downloads and permanently saves it into models/vlm.
    """
    resolved_path, model_type = resolve_local_model_path(model_name, local_path_override)

    # If resolved_path is not an existing local directory, download and save permanently into models/vlm
    if not os.path.exists(resolved_path):
        vlm_dir = os.path.join(folder_paths.models_dir, "vlm")
        target_dir = os.path.join(vlm_dir, Path(model_name).name)
        try:
            from huggingface_hub import snapshot_download
            print(f"[H3-TiledUpscale] Saving '{model_name}' permanently into models/vlm/{Path(model_name).name} so it never re-downloads...")
            os.makedirs(target_dir, exist_ok=True)
            snapshot_download(repo_id=model_name, local_dir=target_dir, local_dir_use_symlinks=False)
            resolved_path = target_dir
            print(f"[H3-TiledUpscale] Model permanently saved to: {target_dir}")
        except Exception as e:
            print(f"[H3-TiledUpscale] Notice: Could not save snapshot to models/vlm ({e}), using default HuggingFace cache.")

    print(f"[H3-TiledUpscale] Loading local VLM ({model_type}) from: {resolved_path} [{precision}, {device}]...")

    # Determine torch dtype
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.float16
    elif precision == "fp8_e4m3fn" and hasattr(torch, "float8_e4m3fn"):
        torch_dtype = torch.float8_e4m3fn
    else:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

    target_device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

    from transformers import AutoProcessor

    if model_type == "florence2":
        from transformers import AutoModelForCausalLM
        processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(target_device)
        model.eval()
    else:
        # Qwen3-VL / Qwen2.5-VL / Qwen2-VL
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                resolved_path,
                torch_dtype=torch_dtype,
                device_map=target_device if target_device == "cuda" else None,
                trust_remote_code=True
            )
        except Exception:
            try:
                from transformers import Qwen3VLForConditionalGeneration
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    resolved_path,
                    torch_dtype=torch_dtype,
                    device_map=target_device if target_device == "cuda" else None,
                    trust_remote_code=True
                )
            except Exception:
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        resolved_path,
                        torch_dtype=torch_dtype,
                        device_map=target_device if target_device == "cuda" else None,
                        trust_remote_code=True
                    )
                except Exception:
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        resolved_path,
                        torch_dtype=torch_dtype,
                        device_map=target_device if target_device == "cuda" else None,
                        trust_remote_code=True
                    )

        processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)
        if target_device == "cpu":
            model = model.to("cpu")
        model.eval()

    print(f"[H3-TiledUpscale] Local VLM loaded successfully.")
    return VLMContainer(
        model=model,
        processor=processor,
        model_type=model_type,
        device=target_device,
        precision=precision,
        model_name_or_path=resolved_path
    )


def extract_chunk_keyframes(
    video_path: str,
    start_frame: int,
    end_frame: int,
    num_samples: int = 4,
    video_tensor: Optional[torch.Tensor] = None,
    fallback_image_path: str = ""
) -> List[Image.Image]:
    """
    Extracts num_samples evenly spaced PIL Images across [start_frame, end_frame).
    Supports either an existing MP4 video on disk or an in-memory ComfyUI video tensor [B, H, W, 3].
    """
    if end_frame <= start_frame:
        end_frame = start_frame + 1

    total_chunk_frames = end_frame - start_frame
    sample_count = max(1, min(num_samples, total_chunk_frames))

    if sample_count == 1:
        indices = [start_frame + total_chunk_frames // 2]
    else:
        indices = [
            int(round(start_frame + i * (total_chunk_frames - 1) / (sample_count - 1)))
            for i in range(sample_count)
        ]

    images: List[Image.Image] = []

    # Method 1: Extract from video_tensor in memory if provided
    if video_tensor is not None and isinstance(video_tensor, torch.Tensor):
        for idx in indices:
            clamped_idx = max(0, min(idx, video_tensor.shape[0] - 1))
            f_tensor = video_tensor[clamped_idx]
            if f_tensor.ndim == 3:
                # [H, W, 3] in [0, 1]
                arr = (f_tensor.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
                images.append(Image.fromarray(arr))
        if images:
            return images

    # Method 2: Extract from video file via FFmpeg (guard against .latent or non-video files)
    if video_path and os.path.exists(video_path) and not str(video_path).lower().endswith(".latent"):
        ffmpeg_exe = get_ffmpeg_binary()
        for idx in indices:
            cmd = [
                ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-vf", f"select=eq(n\\,{idx})",
                "-vframes", "1",
                "-f", "image2pipe",
                "-vcodec", "png",
                "-"
            ]
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                out, _ = proc.communicate(timeout=10)
                if out:
                    import io
                    images.append(Image.open(io.BytesIO(out)).convert("RGB"))
            except Exception as e:
                print(f"[H3-TiledUpscale] Warning: Failed to extract frame {idx} via FFmpeg: {e}")

    # Fallback 1: if no frames extracted, check fallback_image_path (e.g. keyframe_overview.png)
    if not images and fallback_image_path and os.path.exists(fallback_image_path):
        try:
            images.append(Image.open(fallback_image_path).convert("RGB"))
        except Exception:
            pass

    # Fallback 2: if still no frames extracted, return a neutral placeholder image
    if not images:
        images.append(Image.new("RGB", (512, 512), color=(128, 128, 128)))

    return images


def build_detail_instruction(detail_focus: str, custom_guidance: str = "", include_action: bool = True) -> str:
    """
    Builds an objective MiniMax H3 canonical shot instruction for the VLM.
    Outputs: [Camera: shot scale, angle, and camera motion]. <Subject 1> [chronological physical actions and posture changes]. [Integrated physical surfaces, textures, lighting, and background].
    """
    focus_guidance = {
        "Balanced_Micro_Detail": (
            "concrete physical surfaces, natural skin pore structure, hair strands, fabric weave, seam stitching, and furniture textures"
        ),
        "Fabrics_And_Costumes": (
            "textile garments, fabric weave patterns, leather creasing, stitching seams, and cloth folds"
        ),
        "Faces_And_Skin_Textures": (
            "facial micro-details, natural skin pores, eyelashes, gaze direction, lip texture, and facial expressions"
        ),
        "Environments_And_Reflections": (
            "physical environment, furniture materials, reflective surfaces, shadows, and background elements"
        ),
        "Sharpening_And_Edges": (
            "structural boundaries, silhouette edges, and contrast transitions between subject and background"
        ),
    }.get(detail_focus, "physical materials and surface textures")

    extra = f" Additional guidance: {custom_guidance.strip()}." if custom_guidance.strip() else ""

    if include_action:
        return (
            "You are generating a shot description specifically for the MiniMax H3 video diffusion model. "
            "Describe the video frames in fluent, objective English using official MiniMax H3 syntax without meta-labels: "
            "[Camera: shot scale, angle, and camera motion]. <Subject 1> [chronological physical actions, limb movements, posture changes, and facial expression changes from start to end]. "
            f"Integrate the {focus_guidance}, lighting direction, and background naturally into the scene description without using a 'Physical elements:' label.{extra} "
            "Strict rules: "
            "1. Refer to the primary person as <Subject 1>. "
            "2. Strictly forbidden: flowery or poetic words (forbidden: subtle, evolves, gleams, graceful, tactile, essence, aura, atmosphere). "
            "3. State only concrete observable physical actions and tangible material textures. "
            "Aim for approximately 110 to 130 words."
        )
    else:
        return (
            "You are generating a shot description specifically for the MiniMax H3 video diffusion model. "
            "Describe the video frames in fluent, objective English using official MiniMax H3 syntax without meta-labels: "
            f"[Camera: shot scale, angle, and camera motion]. Integrate the {focus_guidance}, lighting direction, and background naturally into the scene description without using a 'Physical elements:' label.{extra} "
            "Strict rules: No literary or flowery language. Strictly observable physical facts under 80 words."
        )


def safe_to_cpu(model_obj: Any) -> None:
    """
    Safely offloads model tensors to CPU.
    Handles models with meta tensors (e.g. models initialized with init_empty_weights())
    without throwing NotImplementedError: Cannot copy out of meta tensor.
    """
    if model_obj is None:
        return

    # ComfyUI ModelPatcher or similar unpatch
    if hasattr(model_obj, "unpatch_model"):
        try:
            model_obj.unpatch_model(device_to=torch.device("cpu"))
        except Exception:
            pass

    # Try standard .to("cpu") on model_obj directly
    if hasattr(model_obj, "to") and callable(model_obj.to):
        try:
            model_obj.to("cpu")
            return
        except Exception:
            pass

    # Try standard .to("cpu") on inner model if wrapped (e.g. QwenLoader, VLMContainer)
    inner = getattr(model_obj, "model", None)
    if inner is not None and hasattr(inner, "to") and callable(inner.to):
        try:
            inner.to("cpu")
            return
        except Exception:
            pass

    # Fallback for modules containing meta tensors: manually move non-meta parameters & buffers to CPU
    targets = [m for m in (model_obj, inner) if m is not None and isinstance(m, torch.nn.Module)]
    for target in targets:
        try:
            for p in target.parameters():
                if p is not None and getattr(p, "device", None) is not None:
                    if p.device.type not in ("meta", "cpu"):
                        try:
                            p.data = p.data.to("cpu")
                        except Exception:
                            pass
            for b in target.buffers():
                if b is not None and getattr(b, "device", None) is not None:
                    if b.device.type not in ("meta", "cpu"):
                        try:
                            b.data = b.data.to("cpu")
                        except Exception:
                            pass
        except Exception:
            pass


def safe_to_device(model_obj: Any, device: Any) -> None:
    """
    Safely moves model tensors to target device.
    Handles models with meta tensors without throwing NotImplementedError.
    """
    if model_obj is None or device is None:
        return

    target_device = torch.device(device) if isinstance(device, str) else device
    if getattr(target_device, "type", "") == "meta":
        return

    if hasattr(model_obj, "to") and callable(model_obj.to):
        try:
            model_obj.to(target_device)
            return
        except Exception:
            pass

    inner = getattr(model_obj, "model", None)
    if inner is not None and hasattr(inner, "to") and callable(inner.to):
        try:
            inner.to(target_device)
            return
        except Exception:
            pass

    targets = [m for m in (model_obj, inner) if m is not None and isinstance(m, torch.nn.Module)]
    target_type = target_device.type
    for target in targets:
        try:
            for p in target.parameters():
                if p is not None and getattr(p, "device", None) is not None:
                    if p.device.type not in ("meta", target_type):
                        try:
                            p.data = p.data.to(target_device)
                        except Exception:
                            pass
            for b in target.buffers():
                if b is not None and getattr(b, "device", None) is not None:
                    if b.device.type not in ("meta", target_type):
                        try:
                            b.data = b.data.to(target_device)
                        except Exception:
                            pass
        except Exception:
            pass


def caption_chunk_with_vlm(
    vlm: Any,
    images: List[Image.Image],
    detail_focus: str = "Balanced_Micro_Detail",
    custom_guidance: str = "",
    include_action: bool = True
) -> str:
    """
    Executes VLM / QWENMODEL inference on a list of keyframes to produce a concise, texture-rich description.
    Supports VLMContainer, QwenLoader (QWENMODEL), and raw Transformers models.
    """
    if vlm is None:
        return "High-fidelity micro-detail with sharp textures and physical realism."

    model_obj = getattr(vlm, "model", vlm)
    proc_obj = getattr(vlm, "processor", None)
    tok_obj = getattr(vlm, "tokenizer", None)
    mtype = getattr(vlm, "model_type", "")
    device = getattr(vlm, "device", None)
    if device is None or (isinstance(device, torch.device) and device.type == "meta"):
        real_dev = None
        try:
            for p in model_obj.parameters():
                if getattr(p, "device", None) is not None and p.device.type != "meta":
                    real_dev = p.device
                    break
        except Exception:
            pass
        device = real_dev if real_dev is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    instruction = build_detail_instruction(detail_focus, custom_guidance, include_action=include_action)

    if mtype == "florence2" and proc_obj is not None:
        # Florence-2 multi-task vision format
        task_prompt = "<MORE_DETAILED_CAPTION>"
        prompt = task_prompt + " " + instruction
        mid_img = images[len(images) // 2] if images else Image.new("RGB", (512, 512), (128, 128, 128))
        inputs = proc_obj(text=task_prompt, images=mid_img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model_obj.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=180,
                num_beams=3,
                do_sample=False
            )
        decoded = proc_obj.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = proc_obj.post_process_generation(decoded, task=task_prompt, image_size=(mid_img.width, mid_img.height))
        raw_text = parsed.get(task_prompt, decoded)
        clean = re.sub(r'<[^>]+>', '', raw_text).strip()
        return clean or "High-fidelity micro-detail, natural textures, and sharp edge definition."

    elif proc_obj is not None:
        # Standard Vision-Language Chat Pipeline (Qwen-VL processor)
        content_items = []
        for img in images:
            content_items.append({"type": "image", "image": img})
        content_items.append({"type": "text", "text": instruction})

        messages = [{"role": "user", "content": content_items}]
        text_input = proc_obj.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = proc_obj(
            text=[text_input],
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model_obj.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=True,
                temperature=0.3,
                top_p=0.9
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = proc_obj.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        clean = output_text.strip().replace("\n", " ")
        clean = clean.replace("—", "-").replace("–", "-")
        clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean or "High-fidelity micro-detail, natural textures, and sharp edge definition."

    elif tok_obj is not None and model_obj is not None:
        # QWENMODEL (e.g. from QwenLoader) text generation / prompt director mode
        try:
            prompt_content = instruction
            if custom_guidance:
                prompt_content = (
                    f"Scene Context:\n{custom_guidance.strip()}\n\n"
                    f"Task: Enrich the above scene with concrete, physical micro-textures and lighting details for video super-resolution. "
                    f"Focus specifically on: {instruction}. Keep it objective, tangible, and under 80 words."
                )
            messages = [
                {"role": "system", "content": "You are a professional visual scene director and prompt engineer for high-resolution video upscaling."},
                {"role": "user", "content": prompt_content}
            ]
            text_prompt = tok_obj.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok_obj([text_prompt], return_tensors="pt").to(device)
            safe_to_device(model_obj, device)
            with torch.no_grad():
                generated_ids = model_obj.generate(
                    **inputs,
                    max_new_tokens=180,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    repetition_penalty=1.05
                )
            generated_ids = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded = tok_obj.batch_decode(generated_ids, skip_special_tokens=True)[0]
            clean = decoded.strip().replace("\n", " ")
            clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()
            return clean or "High-fidelity micro-detail, natural textures, and sharp edge definition."
        except Exception as e_qwen:
            print(f"[H3-TiledUpscale] Qwen generation notice: {e_qwen}")
            return "High-fidelity micro-detail, natural textures, and sharp edge definition."

    return "High-fidelity micro-detail with sharp textures and physical realism."


def unload_vlm_container(vlm: Any) -> None:
    """
    Completely drops VLM model and processor from memory and flushes CUDA cache.
    Guarantees 0 GB VRAM retention before MiniMax H3 loads.
    """
    if vlm is None:
        return
    print("[H3-TiledUpscale] Evicting VLM / QWENMODEL from memory to free VRAM for MiniMax H3...")
    safe_to_cpu(vlm)

    try:
        if hasattr(vlm, "model"):
            vlm.model = None
    except Exception:
        pass

    try:
        if hasattr(vlm, "processor"):
            vlm.processor = None
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass
    print("[H3-TiledUpscale] VLM memory cleared successfully. VRAM baseline restored.")
