import re
from typing import Dict, Any, List

def parse_h3_prompt(text: str) -> Dict[str, Any]:
    """
    Parses an incoming MiniMax H3 prompt into its standard 6 sections,
    tolerating formatting variations, missing headers, or preamble tags.
    """
    text = text.strip()
    sections = {
        "subject_definitions": "",
        "summary": "",
        "retention_analysis": "",
        "detailed_description": "",
        "overall_soundscape": "",
        "non_diegetic_music": ""
    }
    
    header_pattern = re.compile(
        r'^(subject_definitions|summary|retention_analysis|detailed_description|integrated_multimodal_description|overall_soundscape|non_diegetic_music|non_diagetic_music)\s*:\s*',
        re.IGNORECASE | re.MULTILINE
    )
    
    matches = list(header_pattern.finditer(text))
    is_structured = bool(matches) or ("<Subject" in text or "<Picture" in text or "<Video" in text)
    
    if not is_structured:
        return {"is_structured": False, "raw_text": text}

    if matches:
        for i, m in enumerate(matches):
            header = m.group(1).lower()
            if header == "integrated_multimodal_description":
                header = "detailed_description"
            elif header == "non_diagetic_music":
                header = "non_diegetic_music"
                
            start_pos = m.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections[header] = text[start_pos:end_pos].strip()
            
        preamble = text[:matches[0].start()].strip()
        if preamble:
            if not sections["subject_definitions"]:
                sections["subject_definitions"] = preamble
            else:
                sections["subject_definitions"] = (preamble + "\n" + sections["subject_definitions"]).strip()
    else:
        sections["subject_definitions"] = text
        
    return {"is_structured": True, "sections": sections, "raw_text": text}


def format_cut_timestamp(seconds: float) -> str:
    """Formats seconds into canonical MM:SS.mmm notation."""
    total_ms = max(0, int(round(seconds * 1000)))
    minutes = total_ms // 60000
    remainder_ms = total_ms % 60000
    secs = remainder_ms // 1000
    ms = remainder_ms % 1000
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def strip_shot_headers(text: str) -> str:
    """
    Strips raw shot headers such as:
    - '[0s-4.5s]Shot 1:'
    - '[Shot 1]:' or '[Shot 1]'
    - '[Shot 1] (0.0s - 5.2s):'
    - 'Shot 1:'
    - '[Shot 2] At 00:04.500, the shot transitions to:'
    - '[Shot 2] At 00:04.500:'
    - leading '[0.0s - 5.0s]' or '(0.0s - 5.0s)'
    from a text segment so it can be seamlessly embedded into a shot description
    without creating duplicate or nested shot headers.
    """
    if not text:
        return ""
    cleaned = re.sub(
        r'^\s*(?:\[\s*\d+(?:\.\d+)?\s*s\s*-\s*\d+(?:\.\d+)?\s*s\s*\])?\s*(?:\[\s*Shot\s*\d+\s*\]|Shot\s*\d+)\s*(?:\(?\s*[\d\.]+\s*s\s*-\s*[\d\.]+\s*s\s*\)?)?\s*(?:At\s*[\d\:\.]+,?\s*(?:the\s*(?:camera|shot)\s*(?:cuts|transitions|changes|switches)\s*(?:to|with),?\s*)?)?:?\s*',
        '',
        text,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r'^\s*[\[\(]\s*\d+(?:\.\d+)?\s*s\s*-\s*\d+(?:\.\d+)?\s*s\s*[\]\)]:?\s*',
        '',
        cleaned,
        flags=re.IGNORECASE
    )
    return cleaned.strip()


def parse_shot_ranges(orig_details: str) -> List[Any]:
    """
    Splits detailed_description into distinct shot blocks, extracting the time range
    (start_sec, end_sec, raw_block) for each shot.
    Supports [0s-4.5s] notation, [Shot N] At MM:SS.mmm notation, and sequential shots.
    """
    if not orig_details.strip():
        return []
    shot_blocks = re.split(r'(?=(?:^|\n)\s*(?:\[\s*(?:\d+(?:\.\d+)?\s*s\s*-\s*\d+(?:\.\d+)?\s*s|\w+\s*\d+)\s*\]|Shot\s*\d+))', orig_details.strip())
    shot_blocks = [b.strip() for b in shot_blocks if b.strip()]
    
    parsed = []
    for b in shot_blocks:
        m_range = re.search(r'\[\s*([\d\.]+)\s*s\s*-\s*([\d\.]+)\s*s\s*\]', b)
        if m_range:
            s_t, e_t = float(m_range.group(1)), float(m_range.group(2))
            parsed.append((s_t, e_t, b))
            continue
            
        m_cut = re.search(r'(?:At\s+|\[\s*)(\d{1,2}):(\d{2})(?:\.(\d+))?', b, re.IGNORECASE)
        if m_cut:
            mins = int(m_cut.group(1))
            secs = int(m_cut.group(2))
            ms_str = m_cut.group(3) or "0"
            s_t = mins * 60.0 + secs + float(f"0.{ms_str}")
            parsed.append((s_t, None, b))
            continue
            
        parsed.append((0.0, None, b))
        
    for i in range(len(parsed)):
        s_t, e_t, b = parsed[i]
        if e_t is None:
            if i + 1 < len(parsed):
                e_t = parsed[i+1][0]
            else:
                e_t = float('inf')
            parsed[i] = (s_t, e_t, b)
            
    return parsed


def match_shots_to_chunk(parsed_shots: List[Any], start_sec: float, end_sec: float) -> List[Any]:
    """
    Finds shots from parsed_shots that significantly overlap with [start_sec, end_sec].
    A shot matches if overlap >= 1.0s, or >= 30% of chunk duration, or >= 30% of shot duration.
    """
    duration = max(0.1, end_sec - start_sec)
    matched = []
    for s_t, e_t, raw_text in parsed_shots:
        overlap = max(0.0, min(end_sec, e_t) - max(start_sec, s_t))
        shot_dur = max(0.1, e_t - s_t)
        if overlap >= 1.0 or (overlap / duration >= 0.3) or (overlap / shot_dur >= 0.3):
            matched.append((s_t, e_t, raw_text, overlap))
    return matched


def extract_chunk_summary(narrative_text: str, clean_vlm: str, has_video_ref: bool = False) -> str:
    """
    Extracts a concise, natural 1-sentence summary describing what actually occurs
    in this chunk's timeframe, avoiding boilerplate meta-language.
    """
    if narrative_text:
        m_tag = re.match(r'^\[(.*?)\]\.?\s*(.*)', narrative_text, re.DOTALL)
        if m_tag:
            tag_content = m_tag.group(1).strip()
            rest = m_tag.group(2).strip()
            first_sent = re.split(r'(?<=[.!?])\s+', rest)[0].strip().rstrip('.') if rest else ""
            if first_sent:
                return f"In a {tag_content}, {first_sent}."
            else:
                return f"A {tag_content} sequence matching <Video 1>." if has_video_ref else f"A {tag_content} sequence with crisp micro-detail."
        else:
            first_sent = re.split(r'(?<=[.!?])\s+', narrative_text.strip())[0].strip().rstrip('.')
            return f"{first_sent}."
    elif clean_vlm:
        clean_text = clean_vlm[0].lower() + clean_vlm[1:] if not clean_vlm.startswith('<') else clean_vlm
        return f"The sequence portrays {clean_text}."
    else:
        return "The sequence adheres to the motion, timing, and composition in <Video 1>." if has_video_ref else "The sequence maintains continuous motion and crisp high-resolution detail."


def build_tile_ref2va_prompt(
    tile_idx: int,
    row: int,
    col: int,
    total_rows: int,
    total_cols: int,
    user_prompt: str,
    duration_sec: float = 5.0,
    start_sec: float = 0.0,
    has_prefix_video: bool = False,
    chunk_vlm_description: str = "",
    has_video_ref: bool = False
) -> str:
    """
    Builds the standardized 6-section MiniMax H3 Ref2VA prompt for a spatial quadrant.
    Automatically parses and refactors original structured MiniMax H3 prompts (preserving
    subjects, scene semantics, and audio definitions) to work seamlessly with spatial tiling.
    Fuses chunk-specific VLM micro-texture descriptions when provided.
    Preserves seamless cross-tile semantic harmony by avoiding artificial framing tags.
    """
    quadrant_desc = "the entire frame"
    
    prefix_def = ""
    prefix_retention = ""
    prefix_desc = ""
    
    if has_prefix_video:
        prefix_def = "\n<Video 2> is the preceding 0.5-second sequence (-0.5s - 0.0s) establishing continuous momentum, camera trajectory, and lighting."
        prefix_retention = "\n- Seamlessly connect camera momentum, head/body posture, and lighting from <Video 2>."
        prefix_desc = "Continuing seamlessly from the camera momentum and subject posture of <Video 2>, "

    # Tiled latent super-resolution maintains global semantic coherence across all tiles
    # by using uniform scene framing rather than contradictory quadrant tags.
    framing_prefix = ""

    parsed = parse_h3_prompt(user_prompt)
    chunk_end = start_sec + duration_sec
    sections = parsed.get("sections", {}) if parsed.get("is_structured") else {}

    # Extract scene narrative strictly within this chunk's timeframe
    narrative_text = ""
    orig_details = sections.get("detailed_description", "").strip()
    if orig_details:
        parsed_shots = parse_shot_ranges(orig_details)
        if len(parsed_shots) > 1:
            matched = match_shots_to_chunk(parsed_shots, start_sec, chunk_end)
            if matched:
                shot_parts = []
                for idx, (s_t, e_t, b_text, ov) in enumerate(matched):
                    stripped = strip_shot_headers(b_text)
                    if idx == 0:
                        shot_parts.append(stripped)
                    else:
                        cut_rel = max(0.0, s_t - start_sec)
                        shot_parts.append(f"At {format_cut_timestamp(cut_rel)}, the shot transitions to {stripped}")
                narrative_text = " ".join(shot_parts)
        else:
            if start_sec < 5.0 or not chunk_vlm_description:
                narrative_text = strip_shot_headers(orig_details)
    elif not parsed.get("is_structured") and (start_sec < 5.0 or not chunk_vlm_description):
        narrative_text = strip_shot_headers(user_prompt)

    clean_vlm = chunk_vlm_description.strip().rstrip(".")

    # Assemble detailed description focusing purely on what is happening
    if narrative_text and clean_vlm:
        scene_body = f"{narrative_text.rstrip('.')}. Observable visual details feature {clean_vlm}."
    elif narrative_text:
        scene_body = f"{narrative_text.rstrip('.')}."
    elif clean_vlm:
        clean_lead = clean_vlm[0].upper() + clean_vlm[1:] if not clean_vlm.startswith('<') else clean_vlm
        scene_body = f"{clean_lead}."
    else:
        scene_body = "The scene maintains continuous movement and visual micro-detail aligned with <Video 1>." if has_video_ref else "The scene maintains continuous movement, crisp micro-textures, and high dynamic detail."

    if scene_body:
        scene_body = scene_body[0].upper() + scene_body[1:]

    tile_detailed = f"[Shot 1] {prefix_desc}{framing_prefix}{scene_body}".strip()

    # 1. Adapt subject_definitions
    orig_subjs = sections.get("subject_definitions", "").strip()
    cleaned_subjs = []
    has_subject_1 = False
    has_picture_1 = False

    if orig_subjs:
        for line in orig_subjs.splitlines():
            line_s = line.strip()
            if not line_s or line_s.startswith("<Video 1>") or line_s.startswith("<Video 2>"):
                continue
            if "<Subject 1>" in line_s: has_subject_1 = True
            if "<Picture 1>" in line_s: has_picture_1 = True
            cleaned_subjs.append(line_s)

    if not has_picture_1:
        cleaned_subjs.insert(0, "<Picture 1> is the reference frame establishing global spatial position, environment, and appearance.")
    if not has_subject_1:
        clean_name = re.sub(r'[\r\n]+', ' ', user_prompt[:80]).strip() or "primary subject"
        cleaned_subjs.insert(0, f"<Subject 1> is the {clean_name}.")

    if has_video_ref:
        cleaned_subjs.append(f"<Video 1> is the motion, camera trajectory, and visual composition for {quadrant_desc}.{prefix_def}")
    elif prefix_def:
        cleaned_subjs.append(prefix_def.strip())

    tile_subject_defs = "\n".join(cleaned_subjs)

    # 2. Adapt summary: describe what actually happens during this chunk's timeframe
    task_prefix = "[video editing + reference generation + audio reuse]"
    chunk_summary = extract_chunk_summary(narrative_text, clean_vlm, has_video_ref=has_video_ref)
    tile_summary = f"{task_prefix} {chunk_summary}"

    # 3. Adapt retention_analysis: focus on preserving camera, subject, and physical details
    if has_video_ref:
        tile_retention = f"""- Fully preserve camera velocity, trajectories, object silhouettes, and lighting from <Video 1>.{prefix_retention}
- Preserve character appearance, facial identity, and clothing from <Picture 1> and <Video 1>.
- Faithfully render visible textures, material reflections, and environmental details matching <Video 1>.""".strip()
    else:
        tile_retention = f"""- Preserve character appearance, facial identity, and clothing from <Picture 1>.{prefix_retention}
- Faithfully render visible textures, material reflections, and crisp surface micro-details.
- Maintain smooth temporal motion and consistent illumination across all frames.""".strip()

    # 5. Overall Soundscape & Music
    tile_soundscape = sections.get("overall_soundscape", "").strip() or ("Synchronized ambient sound matching the motion in <Video 1>." if has_video_ref else "Synchronized ambient sound matching the scene motion.")
    tile_music = sections.get("non_diegetic_music", "").strip() or "None."

    return f"""subject_definitions:
{tile_subject_defs}
 
summary:
{tile_summary}
 
retention_analysis:
{tile_retention}
 
detailed_description:
{tile_detailed}
 
overall_soundscape:
{tile_soundscape}
 
non_diegetic_music:
{tile_music}""".strip()


def build_face_refine_prompt(
    character_name: str = "Character",
    duration_sec: float = 5.0,
    has_prefix_video: bool = False,
    has_portrait: bool = False
) -> str:
    """
    Builds the standardized 6-section MiniMax H3 Ref2VA prompt for face enhancement,
    optionally conditioning on a character portrait reference (<Picture 1>).
    """
    prefix_def = ""
    prefix_retention = ""
    prefix_desc = ""
    if has_prefix_video:
        prefix_def = "\n<Video 2> is the preceding 0.5-second sequence of the face maintaining continuous expression and eye gaze."
        prefix_retention = "\n- Seamlessly connect head tilt, mouth posture, and gaze from <Video 2>."
        prefix_desc = " Continuing seamlessly from <Video 2>,"

    portrait_def = "<Picture 1> is the high-resolution character portrait establishing exact facial identity, skin texture, and bone structure." if has_portrait else "<Picture 1> is the anchor frame of the subject."

    prompt = f"""subject_definitions:
<Subject 1> is the face and head features of {character_name}.
{portrait_def}
<Video 1> is the animated motion, lip sync, and micro-expressions of the facial crop.{prefix_def}

summary:
[video editing + reference generation] High-fidelity facial enhancement preserving identity, skin pore detail, eye highlights, and exact expression dynamics.

retention_analysis:
- Lock facial bone structure, head shape, and eye gaze perfectly to <Video 1>.{prefix_retention}
- Synthesize realistic skin pores, subtle highlights in irises, and sharp eyelashes without altering facial expression.
- Match lighting and color tones precisely to <Video 1>.

detailed_description:
[Shot 1]{prefix_desc} Close-up facial detail refinement. Preserve exact timing of blinking, talking, and micro-expressions from <Video 1>. Apply sharp photorealistic skin texture, natural peach fuzz, and clear pupil reflections aligned with <Picture 1>.

overall_soundscape:
Synchronized speech, breathing, and subtle rustles matching <Video 1>.

non_diegetic_music:
None.
"""
    return prompt.strip()


def inject_reference_bundle(conditioning, reference_bundle, vae=None, width=None, height=None):
    """
    Attaches reference pictures, videos, and audios from H3ReferenceAssetBundle
    into MiniMax H3's native conditioning dictionary format:
    - 'minimax_ref_items': List of {'type': 'image'|'video'|'audio', 'data': tensor, 'index': int}
    - 'minimax_ref_blocks': List of {'kind': 'image'|'video'|'audio', 'latent': z_tensor, 'latent_h': int, 'latent_w': int}
    """
    if conditioning is None or reference_bundle is None:
        return conditioning

    import torch
    import math

    # Normalize conditioning into standard ComfyUI format: list of [emb, dict]
    if isinstance(conditioning, torch.Tensor):
        conditioning = [[conditioning, {}]]
    elif isinstance(conditioning, (list, tuple)):
        normalized = []
        for item in conditioning:
            if isinstance(item, (list, tuple)):
                if len(item) == 2 and isinstance(item[1], dict):
                    normalized.append([item[0], dict(item[1])])
                elif len(item) == 1:
                    normalized.append([item[0], {}])
                else:
                    normalized.append([item[0], dict(item[1]) if len(item) > 1 and isinstance(item[1], dict) else {}])
            elif isinstance(item, torch.Tensor):
                normalized.append([item, {}])
            else:
                normalized.append([item, {}])
        conditioning = normalized
    else:
        return conditioning

    if len(conditioning) == 0:
        conditioning = [[torch.zeros((1, 1, 2048), dtype=torch.float32), {}]]

    CANVAS_MULTIPLE = 32
    REF_IMAGE_SHORT_EDGE = 2048

    ref_image_size = "match"
    if isinstance(reference_bundle, dict):
        ref_image_size = reference_bundle.get("ref_image_size", "match")
    elif hasattr(reference_bundle, "ref_image_size"):
        ref_image_size = getattr(reference_bundle, "ref_image_size", "match")

    target_w = width if width is not None else 1344
    target_h = height if height is not None else 768
    target_area = target_w * target_h

    out_cond = []
    for text_emb, cond_dict in conditioning:
        new_dict = dict(cond_dict)
        ref_items = list(new_dict.get("minimax_ref_items", []))
        ref_blocks = list(new_dict.get("minimax_ref_blocks", []))

        # 1. Process Reference Pictures (<Picture 1..9>)
        for idx, pic_tensor in enumerate(reference_bundle.get("pictures", []), start=1):
            h, w = pic_tensor.shape[1], pic_tensor.shape[2]
            if ref_image_size == "match":
                # aspect-preserving scale (down only) to the generation's pixel area
                scale = min(1.0, math.sqrt(target_area / (w * h)))
            else:
                # 2048 short edge for best identity fidelity
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))

            tw = max(CANVAS_MULTIPLE, int(round(w * scale / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, int(round(h * scale / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)

            if th != h or tw != w:
                resized_pic = torch.nn.functional.interpolate(
                    pic_tensor[..., :3].movedim(-1, 1), size=(th, tw), mode="bilinear", align_corners=False
                ).movedim(1, -1)
            else:
                resized_pic = pic_tensor[..., :3]

            ref_items.append({"type": "image", "data": resized_pic, "index": idx})
            if vae is not None:
                z_pic = vae.encode(resized_pic.to(vae.device))
                ref_blocks.append({
                    "kind": "image",
                    "latent": z_pic,
                    "latent_h": th // 16,
                    "latent_w": tw // 16,
                    "index": idx
                })

        # 2. Process Reference Videos (<Video 1..3>)
        for idx, vid_tensor in enumerate(reference_bundle.get("videos", []), start=1):
            ref_items.append({"type": "video", "data": vid_tensor, "index": idx})
            if vae is not None:
                z_vid = vae.encode(vid_tensor.to(vae.device))
                ref_blocks.append({
                    "kind": "video",
                    "latent": z_vid,
                    "latent_t": z_vid.shape[2] if z_vid.ndim == 5 else 1,
                    "latent_h": z_vid.shape[-2],
                    "latent_w": z_vid.shape[-1],
                    "ref_audio_t": 0,
                    "index": idx
                })

        new_dict["minimax_ref_items"] = ref_items
        new_dict["minimax_refs"] = ref_blocks
        new_dict["minimax_ref_blocks"] = ref_blocks
        out_cond.append([text_emb, new_dict])

    return out_cond


def inject_keyframe_anchor(conditioning, z_prop):
    """
    Injects the decoded and re-encoded trailing anchor frame from Chunk k-1
    into Chunk k's conditioning as <Picture N+1> to lock micro-textures and lighting.
    """
    if conditioning is None or z_prop is None:
        return conditioning

    import torch

    if isinstance(conditioning, torch.Tensor):
        conditioning = [[conditioning, {}]]
    elif isinstance(conditioning, (list, tuple)):
        normalized = []
        for item in conditioning:
            if isinstance(item, (list, tuple)):
                if len(item) == 2 and isinstance(item[1], dict):
                    normalized.append([item[0], dict(item[1])])
                elif len(item) == 1:
                    normalized.append([item[0], {}])
                else:
                    normalized.append([item[0], dict(item[1]) if len(item) > 1 and isinstance(item[1], dict) else {}])
            elif isinstance(item, torch.Tensor):
                normalized.append([item, {}])
            else:
                normalized.append([item, {}])
        conditioning = normalized
    else:
        return conditioning

    if len(conditioning) == 0:
        return conditioning

    out_cond = []
    for text_emb, cond_dict in conditioning:
        new_dict = dict(cond_dict)
        ref_blocks = list(new_dict.get("minimax_ref_blocks", []))
        cur_pic_count = sum(1 for b in ref_blocks if b.get("kind") == "image")
        next_pic_idx = cur_pic_count + 1

        ref_blocks.append({
            "kind": "image",
            "latent": z_prop,
            "latent_h": z_prop.shape[-2],
            "latent_w": z_prop.shape[-1],
            "index": next_pic_idx,
            "is_keyframe_anchor": True
        })

        new_dict["minimax_refs"] = ref_blocks
        new_dict["minimax_ref_blocks"] = ref_blocks
        out_cond.append([text_emb, new_dict])
    return out_cond

