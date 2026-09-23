import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import folder_paths
from .schemas import H3Context

def get_h3_prompt_output_dir() -> Path:
    """Returns the output directory for saving prompt files."""
    try:
        base_output = folder_paths.get_output_directory()
        out_dir = Path(base_output) / "h3_chunk_prompts"
    except Exception:
        out_dir = Path(os.getcwd()) / "output" / "h3_chunk_prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def format_human_readable_prompts(
    session_id: str,
    master_prompt: str,
    chunk_prompts: Dict[int, Dict[str, Any]],
    fps: float
) -> str:
    """Formats chunk prompts and master prompt into a clean human-readable text document."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 80,
        "MINIMAX H3 TILED UPSCALE - CHUNK PROMPTS MANIFEST",
        f"Session ID : {session_id}",
        f"Timestamp  : {timestamp}",
        f"FPS        : {fps:.2f}",
        f"Total Chunks: {len(chunk_prompts)}",
        "=" * 80,
        "",
        "-" * 80,
        "[SECTION 1: COMBINED MASTER MINIMAX H3 PROMPT]",
        "-" * 80,
        master_prompt.strip(),
        "",
        "-" * 80,
        "[SECTION 2: INDIVIDUAL TEMPORAL CHUNK BREAKDOWNS]",
        "-" * 80,
    ]

    for k in sorted(chunk_prompts.keys()):
        c = chunk_prompts[k]
        s_f = c.get("start_frame", 0)
        e_f = c.get("end_frame", 0)
        s_sec = s_f / fps if fps > 0 else 0.0
        e_sec = e_f / fps if fps > 0 else 0.0
        desc = c.get("description", "").strip()

        lines.extend([
            f">> Chunk {k}: Frames {s_f} to {e_f} ({s_sec:.2f}s - {e_sec:.2f}s)",
            f"   Visual Description:",
            f"   {desc}",
            ""
        ])

    lines.append("=" * 80)
    return "\n".join(lines)

def save_chunk_prompts(
    h3_context: H3Context,
    master_prompt: str,
    chunk_prompts: Dict[int, Dict[str, Any]],
    save_to_output: bool = True
) -> str:
    """
    Saves prompts to disk.
    If save_to_output is True: writes to output/h3_chunk_prompts/h3_prompts_<session_id>.txt and .json
    Always saves to temp_dir if available.
    Returns path of the saved text file, or empty string if not saved to output.
    """
    session_id = h3_context.session_id
    fps = h3_context.fps

    text_content = format_human_readable_prompts(session_id, master_prompt, chunk_prompts, fps)
    
    json_data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "fps": fps,
        "total_chunks": len(chunk_prompts),
        "master_prompt": master_prompt,
        "chunks": [
            {
                "chunk_idx": k,
                "start_frame": chunk_prompts[k].get("start_frame", 0),
                "end_frame": chunk_prompts[k].get("end_frame", 0),
                "start_sec": (chunk_prompts[k].get("start_frame", 0) / fps) if fps > 0 else 0.0,
                "end_sec": (chunk_prompts[k].get("end_frame", 0) / fps) if fps > 0 else 0.0,
                "description": chunk_prompts[k].get("description", "")
            }
            for k in sorted(chunk_prompts.keys())
        ]
    }

    # 1. Always save to temp_dir if available for current session cache
    if h3_context.temp_dir and h3_context.temp_dir.exists():
        temp_txt = h3_context.temp_dir / "chunk_prompts.txt"
        temp_json = h3_context.temp_dir / "chunk_prompts.json"
        try:
            with open(temp_txt, "w", encoding="utf-8") as f:
                f.write(text_content)
            with open(temp_json, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[H3-TiledUpscale] Warning: Could not write prompt cache to temp_dir: {e}")

    # 2. Save to ComfyUI output directory if toggled
    saved_path_str = ""
    if save_to_output:
        out_dir = get_h3_prompt_output_dir()
        out_txt = out_dir / f"h3_prompts_{session_id}.txt"
        out_json = out_dir / f"h3_prompts_{session_id}.json"
        try:
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(text_content)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            saved_path_str = str(out_txt)
            print(f"[H3-TiledUpscale] Saved chunk prompts for inspection: {out_txt}")
        except Exception as e:
            print(f"[H3-TiledUpscale] Warning: Failed to save chunk prompts to output: {e}")

    return saved_path_str
