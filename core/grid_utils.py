import math
from typing import List, Tuple
from .schemas import TemporalChunkDescriptor

def parse_aspect_ratio(ratio_str: str, src_w: int, src_h: int, custom_ratio: str = "16:10") -> float:
    """Parses aspect ratio string to float width/height."""
    ratio_str = ratio_str.strip()
    if ratio_str == "Match_Input":
        return float(src_w) / float(src_h)
    elif "16:9" in ratio_str:
        return 16.0 / 9.0
    elif "9:16" in ratio_str:
        return 9.0 / 16.0
    elif "2.39:1" in ratio_str:
        return 2.39
    elif "4:3" in ratio_str:
        return 4.0 / 3.0
    elif "1:1" in ratio_str:
        return 1.0
    elif "21:9" in ratio_str:
        return 21.0 / 9.0
    elif ratio_str == "Custom":
        try:
            if ":" in custom_ratio:
                num, den = custom_ratio.split(":")
                return float(num) / float(den)
            elif "/" in custom_ratio:
                num, den = custom_ratio.split("/")
                return float(num) / float(den)
            else:
                return float(custom_ratio)
        except Exception:
            return 16.0 / 10.0
    return float(src_w) / float(src_h)

def calculate_canvas_dimensions(megapixels: float, aspect_ratio: float) -> Tuple[int, int]:
    """Calculates width and height rounded to multiples of 32 (MiniMax DiT requires 2x2 latent patch = 32px)."""
    if megapixels <= 0:
        return 0, 0
    h_raw = math.sqrt((megapixels * 1e6) / aspect_ratio)
    w_raw = aspect_ratio * h_raw
    h = int(round(h_raw / 32.0)) * 32
    w = int(round(w_raw / 32.0)) * 32
    return max(32, w), max(32, h)

def compute_tile_intervals(canvas_dim: int, tile_dim: int, overlap_pct: float) -> List[Tuple[int, int]]:
    """Calculates coordinate slices (start, end) guaranteeing edge-to-edge coverage with uniform overlap."""
    canvas_dim = max(32, int(round(canvas_dim / 32.0)) * 32)
    tile_dim = max(32, int(round(tile_dim / 32.0)) * 32)
    if canvas_dim <= tile_dim or tile_dim <= 0:
        return [(0, canvas_dim)]
    
    step = int(tile_dim * (1.0 - overlap_pct))
    if step <= 0:
        return [(0, canvas_dim)]

    num_tiles = max(1, int(math.ceil((canvas_dim - tile_dim) / max(1, step)))) + 1
    if num_tiles <= 1:
        return [(0, canvas_dim)]
    
    # Recalculate actual step to distribute overlap uniformly
    actual_step = (canvas_dim - tile_dim) / (num_tiles - 1)
    
    intervals = []
    for i in range(num_tiles):
        start = int(round(i * actual_step / 32.0)) * 32
        end = start + tile_dim
        # Guard boundary
        if end > canvas_dim:
            end = canvas_dim
            start = max(0, canvas_dim - tile_dim)
        intervals.append((start, end))
    return intervals

def calculate_optimal_tile_dimensions(
    canvas_w: int,
    canvas_h: int,
    tile_megapixels: float,
    overlap_pct: float,
    tight_tile_overlap: bool = False,
    min_megapixels: float = 0.25
) -> Tuple[int, int, int, int]:
    """
    Calculates 2D tile dimensions (width, height) and grid counts (num_tiles_w, num_tiles_h).
    When tight_tile_overlap is True, minimizes dimensions to satisfy overlap_pct without
    ballooning redundant overlap, strictly locking the tile to the canvas aspect ratio
    and respecting a semantic safety floor.
    """
    if tile_megapixels <= 0:
        return canvas_w, canvas_h, 1, 1

    aspect_ratio = float(canvas_w) / float(max(1, canvas_h))
    h_raw = math.sqrt((tile_megapixels * 1e6) / aspect_ratio)
    w_raw = aspect_ratio * h_raw
    tile_w = max(32, int(round(w_raw / 32.0)) * 32)
    tile_h = max(32, int(round(h_raw / 32.0)) * 32)

    if tile_w >= canvas_w and tile_h >= canvas_h:
        return canvas_w, canvas_h, 1, 1

    step_w = max(1, int(tile_w * (1.0 - overlap_pct)))
    step_h = max(1, int(tile_h * (1.0 - overlap_pct)))

    def compute_tiles_count(canvas_dim, tile_dim, step_dim):
        frac = (canvas_dim - tile_dim) / max(1, step_dim)
        return max(1, int(math.floor(frac) if (frac - math.floor(frac)) < 0.06 else math.ceil(frac))) + 1

    num_tiles_w = compute_tiles_count(canvas_w, tile_w, step_w)
    num_tiles_h = compute_tiles_count(canvas_h, tile_h, step_h)

    if tight_tile_overlap and (num_tiles_w > 1 or num_tiles_h > 1):
        min_h_w = (canvas_w / aspect_ratio) / (1.0 + (num_tiles_w - 1) * (1.0 - overlap_pct)) if num_tiles_w > 1 else canvas_h
        min_h_h = canvas_h / (1.0 + (num_tiles_h - 1) * (1.0 - overlap_pct)) if num_tiles_h > 1 else canvas_h
        tight_h = max(min_h_w, min_h_h)

        # Enforce semantic receptive field safety floor
        floor_h = math.sqrt((min_megapixels * 1e6) / aspect_ratio)
        tight_h = max(tight_h, floor_h)
        tight_w = tight_h * aspect_ratio

        tile_h = int(math.ceil(tight_h / 32.0)) * 32
        tile_w = int(math.ceil(tight_w / 32.0)) * 32

    tile_w = min(canvas_w, max(32, tile_w))
    tile_h = min(canvas_h, max(32, tile_h))
    return tile_w, tile_h, num_tiles_w, num_tiles_h

def minimax_latents_to_frames(t_lat: int) -> int:
    """Converts MiniMax H3 latent time tokens to video frame count.
    Exact formula: frames = (t_lat - 2) // 5 * 17 + 5 (for t_lat > 1).
    """
    if t_lat <= 1:
        return 1
    return max(1, (t_lat - 2) // 5 * 17 + 5)

def minimax_frames_to_latents(frames: int) -> int:
    """Converts video frame count to MiniMax H3 latent time tokens.
    Exact formula: latents = (frames - 5) // 17 * 5 + 2 (for frames > 1).
    """
    if frames <= 1:
        return 1
    return max(1, (frames - 5) // 17 * 5 + 2)

def compute_temporal_chunks(
    total_frames: int,
    chunk_frames: int = 124,
    prefix_frames: int = 12,
    blend_frames: int = 8
) -> List[TemporalChunkDescriptor]:
    """Partitions video into temporal chunks with momentum prefix and overlapping seam blend intervals.
    Enforces chunk boundaries to align with MiniMax H3's 17-frame (5 latent token) clip period
    to guarantee phase alignment and eliminate inter-chunk color flashing.
    """
    chunks = []
    if chunk_frames <= 0 or total_frames <= chunk_frames:
        chunks.append(TemporalChunkDescriptor(
            chunk_idx=0,
            start_frame=0,
            end_frame=total_frames,
            prefix_start_frame=0,
            prefix_frame_count=0,
            blend_frames=0
        ))
        return chunks

    # Align chunk_frames and blend_frames to multiples of 17 (MiniMax H3 clip period)
    # 1 clip = 17 frames = 5 latent tokens
    c_frames = max(17, int(round(chunk_frames / 17.0)) * 17)
    
    # Enforce blend_frames to be at least 17 frames (1 clip = 5 tokens) when chunking
    if blend_frames > 0:
        b_frames = max(17, int(round(blend_frames / 17.0)) * 17)
    else:
        b_frames = 17
    
    if b_frames >= c_frames:
        b_frames = max(17, c_frames - 17)
    
    step = max(17, c_frames - b_frames)
    p_frames = max(0, int(round(prefix_frames / 17.0)) * 17) if prefix_frames > 0 else 0

    curr_start = 0
    k = 0
    while curr_start < total_frames:
        curr_end = min(curr_start + c_frames, total_frames)
        
        # If remaining frames at the end are fewer than 17 or too small,
        # merge them into the current chunk to avoid degenerate tiny final chunk
        if (total_frames - curr_end) < 17:
            curr_end = total_frames

        if k == 0:
            p_start = 0
            p_count = 0
            blend_f = 0
        else:
            p_start = max(0, curr_start - p_frames)
            p_count = curr_start - p_start
            blend_f = b_frames

        chunks.append(TemporalChunkDescriptor(
            chunk_idx=k,
            start_frame=curr_start,
            end_frame=curr_end,
            prefix_start_frame=p_start,
            prefix_frame_count=p_count,
            blend_frames=blend_f
        ))

        k += 1
        if curr_end >= total_frames:
            break
        curr_start += step

    return chunks

