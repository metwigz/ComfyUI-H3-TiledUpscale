from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class H3ReferenceBundle:
    pictures: List[Any] = field(default_factory=list)  # max 9, each torch.Tensor [B, H, W, C]
    videos: List[Tuple[Any, Optional[Any]]] = field(default_factory=list)  # max 3, each (video_frames, audio_soundtrack)
    audios: List[Any] = field(default_factory=list)  # max 3

    def total_references(self) -> int:
        return len(self.pictures) + len(self.videos) + len(self.audios)

@dataclass
class TemporalChunkDescriptor:
    chunk_idx: int
    start_frame: int
    end_frame: int
    prefix_start_frame: int = 0
    prefix_frame_count: int = 0
    blend_frames: int = 0
    prefix_video_path: Optional[Path] = None


def slice_audio_latent_for_video_chunk(
    audio_latent: Any,
    video_start_token: int,
    video_end_token: int,
    video_total_tokens: int
) -> Any:
    """
    Slices a 4D audio latent tensor [1, 32, 2, T_audio] proportionally to match
    an active temporal video chunk [video_start_token : video_end_token].
    Guarantees zero audio pitch-shifting, drift, or temporal desync.
    """
    if audio_latent is None:
        return None
    t_audio_total = audio_latent.shape[-1]
    ratio = t_audio_total / max(1, video_total_tokens)
    a_start = int(round(video_start_token * ratio))
    a_end = int(round(video_end_token * ratio))
    a_end = min(t_audio_total, max(a_start + 1, a_end))
    return audio_latent[..., a_start:a_end]


@dataclass
class H3Context:
    session_id: str
    temp_dir: Path
    # Target Canvas
    target_width: int
    target_height: int
    target_megapixels: float
    output_aspect_ratio: str
    framing_mode: str  # "Crop_To_Fill", "Pad_Letterbox", "Stretch_Anamorphic"
    # Source Video Metadata
    src_width: int
    src_height: int
    fps: float
    total_frames: int
    # Temporal Chunking
    temporal_chunk_frames: int = 120
    temporal_prefix_frames: int = 12
    temporal_blend_frames: int = 8
    total_temporal_chunks: int = 1
    face_refine_scope: str = "Per_Temporal_Chunk"  # "Per_Temporal_Chunk" or "Whole_Video"
    # Tile Geometry
    tile_width: int = 0
    tile_height: int = 0
    tile_megapixels: float = 0.92
    grid_cols: int = 1
    grid_rows: int = 1
    overlap_percent: float = 0.25
    overlap_px_x: int = 0
    overlap_px_y: int = 0
    # File Anchors
    overview_keyframe_path: Optional[Path] = None
    base_canvas_video_path: Path = Path(".")
    audio_path: Optional[Path] = None
    # Debug / Cleanup
    keep_cache: bool = False
    debug_decode_temp_chunks: bool = False
    save_latent: bool = False
    save_latents_separately: bool = False
    saved_latent_path: Optional[Path] = None
    saved_latent_paths: List[Path] = field(default_factory=list)
    # Vision Model Chunk Prompts
    chunk_prompts: Dict[int, str] = field(default_factory=dict)
    saved_prompt_path: Optional[Path] = None
    # Latent-Native Pipeline Extensions
    is_latent_source: bool = True
    source_latent_path: Optional[Path] = None
    video_latent: Optional[Any] = None
    audio_latent: Optional[Any] = None
    base_canvas_latent_path: Optional[Path] = None
    preview_vae: Optional[Any] = None
    # Reference Asset Bundle & Prompt Consolidation
    reference_bundle: Optional[H3ReferenceBundle] = None
    h3_prompt: str = ""
    last_enhanced_frame_latent: Optional[Any] = None

@dataclass
class TileDescriptor:
    tile_idx: int
    chunk_idx: int
    row: int
    col: int
    # Bounding box in Canvas space: (x1, y1, x2, y2)
    bbox_canvas: Tuple[int, int, int, int]
    base_crop_video_path: Path
    enhanced_crop_video_path: Path
    prefix_crop_video_path: Optional[Path] = None  # Preceding 0.5s from Chunk k-1 (<Video 2>)
    prompt_sections: Dict[str, str] = field(default_factory=dict)
    chunk_start_frame: int = 0
    # Latent tile descriptors
    base_crop_latent_path: Optional[Path] = None
    enhanced_crop_latent_path: Optional[Path] = None
    tile_latent: Optional[Any] = None
    enhanced_tile_latent: Optional[Any] = None

@dataclass
class VLMContainer:
    model: Any
    processor: Any
    model_type: str  # "qwen2_5_vl", "florence2", "custom"
    device: str
    precision: str
    model_name_or_path: str
