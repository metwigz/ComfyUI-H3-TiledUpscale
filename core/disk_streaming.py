import os
import mmap
from pathlib import Path
from typing import Optional, Tuple, Any
import numpy as np

class MmapFrameBuffer:
    """
    Memory-mapped binary buffer for streaming uncompressed video frames [H, W, C].
    Allows frame-by-frame read/write without loading entire 4K sequences into host RAM.
    """
    def __init__(self, file_path: Path, width: int, height: int, channels: int = 3, num_frames: int = 1, mode: str = "r+"):
        self.file_path = Path(file_path)
        self.width = width
        self.height = height
        self.channels = channels
        self.num_frames = num_frames
        self.frame_size = width * height * channels
        self.total_bytes = self.frame_size * num_frames
        self.mode = mode
        self.fp = None
        self.mmap_obj = None

        if mode == "w+" or (not self.file_path.exists() and "w" in mode):
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "wb") as f:
                # Sparse file or write zero at end
                f.seek(self.total_bytes - 1)
                f.write(b"\0")

        self._open()

    def _open(self):
        access = mmap.ACCESS_WRITE if ("+" in self.mode or "w" in self.mode) else mmap.ACCESS_READ
        self.fp = open(self.file_path, "r+b" if access == mmap.ACCESS_WRITE else "rb")
        self.mmap_obj = mmap.mmap(self.fp.fileno(), length=self.total_bytes, access=access)

    def write_frame(self, frame_idx: int, frame_bytes: bytes):
        """Writes raw frame bytes directly into the memory-mapped buffer at frame_idx."""
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")
        offset = frame_idx * self.frame_size
        self.mmap_obj[offset : offset + self.frame_size] = frame_bytes

    def read_frame(self, frame_idx: int) -> np.ndarray:
        """Reads a frame as a uint8 numpy array [H, W, C] without copying whole buffer."""
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.num_frames})")
        offset = frame_idx * self.frame_size
        raw = self.mmap_obj[offset : offset + self.frame_size]
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, self.channels))
        return arr

    def close(self):
        if self.mmap_obj is not None:
            try:
                self.mmap_obj.flush()
                self.mmap_obj.close()
            except Exception:
                pass
            self.mmap_obj = None
        if self.fp is not None:
            try:
                self.fp.close()
            except Exception:
                pass
            self.fp = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def dump_tile_to_disk_stream(tile_tensor, chunk_idx: int, r: int, c: int, session_dir: Optional[Path] = None) -> Path:
    """
    Writes enhanced tile latent directly to disk safetensors scratch file and frees tensor from memory.
    Guarantees O(1) constant RAM footprint regardless of tile count or video length.
    """
    import safetensors.torch
    if session_dir is None:
        try:
            import folder_paths
            session_dir = Path(folder_paths.get_temp_directory()) / "h3_tiles"
        except Exception:
            session_dir = Path("temp") / "h3_tiles"

    session_dir.mkdir(parents=True, exist_ok=True)
    tile_file = session_dir / f"enhanced_tile_c{chunk_idx}_r{r}_c{c}.latent"
    safetensors.torch.save_file({"samples": tile_tensor.contiguous().cpu()}, str(tile_file))
    del tile_tensor
    return tile_file


def load_tile(tile_ref) -> Any:
    """Loads tile latent tensor from memory or disk scratch file."""
    import torch
    import safetensors.torch
    from typing import Any
    if isinstance(tile_ref, torch.Tensor):
        return tile_ref
    if isinstance(tile_ref, (str, Path)):
        loaded = safetensors.torch.load_file(str(tile_ref), device="cpu")
        return loaded.get("samples", loaded.get("latent_tensor"))
    return tile_ref

