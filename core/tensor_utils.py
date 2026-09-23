import torch
import numpy as np
from pathlib import Path
from typing import Union, Any, Tuple, Optional, Dict

try:
    import comfy
    import comfy.nested_tensor
except ImportError:
    comfy = None

def unpack_latent(latent: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Universally and losslessly extracts (video_latent, audio_latent) from ANY
    ComfyUI LATENT container format:
      1. Standard ComfyUI NestedTensor: latent["samples"].is_nested == True -> [video, audio]
      2. Legacy / Custom tuple/list .tensors: latent["samples"].tensors -> [video, audio]
      3. Dict with separate keys: latent["samples"] (video) and latent["audio_samples"] (audio)
      4. Single video tensor: latent["samples"] -> (video, None)
      5. Raw Tensor passed directly
    """
    if latent is None:
        raise ValueError("[unpack_latent] Received None as latent input.")

    video_tensor = None
    audio_tensor = None

    if isinstance(latent, dict):
        raw_samples = latent.get("samples", None)
        audio_tensor = latent.get("audio_samples", None)
    elif isinstance(latent, torch.Tensor):
        raw_samples = latent
    else:
        raw_samples = latent

    # Check if raw_samples is a NestedTensor or wrapper
    if raw_samples is not None:
        if hasattr(raw_samples, "is_nested") and raw_samples.is_nested:
            unbound = raw_samples.unbind()
            if len(unbound) > 0:
                video_tensor = unbound[0]
            if len(unbound) > 1 and audio_tensor is None:
                audio_tensor = unbound[-1]
        elif hasattr(raw_samples, "tensors"):
            t_list = raw_samples.tensors
            if len(t_list) > 0:
                video_tensor = t_list[0]
            if len(t_list) > 1 and audio_tensor is None:
                audio_tensor = t_list[-1]
        elif isinstance(raw_samples, torch.Tensor):
            video_tensor = raw_samples
        else:
            video_tensor = raw_samples

    if video_tensor is None and isinstance(latent, dict):
        for k in ("latent_tensor", "video_latent", "video"):
            if k in latent and isinstance(latent[k], torch.Tensor):
                video_tensor = latent[k]
                break

    if audio_tensor is None and isinstance(latent, dict):
        for k in ("audio_latent_tensor", "audio_latent", "audio"):
            if k in latent and isinstance(latent[k], torch.Tensor):
                audio_tensor = latent[k]
                break

    # Detect if video_tensor is actually a standalone MiniMax audio latent [B, 32, 2, T_audio]
    if video_tensor is not None and audio_tensor is None:
        if video_tensor.ndim == 4 and video_tensor.shape[1] == 32 and video_tensor.shape[2] == 2:
            audio_tensor = video_tensor
            video_tensor = None

    if video_tensor is None and audio_tensor is None:
        raise ValueError(f"[unpack_latent] Could not extract valid video or audio latent tensor from input of type {type(latent)}.")

    return video_tensor, audio_tensor

def package_latent(
    video_latent: torch.Tensor,
    audio_latent: Optional[torch.Tensor] = None,
    extra_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Packages video and audio latent tensors into 100% compliant standard ComfyUI
    audio-video LATENT container:
      - out["samples"]: NestedTensor([video_latent, audio_latent]) if audio is present
      - out["samples"]: video_latent if pure video
      - out["audio_samples"]: audio_latent (for backwards and 3rd-party compatibility)
    """
    out = dict(extra_dict) if extra_dict is not None else {}

    if audio_latent is not None and comfy is not None and hasattr(comfy, "nested_tensor"):
        samples_payload = comfy.nested_tensor.NestedTensor([video_latent, audio_latent])
    else:
        samples_payload = video_latent

    out["samples"] = samples_payload
    if audio_latent is not None:
        out["audio_samples"] = audio_latent
    else:
        out.pop("audio_samples", None)

    return out

def tensor_frame_to_rgb24_bytes(frame: torch.Tensor) -> bytes:
    """
    Converts a single frame tensor [H, W, 3] or [1, H, W, 3] in range [0.0, 1.0]
    to contiguous RGB24 bytes for FFmpeg stdin pipes.
    """
    if frame.ndim == 4:
        frame = frame.squeeze(0)
    # Clamp, scale to 255, cast to uint8
    uint8_img = torch.clamp(frame, 0.0, 1.0).mul(255.0).to(torch.uint8).cpu().contiguous().numpy()
    return uint8_img.tobytes()

def rgb24_bytes_to_tensor_frame(raw_bytes: bytes, height: int, width: int, device: str = "cpu") -> torch.Tensor:
    """
    Converts raw RGB24 bytes from FFmpeg stdout pipe into a ComfyUI tensor [1, H, W, 3]
    in float32 range [0.0, 1.0].
    """
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, 3))
    tensor = torch.from_numpy(arr.copy()).float().div(255.0).unsqueeze(0).to(device)
    return tensor

def numpy_to_tensor_frame(img: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Converts a uint8 numpy RGB image [H, W, 3] to a ComfyUI tensor [1, H, W, 3] in [0.0, 1.0].
    """
    arr = img.copy() if not img.flags.writeable else img
    tensor = torch.from_numpy(arr).float().div(255.0).unsqueeze(0).to(device)
    return tensor

def tensor_to_numpy_frame(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a ComfyUI tensor [1, H, W, 3] or [H, W, 3] in [0.0, 1.0] to a uint8 numpy image [H, W, 3].
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    arr = torch.clamp(tensor, 0.0, 1.0).mul(255.0).to(torch.uint8).cpu().numpy()
    return arr

def save_waveform_to_wav(waveform: torch.Tensor, filepath: Union[str, Path], sample_rate: int = 32000) -> bool:
    """
    Saves a 1D, 2D, or 3D waveform tensor ([channels, samples] or [samples, channels])
    to a standard 16-bit PCM WAV audio file with correct channel configuration.
    """
    try:
        import wave
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        waveform = waveform.detach().cpu().float()
        
        # Squeeze batch dimension if present: [B, C, L] or [B, L, C] -> 2D
        if waveform.ndim == 3:
            waveform = waveform[0]
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Ensure shape is [channels, samples]. Audio channels are 1 (mono) or 2 (stereo).
        if waveform.shape[0] > waveform.shape[-1] and waveform.shape[-1] in (1, 2):
            waveform = waveform.t()
        elif waveform.shape[0] > 2 and waveform.ndim == 2 and waveform.shape[-1] <= 8:
            waveform = waveform.t()

        channels = min(2, max(1, waveform.shape[0]))
        waveform = waveform[:channels].clamp(-1.0, 1.0)

        interleaved = waveform.t().numpy()  # [samples, channels]
        int16_data = (interleaved * 32767.0).astype(np.int16)

        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(filepath_obj), 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(int16_data.tobytes())
        return filepath_obj.exists() and filepath_obj.stat().st_size > 0
    except Exception as e_wav:
        print(f"[H3-TiledUpscale] Warning: Failed saving WAV file {filepath}: {e_wav}")
        return False
