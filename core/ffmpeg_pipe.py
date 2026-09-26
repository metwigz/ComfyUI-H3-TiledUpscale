import os
import shutil
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

def get_ffmpeg_binary() -> str:
    """
    Finds ffmpeg strictly within the ComfyUI Python environment or root.
    NEVER queries or executes from the OS system PATH.
    """
    # 1. Explicit ComfyUI environment variable override
    env_path = os.environ.get("COMFYUI_FFMPEG_PATH") or os.environ.get("VHS_FORCE_FFMPEG_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Check imageio_ffmpeg bundled inside ComfyUI Python's site-packages
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return str(exe)
    except (ImportError, Exception):
        pass

    # 3. Check local ComfyUI root and python_embeded directories only
    possible_roots = [
        Path(os.getcwd()),
        Path(os.getcwd()).parent,
        Path(__file__).resolve().parents[3], # ComfyUI base
        Path(__file__).resolve().parents[4], # Portable base
    ]
    for root in possible_roots:
        candidates = [
            root / "ffmpeg.exe",
            root / "python_embeded" / "Scripts" / "ffmpeg.exe",
            root / "python_embeded" / "ffmpeg.exe",
            root / "custom_nodes" / "ComfyUI-DLSS5-Enhancer" / "ffmpeg" / "bin" / "ffmpeg.exe"
        ]
        for cand in candidates:
            if cand.is_file():
                return str(cand)

    raise RuntimeError(
        "FFmpeg binary not found inside ComfyUI environment. "
        "Please run: .\\python_embeded\\python.exe -m pip install imageio-ffmpeg "
        "or place ffmpeg.exe in your ComfyUI root folder."
    )

def check_nvenc() -> bool:
    """Checks if FFmpeg has working h264_nvenc hardware acceleration."""
    try:
        ffmpeg = get_ffmpeg_binary()
        res = subprocess.run([ffmpeg, "-encoders"], capture_output=True, text=True, timeout=5)
        return "h264_nvenc" in res.stdout
    except Exception:
        return False

def probe_video(video_path: str) -> Dict[str, Any]:
    """
    Probes video metadata (width, height, fps, total_frames, duration, has_audio)
    using ComfyUI's native PyAV library (av). Never uses system PATH or external binaries.
    """
    import av
    video_path = str(video_path)
    with av.open(video_path) as container:
        if not container.streams.video:
            raise RuntimeError(f"No video streams found in: {video_path}")
        v_stream = container.streams.video[0]
        has_audio = len(container.streams.audio) > 0
        w = int(v_stream.width)
        h = int(v_stream.height)
        rate = v_stream.average_rate or v_stream.base_rate
        fps = float(rate) if rate else 24.0
        total_frames = int(v_stream.frames or 0)
        if total_frames <= 0 and container.duration:
            duration_sec = float(container.duration / av.time_base)
            total_frames = max(1, int(round(duration_sec * fps)))
        return {
            "width": w,
            "height": h,
            "fps": fps,
            "total_frames": max(1, total_frames),
            "duration": float(container.duration / av.time_base) if container.duration else 0.0,
            "has_audio": has_audio
        }

def spawn_frame_reader(video_path: str, width: int, height: int, start_frame: int = 0, frame_count: Optional[int] = None):
    """
    Spawns an FFmpeg subprocess that outputs raw RGB24 video bytes over stdout.
    Optionally skips to start_frame and limits to frame_count.
    """
    ffmpeg = get_ffmpeg_binary()
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
    
    if start_frame > 0:
        # Note: -vf select or frame stepping. Slower seek via filter ensures exact frame accuracy:
        cmd.extend(["-i", str(video_path), "-vf", f"select=gte(n\\,{start_frame})"])
    else:
        cmd.extend(["-i", str(video_path)])

    if frame_count is not None and frame_count > 0:
        cmd.extend(["-vframes", str(frame_count)])

    cmd.extend([
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "-"
    ])
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=width * height * 3 * 16
    )
    return process

def spawn_frame_writer(output_path: str, width: int, height: int, fps: float, codec: str = "h264_nvenc", bitrate_mbps: int = 40):
    """
    Spawns an FFmpeg subprocess that consumes raw RGB24 video bytes from stdin.
    Supports dynamic hardware and software encoders: h264_nvenc, hevc_nvenc, prores, libx264, libsvtav1.
    """
    ffmpeg = get_ffmpeg_binary()
    codec_lower = codec.lower()

    if "prores" in codec_lower:
        encoder_args = ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
    elif "hevc" in codec_lower or "265" in codec_lower:
        if check_nvenc():
            encoder_args = ["-c:v", "hevc_nvenc", "-preset", "p4", "-b:v", f"{bitrate_mbps}M", "-pix_fmt", "p010le"]
        else:
            encoder_args = ["-c:v", "libx265", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p10le"]
    elif "av1" in codec_lower:
        encoder_args = ["-c:v", "libsvtav1", "-preset", "6", "-crf", "22", "-pix_fmt", "yuv420p10le"]
    elif "libx264" in codec_lower or "cpu" in codec_lower:
        encoder_args = ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]
    else: # Default h264_nvenc with fallback to libx264
        if check_nvenc():
            encoder_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", f"{bitrate_mbps}M", "-pix_fmt", "yuv420p"]
        else:
            encoder_args = ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        *encoder_args,
        str(output_path)
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=width * height * 3 * 16)
    return process

def extract_audio(video_path: str, output_audio_path: str) -> bool:
    """Extracts audio from video to an audio file (e.g. .aac or .wav). Returns True if audio was found."""
    ffmpeg = get_ffmpeg_binary()
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-c:a", "copy",
        str(output_audio_path)
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode == 0 and os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
        return True
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)
    return False

def mux_audio_track(video_path: str, audio_path: Optional[str], output_path: str) -> bool:
    """Muxes audio into final video. If audio_path is None/invalid, moves/copies video without error. Returns True on success."""
    video_path_str = str(video_path)
    output_path_str = str(output_path)
    
    if not audio_path or not os.path.exists(str(audio_path)) or os.path.getsize(str(audio_path)) == 0:
        if video_path_str != output_path_str:
            shutil.copy2(video_path_str, output_path_str)
        return True

    try:
        ffmpeg = get_ffmpeg_binary()
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path_str,
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path_str
        ]
        res = subprocess.run(cmd, capture_output=True)
        return res.returncode == 0 and os.path.exists(output_path_str)
    except Exception:
        return False
