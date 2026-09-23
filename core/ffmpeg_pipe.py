import os
import shutil
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

def get_ffmpeg_binary() -> str:
    """Finds ffmpeg across system PATH, imageio_ffmpeg, and portable directories."""
    # 1. Check system PATH
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 2. Check imageio_ffmpeg
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return str(exe)
    except ImportError:
        pass

    # 3. Check ComfyUI root & python_embeded
    possible_roots = [
        Path(os.getcwd()),
        Path(os.getcwd()).parent,
        Path(__file__).resolve().parents[3]  # ComfyUI base
    ]
    for root in possible_roots:
        candidate = root / "ffmpeg.exe"
        if candidate.exists():
            return str(candidate)
        embed_candidate = root / "python_embeded" / "Scripts" / "ffmpeg.exe"
        if embed_candidate.exists():
            return str(embed_candidate)

    raise RuntimeError("FFmpeg executable not found. Please install imageio-ffmpeg or add ffmpeg to PATH.")

def get_ffprobe_binary() -> Optional[str]:
    """Finds ffprobe across system PATH, adjacent to ffmpeg, or imageio_ffmpeg."""
    found = shutil.which("ffprobe")
    if found:
        return found
    try:
        ffmpeg_exe = Path(get_ffmpeg_binary())
        probe_candidate = ffmpeg_exe.parent / (ffmpeg_exe.stem.replace("ffmpeg", "ffprobe") + ffmpeg_exe.suffix)
        if probe_candidate.exists():
            return str(probe_candidate)
        probe_std = ffmpeg_exe.parent / "ffprobe.exe"
        if probe_std.exists():
            return str(probe_std)
    except Exception:
        pass
    return None

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
    using ffprobe if available, or fallback to ffmpeg decoding check.
    """
    video_path = str(video_path)
    ffprobe = get_ffprobe_binary()
    if ffprobe:
        try:
            cmd = [
                ffprobe, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(res.stdout)
            stream = data["streams"][0]
            w = int(stream["width"])
            h = int(stream["height"])
            r_fps = stream.get("r_frame_rate", "24/1")
            if "/" in r_fps:
                num, den = r_fps.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 24.0
            else:
                fps = float(r_fps)
            
            nb_frames = stream.get("nb_frames")
            if nb_frames and nb_frames.isdigit() and int(nb_frames) > 0:
                total_frames = int(nb_frames)
            else:
                dur = float(stream.get("duration") or data.get("format", {}).get("duration", 0.0))
                total_frames = max(1, int(round(dur * fps)))

            # Check audio stream
            cmd_a = [
                ffprobe, "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=index",
                "-of", "json",
                video_path
            ]
            res_a = subprocess.run(cmd_a, capture_output=True, text=True)
            has_audio = len(json.loads(res_a.stdout).get("streams", [])) > 0

            return {
                "width": w,
                "height": h,
                "fps": fps,
                "total_frames": total_frames,
                "has_audio": has_audio
            }
        except Exception:
            pass

    # Fallback using OpenCV if ffprobe fails
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "width": w,
        "height": h,
        "fps": fps,
        "total_frames": max(1, total_frames),
        "has_audio": False
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
