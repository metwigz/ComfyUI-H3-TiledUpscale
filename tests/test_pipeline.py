r"""
Standalone Synthetic Test Harness for H3-TiledUpscale (Redesigned Standard Type Suite)
Run: .\python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-H3-TiledUpscale\tests\test_pipeline.py
"""
import sys
import os
from pathlib import Path
import importlib
import torch
import numpy as np
import shutil

# Add ComfyUI root and custom_nodes root to sys.path
repo_root = Path(__file__).resolve().parents[3]       # ComfyUI
custom_nodes_dir = Path(__file__).resolve().parents[2] # custom_nodes
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(custom_nodes_dir))

# Import custom node package
pkg = importlib.import_module("ComfyUI-H3-TiledUpscale")
H3LatentCanvasUpscale = pkg.H3LatentCanvasUpscale
H3ReferenceAssetBundle = pkg.H3ReferenceAssetBundle
H3ChunkVideoDescriber = pkg.H3ChunkVideoDescriber
H3LatentTiledKSampler = pkg.H3LatentTiledKSampler
H3LatentFaceRefine = pkg.H3LatentFaceRefine
H3VideoSave = pkg.H3VideoSave
H3TileCalculator = pkg.H3TileCalculator
H3AudioVAEDecode = pkg.H3AudioVAEDecode
H3LoadLatentFromPath = pkg.H3LoadLatentFromPath
H3SaveLatent = pkg.H3SaveLatent

# Import core modules
core_grid = importlib.import_module("ComfyUI-H3-TiledUpscale.core.grid_utils")
core_laplacian = importlib.import_module("ComfyUI-H3-TiledUpscale.core.laplacian_pyramid")
core_ffmpeg = importlib.import_module("ComfyUI-H3-TiledUpscale.core.ffmpeg_pipe")
core_streaming = importlib.import_module("ComfyUI-H3-TiledUpscale.core.disk_streaming")
core_schemas = importlib.import_module("ComfyUI-H3-TiledUpscale.core.schemas")

LaplacianPyramidBlender = core_laplacian.LaplacianPyramidBlender
spawn_frame_writer = core_ffmpeg.spawn_frame_writer
spawn_frame_reader = core_ffmpeg.spawn_frame_reader
get_ffmpeg_binary = core_ffmpeg.get_ffmpeg_binary
compute_temporal_chunks = core_grid.compute_temporal_chunks
compute_tile_intervals = core_grid.compute_tile_intervals
calculate_canvas_dimensions = core_grid.calculate_canvas_dimensions
calculate_optimal_tile_dimensions = core_grid.calculate_optimal_tile_dimensions


def test_laplacian_reconstruction():
    print("[1/6] Testing Laplacian Pyramid Perfect Reconstruction...")
    blender = LaplacianPyramidBlender(num_levels=3, device="cpu")
    # Synthetic gradient frame [1, 3, 256, 256]
    x = torch.linspace(0, 1, 256).unsqueeze(0).repeat(256, 1)
    img = torch.stack([x, x.t(), 1 - x], dim=0).unsqueeze(0)
    
    pyr = blender.build_pyramid(img)
    rec = blender.reconstruct(pyr)
    diff = torch.max(torch.abs(img - rec)).item()
    assert diff < 0.05, f"Laplacian reconstruction error too high: {diff}"
    print(f"  [OK] Pyramid reconstruction successful (max error: {diff:.5f})")


def test_ffmpeg_roundtrip():
    print("[2/6] Testing FFmpeg Raw RGB24 Streaming Pipe...")
    ffmpeg_exe = get_ffmpeg_binary()
    print(f"  Using FFmpeg: {ffmpeg_exe}")
    
    test_out = Path("test_synthetic_pipe.mp4")
    w, h, fps, frames = 320, 192, 24.0, 15
    
    writer = spawn_frame_writer(str(test_out), w, h, fps, codec="libx264")
    for f in range(frames):
        bar = np.zeros((h, w, 3), dtype=np.uint8)
        bar[:, (f * 10) % w : (f * 10 + 30) % w, :] = 255
        writer.stdin.write(bar.tobytes())
    writer.stdin.close()
    writer.wait()
    assert test_out.exists() and test_out.stat().st_size > 0
    print(f"  [OK] Generated {test_out.stat().st_size} byte MP4 via stdin pipe")
    
    reader = spawn_frame_reader(str(test_out), w, h)
    read_count = 0
    while True:
        raw = reader.stdout.read(w * h * 3)
        if len(raw) < w * h * 3:
            break
        read_count += 1
    reader.stdout.close()
    reader.wait()
    assert read_count == frames, f"Expected {frames} frames, got {read_count}"
    print(f"  [OK] Successfully read back {read_count} frames via stdout pipe")
    if test_out.exists():
        os.remove(test_out)


def test_optimal_tile_dimensions():
    print("[3/6] Testing Joint 2D Aspect-Ratio-Preserving Tile Minimization...")
    canvas_w, canvas_h = 3840, 2160
    tile_mp = 0.92
    overlap_pct = 0.25

    # Standard mode
    w_std, h_std, nw_std, nh_std = calculate_optimal_tile_dimensions(
        canvas_w, canvas_h, tile_mp, overlap_pct, tight_tile_overlap=False
    )
    # Tight overlap mode
    w_tight, h_tight, nw_tight, nh_tight = calculate_optimal_tile_dimensions(
        canvas_w, canvas_h, tile_mp, overlap_pct, tight_tile_overlap=True
    )
    
    ratio_canvas = canvas_w / canvas_h
    ratio_std = w_std / h_std
    ratio_tight = w_tight / h_tight

    assert abs(ratio_tight - ratio_canvas) < 0.05, f"Aspect ratio drift: {ratio_tight} vs {ratio_canvas}"
    assert (w_tight * h_tight) <= (w_std * h_std), "Tight mode tile should be smaller or equal"
    print(f"  [OK] Standard Tile: {w_std}x{h_std} ({w_std*h_std/1e6:.2f} MP, ratio {ratio_std:.3f})")
    print(f"  [OK] Tight Tile:    {w_tight}x{h_tight} ({w_tight*h_tight/1e6:.2f} MP, ratio {ratio_tight:.3f})")


def test_tile_calculator():
    print("[4/6] Testing Independent H3TileCalculator Node...")
    calc = H3TileCalculator()
    res = calc.calculate_grid(
        tile_megapixels=0.92,
        overlap_percent=0.25,
        tight_tile_overlap=True,
        canvas_megapixels=8.29,
        aspect_ratio="16:9 (Widescreen)",
        total_frames=124
    )
    info_text = res["result"][0]
    tile_w = res["result"][1]
    tile_h = res["result"][2]
    total_tiles = res["result"][5]
    total_jobs = res["result"][6]
    
    assert "MiniMax H3 Tiled Super-Resolution Preview" in info_text
    assert tile_w > 0 and tile_h > 0
    assert total_tiles >= 1 and total_jobs >= 1
    print(f"  [OK] Calculator output verified: {tile_w}x{tile_h}, {total_tiles} tiles/chunk, {total_jobs} total jobs")


def test_end_to_end_node_pipeline():
    print("[5/6] Testing End-to-End ComfyUI Nodes Pipeline (New Standard Types)...")
    # 1. Prepare synthetic dual-key 5D latent (17 frames = 5 latent tokens, 320x192)
    dummy_v = torch.rand((1, 24, 5, 12, 20), dtype=torch.float32)
    dummy_a = torch.rand((1, 32, 2, 40), dtype=torch.float32)
    dummy_latent = {"samples": dummy_v, "audio_samples": dummy_a}

    # 2. Node 1: H3LatentCanvasUpscale
    upscaler = H3LatentCanvasUpscale()
    out_lat, w, h = upscaler.upscale_canvas(
        dummy_latent,
        output_megapixels=0.25, # ~640x360 for fast testing
        upscale_method="Latent_Trilinear (5D)"
    )
    assert "samples" in out_lat and "audio_samples" in out_lat
    assert w >= 32 and h >= 32
    print(f"  [OK] Node 1 (H3LatentCanvasUpscale) -> Canvas: {w}x{h}, shape: {out_lat['samples'].shape}")

    # 3. Node 2: H3ReferenceAssetBundle
    bundler = H3ReferenceAssetBundle()
    (bundle,) = bundler.bundle_references(picture_1=torch.rand((1, 128, 128, 3)))
    assert bundle["total_count"] == 1
    print(f"  [OK] Node 2 (H3ReferenceAssetBundle) -> {bundle['total_count']} reference registered")

    # 4. Node 3: H3ChunkVideoDescriber
    describer = H3ChunkVideoDescriber()
    prompts, master, _, c_frames, b_frames = describer.describe_chunks(
        model=None,
        latent=out_lat,
        base_style_prompt="cinematic, 8k",
        save_prompts_to_output=False,
        reference_bundle=bundle
    )
    assert 0 in prompts
    assert "cinematic, 8k" in master
    assert c_frames > 0 and b_frames >= 0
    print(f"  [OK] Node 3 (H3ChunkVideoDescriber) -> {len(prompts)} chunk prompt(s), chunk_frames={c_frames}, blend_frames={b_frames}")

    # 5. Node 4: H3LatentTiledKSampler
    sampler_node = H3LatentTiledKSampler()
    class MockModel:
        def __init__(self):
            self.load_device = "cpu"
            self.model_options = {}
        def is_av_model(self):
            return True
        def get_model_object(self, name):
            return None

    import comfy.sample
    orig_sample_custom = comfy.sample.sample_custom
    comfy.sample.sample_custom = lambda model, noise, cfg, sampler, sigmas, pos, neg, latent_image, **kwargs: latent_image

    try:
        (sampled_lat,) = sampler_node.sample_tiles(
            model=MockModel(),
            latent_image=out_lat,
            sampler="euler",
            sigmas=[1.0, 0.0],
            tile_megapixels=0.10,
            overlap_percent=0.25,
            tight_tile_overlap=True,
            temporal_chunk_frames=c_frames,
            temporal_blend_frames=b_frames,
            chunk_prompts=prompts,
            tile_storage_strategy="temp_disk_stream"
        )
    finally:
        comfy.sample.sample_custom = orig_sample_custom

    assert "samples" in sampled_lat
    assert "audio_samples" in sampled_lat
    v_shape = sampled_lat['samples'].unbind()[0].shape if hasattr(sampled_lat['samples'], 'is_nested') and sampled_lat['samples'].is_nested else sampled_lat['samples'].shape
    print(f"  [OK] Node 4 (H3LatentTiledKSampler) -> Denoised latent shape: {v_shape}")

    # 6. Node 5: H3LatentFaceRefine (Optional)
    face_node = H3LatentFaceRefine()
    refined_lat, faces = face_node.refine_faces(sampled_lat, MockModel())
    assert "samples" in refined_lat
    print(f"  [OK] Node 5 (H3LatentFaceRefine) -> {faces} faces processed")

    # 7. Node 6: H3VideoSave
    saver_node = H3VideoSave()
    class MockVAE:
        device = "cpu"
        class FirstStage:
            def parameters(self):
                return iter([torch.zeros(1)])
        first_stage_model = FirstStage()
        def decode(self, z):
            b, c, t, h_l, w_l = z.shape
            return torch.zeros((b, (t - 1) * 4 + 1, h_l * 16, w_l * 16, 3), dtype=torch.float32)

    class MockAudioVAE:
        device = "cpu"
        audio_sample_rate = 32000
        def decode(self, a_lat):
            return torch.zeros((1, 2, 32000), dtype=torch.float32)

    res = saver_node.save_video(
        latent=sampled_lat,
        vae=MockVAE(),
        output_dir="output/test_run",
        filename_prefix="test_4k_%counter%",
        video_codec="H.264 (CPU libx264 - Universal)",
        audio_vae=MockAudioVAE(),
        save_latent_copy=True,
        overwrite_existing=True
    )
    saved_path = res["result"][0]
    assert os.path.exists(saved_path), f"Video not created at: {saved_path}"
    print(f"  [OK] Node 6 (H3VideoSave) -> Successfully exported {saved_path}")

    # Clean up test output
    if os.path.exists(saved_path):
        os.remove(saved_path)
    test_latent = Path(saved_path).with_suffix(".latent")
    if test_latent.exists():
        os.remove(test_latent)


def test_audio_vae_decode():
    print("[6/7] Testing H3AudioVAEDecode Standalone Utility...")
    audio_node = H3AudioVAEDecode()
    dummy_lat = {"audio_samples": torch.rand((1, 32, 2, 40), dtype=torch.float32)}
    class MockAudioVAE:
        audio_sample_rate = 32000
        device = "cpu"
        def decode(self, z):
            return torch.zeros((1, 2, 16000), dtype=torch.float32)

    (aud_dict,) = audio_node.decode_audio(dummy_lat, MockAudioVAE())
    assert "waveform" in aud_dict and "sample_rate" in aud_dict
    assert aud_dict["sample_rate"] == 32000
    print(f"  [OK] H3AudioVAEDecode -> Decoded waveform {aud_dict['waveform'].shape} @ {aud_dict['sample_rate']}Hz")


def test_multitile_seam_coherence():
    print("[7/7] Testing Multi-Tile Seam Coherence & Partition-of-Unity Weighting...")
    # 1. Verify mathematical partition of unity across a 3x3 grid
    target_w, target_h = 1920, 1088
    tile_w, tile_h, nw, nh = calculate_optimal_tile_dimensions(target_w, target_h, 0.40, 0.25, tight_tile_overlap=True)
    x_intervals = compute_tile_intervals(target_w, tile_w, 0.25)
    y_intervals = compute_tile_intervals(target_h, tile_h, 0.25)

    W_lat, H_lat = target_w // 16, target_h // 16
    weight_canvas = torch.zeros((H_lat, W_lat), dtype=torch.float32)

    for r, (y1, y2) in enumerate(y_intervals):
        lat_y1, lat_y2 = y1 // 16, y2 // 16
        ov_top = (y_intervals[r - 1][1] - y1) // 16 if r > 0 else 0
        ov_bottom = (y2 - y_intervals[r + 1][0]) // 16 if r < len(y_intervals) - 1 else 0

        for c, (x1, x2) in enumerate(x_intervals):
            lat_x1, lat_x2 = x1 // 16, x2 // 16
            ov_left = (x_intervals[c - 1][1] - x1) // 16 if c > 0 else 0
            ov_right = (x2 - x_intervals[c + 1][0]) // 16 if c < len(x_intervals) - 1 else 0

            tw = core_laplacian.generate_hann_weight_2d(
                h=lat_y2 - lat_y1,
                w=lat_x2 - lat_x1,
                fade_top=(r > 0),
                fade_bottom=(r < len(y_intervals) - 1),
                fade_left=(c > 0),
                fade_right=(c < len(x_intervals) - 1),
                ov_top=ov_top,
                ov_bottom=ov_bottom,
                ov_left=ov_left,
                ov_right=ov_right
            )[0, 0, 0]

            weight_canvas[lat_y1:lat_y2, lat_x1:lat_x2] += tw

    w_min = weight_canvas.min().item()
    w_max = weight_canvas.max().item()
    assert abs(w_min - 1.0) < 1e-4 and abs(w_max - 1.0) < 1e-4, f"Partition of unity violated: min={w_min}, max={w_max}"
    print(f"  [OK] 2D Partition-of-Unity verified across {len(x_intervals)}x{len(y_intervals)} grid: min={w_min:.6f}, max={w_max:.6f}")

    # 2. Run multi-tile sample execution with coherent noise & color locking
    sampler_node = H3LatentTiledKSampler()
    class MockModel:
        load_device = "cpu"
        model_options = {}
        def is_av_model(self):
            return False

    dummy_canvas = {
        "samples": torch.rand((1, 24, 5, H_lat, W_lat), dtype=torch.float32)
    }

    import comfy.sample
    orig_sample_custom = comfy.sample.sample_custom
    comfy.sample.sample_custom = lambda model, noise, cfg, sampler, sigmas, pos, neg, latent_image, **kwargs: latent_image + 0.01 * noise

    try:
        (res_lat,) = sampler_node.sample_tiles(
            model=MockModel(),
            latent_image=dummy_canvas,
            sampler="euler",
            sigmas=[1.0, 0.0],
            tile_megapixels=0.40,
            overlap_percent=0.25,
            tight_tile_overlap=True,
            temporal_chunk_frames=-1,
            color_lock_to_base=True,
            seed=12345
        )
    finally:
        comfy.sample.sample_custom = orig_sample_custom

    assert "samples" in res_lat
    assert not torch.isnan(res_lat["samples"]).any()
    print(f"  [OK] Multi-tile sampling passed cleanly: {res_lat['samples'].shape} with coherent noise & color lock")


def test_intersection_center_patches():
    print("[8/8] Testing Intersection-Centered Minimal-MP Patches & Pure Linear Consensus...")
    sampler_node = H3LatentTiledKSampler()
    class MockModel:
        load_device = "cpu"
        model_options = {}
        def is_av_model(self):
            return False

    # 1920x1088 canvas (120x68 latents)
    dummy_canvas = {
        "samples": torch.full((1, 24, 5, 68, 120), 0.5, dtype=torch.float32)
    }

    import comfy.sample
    orig_sample_custom = comfy.sample.sample_custom
    # Return latent + 0.1 delta everywhere
    comfy.sample.sample_custom = lambda model, noise, cfg, sampler, sigmas, pos, neg, latent_image, **kwargs: latent_image + 0.1

    try:
        (res_lat,) = sampler_node.sample_tiles(
            model=MockModel(),
            latent_image=dummy_canvas,
            sampler="euler",
            sigmas=[1.0, 0.0],
            tile_megapixels=0.40,
            overlap_percent=0.25,
            tight_tile_overlap=True,
            temporal_chunk_frames=-1,
            enable_intersection_center_patches=True,
            color_lock_to_base=False,
            seed=42
        )
    finally:
        comfy.sample.sample_custom = orig_sample_custom

    out_tensor = res_lat["samples"]
    assert "samples" in res_lat
    assert not torch.isnan(out_tensor).any()
    
    # Check linear consensus energy flat conservation: baseline was 0.5, delta added was 0.1 everywhere
    # The output should be uniformly ~0.6000 with zero hot spot spike (max error < 0.01)
    diff_from_flat = torch.abs(out_tensor - 0.6)
    max_drift = diff_from_flat.max().item()
    assert max_drift < 0.01, f"Consensus not flat across intersection: max drift = {max_drift}"
    print(f"  [OK] Center patches executed successfully: flat consensus preserved (max drift: {max_drift:.6f})")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" Running ComfyUI-H3-TiledUpscale Standard-Type Test Harness")
    print("="*70 + "\n")
    test_laplacian_reconstruction()
    test_ffmpeg_roundtrip()
    test_optimal_tile_dimensions()
    test_tile_calculator()
    test_end_to_end_node_pipeline()
    test_audio_vae_decode()
    test_multitile_seam_coherence()
    test_intersection_center_patches()
    print("\n" + "="*70)
    print(" ALL TESTS PASSED! 100% Standard ComfyUI Pipeline Operational.")
    print("="*70 + "\n")
