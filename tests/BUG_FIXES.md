# Running Bug Fixes & Architectural Changelog

This document tracks identified bugs, root causes, and applied fixes for the **ComfyUI-H3-TiledUpscale** pipeline.

---

## 1. Outbound Audio Latent Truncation & Desync

- **Affected File**: `nodes/latent_tiled_ksampler.py`
- **Symptom**: Output video (`tiled_00006-audio.mp4`) had audio severely out of sync or dialogue abruptly cut off near the end.
- **Root Cause**:
  At the end of `H3LatentTiledKSampler.sample()`, the audio latent was arbitrarily sliced using:
  ```python
  audio_t = int(round(T_lat * (5.0 / 3.0)))
  input_audio_latent = input_audio_latent[..., :audio_t]
  ```
  For an 8-second video at 72 fps with $T_{\text{lat}} = 167$, $167 \times \frac{5}{3} \approx 278$ tokens. At 40 audio tokens/second, 278 tokens equals only **6.95 seconds**, truncating over a full second of the dialogue from the original 318-token (~7.97s) audio stream.
- **Fix**:
  Removed the arbitrary cropping of `input_audio_latent` before `package_latent()`. The sampler now outputs the full, uncropped input audio latent so `VAEDecodeAudio` and `VHS_VideoCombine` receive the complete audio duration.

---

## 2. DiT Audio Cross-Attention Hallucinating Lip-Sync on Off-Screen Dialogue

- **Affected Files**: `nodes/latent_tiled_ksampler.py`, `nodes/tile_calculator.py`
- **Symptom**: In runs with active audio, a character whose face was visible (e.g. Dr. Rush) moved his mouth to all dialogue lines, including off-screen characters (e.g. Colonel Young speaking with his back turned).
- **Root Cause**:
  The MiniMax H3 diffusion backbone uses audiovisual cross-attention. When audio is passed to a spatial tile containing a face, the model deforms the mouth to match the audio waveform. Because Dr. Rush was the sole visible face in that tile, the model forced his lips to articulate both his lines and Young's lines.
- **Fix**:
  Introduced the `tile_audio_mode` parameter:
  - `"active_audio"`: Passes audio tokens into the DiT for audio-driven face animation.
  - `"mute_during_upscale"`: Zeros out audio latent tokens fed to the DiT during tiled diffusion while preserving the full uncompressed audio latent for outbound export. This prevents hallucinated mouth movements during off-screen dialogue while preserving base canvas facial fidelity.

---

## 3. Spatial & Temporal Seam Artifacts / Ghosting ("Cross-Eyed" Glitch)

- **Affected File**: `nodes/latent_tiled_ksampler.py`
- **Symptom**: Characters intersecting spatial tile seams or temporal chunk transitions showed duplicated facial features, double irises, or cross-eyed artifacts.
- **Root Cause**:
  Independent pseudorandom noise generation per tile/chunk caused phase mismatches at tile overlap boundaries. When overlapping diffusion outputs were blended, the phase-shifted high frequencies resulted in ghosting.
- **Fix**:
  - Implemented single unified 5D canvas noise (`full_canvas_noise`) generated from a consistent seed on CPU.
  - All spatial tiles and temporal chunks slice their noise directly from this global canvas noise tensor, guaranteeing identical high-frequency noise at all overlapping boundaries.
  - Applied multiscale Laplacian blending and raised-cosine temporal windowing across transition frames.

---

## 4. MiniMax Video VAE Causal Token Drop (8 Frames / 111ms)

- **Affected Components**: `VAEEncode`, `VAEDecode` (`minimax_h3_video_vae_fp16.safetensors`)
- **Symptom**: 574 input frames at 72 fps (7.972s) decoded back to 566 frames (7.861s), losing 8 frames (~111ms) at the tail.
- **Root Cause**:
  MiniMax Video VAE operates on 17-frame clips producing 5 latent tokens per clip with a 3-token causal drop:
  $$T_{\text{lat}} = \left\lceil \frac{F}{17} \right\rceil \times 5 - 3$$
  For 574 frames, $\lceil 574 / 17 \rceil = 34 \times 5 - 3 = 167$ latent tokens. When decoded back, 167 latent tokens reconstruct exactly 566 frames.
- **Fix**:
  Documented causal boundary drop and ensured temporal chunking aligns on 17-frame / 5-token intervals. Avoided naive frame truncation to maintain temporal consistency with audio streams.

---

## 5. Non-24 FPS Audio Token Alignment for DiT Sampling

- **Affected File**: `nodes/latent_tiled_ksampler.py`
- **Symptom**: Audio-visual desync during sampling when input videos were not at MiniMax's native 24 fps (e.g. 72 fps interpolated video).
- **Root Cause**:
  MiniMax DiT RoPE grid assigns rotary position embeddings based on 24 fps time steps. At 72 fps, real-time audio tokens (40 tokens/s) did not match the temporal span expected by the DiT RoPE coordinates.
- **Fix**:
  Added dynamic linear resampling of chunk audio latent tokens to match the DiT's temporal grid:
  $$\text{dit\_audio\_t} = \max\left(1, \text{round}\left(\text{chunk\_frames} \times \frac{5}{3}\right)\right)$$
  This guarantees 1-to-1 temporal alignment between video frames and audio phonemes during diffusion cross-attention regardless of input framerate.
