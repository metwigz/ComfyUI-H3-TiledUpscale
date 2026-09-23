import torch
from typing import Dict, Any

from ..core.tensor_utils import unpack_latent

class H3AudioVAEDecode:
    """
    Decodes the embedded audio latent ('audio_samples' tensor [1, 32, 2, T_audio])
    from a MiniMax H3 dual-key LATENT into standard ComfyUI AUDIO dict ({'waveform': ..., 'sample_rate': ...}).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {
                    "tooltip": "Latent dictionary containing the embedded 'audio_samples' tensor [1, 32, 2, T_audio] synchronized with the video latent."
                }),
                "audio_vae": ("VAE", {
                    "tooltip": "MiniMax Audio VAE model used to decode the embedded audio latent into standard PCM waveform audio."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode_audio"
    CATEGORY = "H3-TiledUpscale/audio"

    def decode_audio(self, samples: Dict[str, Any], audio_vae: Any):
        _, audio_latent = unpack_latent(samples)
        if audio_latent is None:
            raise ValueError("[H3AudioVAEDecode] Provided LATENT contains no audio samples!")

        dev = audio_vae.device if hasattr(audio_vae, "device") else "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            pcm = audio_vae.decode(audio_latent.to(dev))

        sample_rate = getattr(audio_vae, "audio_sample_rate", 32000)
        if isinstance(pcm, torch.Tensor):
            pcm = pcm.detach().cpu()
            if pcm.ndim == 2:
                pcm = pcm.unsqueeze(0)  # [1, channels, samples]

        return ({"waveform": pcm, "sample_rate": sample_rate},)
