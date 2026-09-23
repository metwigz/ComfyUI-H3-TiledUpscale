from typing import Optional, Dict, Any, Tuple, List
import torch

class H3ReferenceAssetBundle:
    """
    Consolidates multimodal references for MiniMax H3 (up to 9 pictures, 3 videos, 3 audios, max 12 total).
    Progressive dynamic UI reveals next slots as previous slots are connected.
    Downstream outputs plug into BOTH H3ChunkVideoDescriber and H3LatentTiledKSampler.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image_size": (["match", "max"], {
                    "default": "match",
                    "tooltip": "Reference image sizing. 'match' scales each ref (down only, keeping aspect) to the generation's pixel area; 'max' uses the reference pipeline's 2048px short edge for best identity fidelity. Reference tokens ride through every sampling step, so 'max' can be several times slower."
                }),
            },
            "optional": {
                "picture_1": ("IMAGE", {"tooltip": "Primary reference image (<Picture 1>). Connecting unhides picture_2."}),
                "picture_2": ("IMAGE", {"tooltip": "Reference image 2 (<Picture 2>) for character, style, or environmental consistency."}),
                "picture_3": ("IMAGE", {"tooltip": "Reference image 3 (<Picture 3>) for character, style, or environmental consistency."}),
                "picture_4": ("IMAGE", {"tooltip": "Reference image 4 (<Picture 4>) for character, style, or environmental consistency."}),
                "picture_5": ("IMAGE", {"tooltip": "Reference image 5 (<Picture 5>) for character, style, or environmental consistency."}),
                "picture_6": ("IMAGE", {"tooltip": "Reference image 6 (<Picture 6>) for character, style, or environmental consistency."}),
                "picture_7": ("IMAGE", {"tooltip": "Reference image 7 (<Picture 7>) for character, style, or environmental consistency."}),
                "picture_8": ("IMAGE", {"tooltip": "Reference image 8 (<Picture 8>) for character, style, or environmental consistency."}),
                "picture_9": ("IMAGE", {"tooltip": "Reference image 9 (<Picture 9>) for character, style, or environmental consistency."}),
                "video_1": ("IMAGE,VIDEO", {"tooltip": "Reference video clip 1 (<Video 1>) for dynamic motion, temporal progression, or scene reference."}),
                "video_2": ("IMAGE,VIDEO", {"tooltip": "Reference video clip 2 (<Video 2>) for dynamic motion, temporal progression, or scene reference."}),
                "video_3": ("IMAGE,VIDEO", {"tooltip": "Reference video clip 3 (<Video 3>) for dynamic motion, temporal progression, or scene reference."}),
                "audio_1": ("AUDIO", {"tooltip": "Reference audio track 1 (<Audio 1>) for synchronized sound design, voice, or music reference."}),
                "audio_2": ("AUDIO", {"tooltip": "Reference audio track 2 (<Audio 2>) for synchronized sound design, voice, or music reference."}),
                "audio_3": ("AUDIO", {"tooltip": "Reference audio track 3 (<Audio 3>) for synchronized sound design, voice, or music reference."}),
            }
        }

    RETURN_TYPES = ("H3_REFERENCE_BUNDLE",)
    RETURN_NAMES = ("reference_bundle",)
    FUNCTION = "bundle_references"
    CATEGORY = "H3-TiledUpscale"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def bundle_references(self, ref_image_size="match", **kwargs):
        pictures: List[torch.Tensor] = []
        for i in range(1, 10):
            p = kwargs.get(f"picture_{i}")
            if p is not None:
                if isinstance(p, torch.Tensor):
                    pictures.append(p)
                elif isinstance(p, (list, tuple)) and len(p) > 0 and isinstance(p[0], torch.Tensor):
                    pictures.append(p[0])

        videos: List[Any] = []
        for i in range(1, 4):
            v = kwargs.get(f"video_{i}")
            if v is not None:
                videos.append(v)

        audios: List[Any] = []
        for i in range(1, 4):
            a = kwargs.get(f"audio_{i}")
            if a is not None:
                audios.append(a)

        total_refs = len(pictures) + len(videos) + len(audios)
        if total_refs > 12:
            raise ValueError(f"[H3ReferenceAssetBundle] Exceeded maximum 12 total references ({total_refs} provided)!")

        class ReferenceBundleDict(dict):
            """Dict subclass that allows attribute-style access (e.g. bundle.pictures)."""
            def __getattr__(self, item):
                if item in self:
                    return self[item]
                raise AttributeError(f"'ReferenceBundleDict' object has no attribute '{item}'")
            def __setattr__(self, key, value):
                self[key] = value

        bundle = ReferenceBundleDict({
            "pictures": pictures,
            "videos": videos,
            "audios": audios,
            "ref_image_size": ref_image_size,
            "total_count": total_refs
        })

        print(f"[H3-TiledUpscale] Reference Asset Bundle: {len(pictures)} pictures (ref_image_size={ref_image_size}), {len(videos)} videos, {len(audios)} audios (Total: {total_refs}/12).")
        return (bundle,)
