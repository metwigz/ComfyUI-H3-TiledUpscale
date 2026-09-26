import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch

def resolve_yolo_face_model():
    """Searches ComfyUI model directories for face detector, falling back to auto-cached yolov8n-face or yolov8n."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "Face refinement requires 'ultralytics'. "
            "To use this optional feature, run: .\\python_embeded\\python.exe -m pip install ultralytics"
        )
    
    # Try ComfyUI folder_paths if available
    search_dirs = []
    try:
        import folder_paths
        search_dirs.extend([
            os.path.join(folder_paths.models_dir, "facedetection"),
            os.path.join(folder_paths.models_dir, "yolo"),
            os.path.join(folder_paths.models_dir, "ultralytics"),
        ])
    except Exception:
        pass
    
    # Also check local ComfyUI models relative to repo
    base_models = Path(__file__).resolve().parents[3] / "models"
    search_dirs.extend([
        str(base_models / "facedetection"),
        str(base_models / "yolo"),
        str(base_models / "ultralytics"),
    ])

    candidate_names = ["face_yolov8m.pt", "yolov8n-face.pt", "yolov8m-face.pt", "yolov8n.pt"]
    
    for d in search_dirs:
        if os.path.exists(d):
            for name in candidate_names:
                p = os.path.join(d, name)
                if os.path.exists(p):
                    return YOLO(p)
                    
    # Fallback: Ultralytics auto-download
    try:
        return YOLO("yolov8n-face.pt")
    except Exception:
        return YOLO("yolov8n.pt")


class FaceTracker:
    def __init__(self, min_size: int = 64, max_size: int = 768, conf_thresh: float = 0.4, ema_alpha: float = 0.65):
        self.min_size = min_size
        self.max_size = max_size
        self.conf_thresh = conf_thresh
        self.ema_alpha = ema_alpha
        self.model = None
        self.last_box: Optional[np.ndarray] = None

    def reset_stream(self):
        self.last_box = None

    def _ensure_model(self):
        if self.model is None:
            self.model = resolve_yolo_face_model()

    def detect_in_frame(self, frame_bgr_or_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects primary face in a single frame.
        Returns (x1, y1, x2, y2) or None if no valid face found within [min_size, max_size].
        """
        self._ensure_model()
        results = self.model(frame_bgr_or_rgb, verbose=False, conf=self.conf_thresh)
        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        best_box = None
        best_score = -1.0

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            size = max(w, h)
            if self.min_size <= size <= self.max_size:
                score = conf * size
                if score > best_score:
                    best_score = score
                    best_box = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

        return best_box

    def detect_and_smooth_frame(self, frame_bgr_or_rgb: np.ndarray, img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects primary face with real-time temporal EMA smoothing across video stream.
        """
        raw_box = self.detect_in_frame(frame_bgr_or_rgb)
        if raw_box is not None:
            box_arr = np.array(raw_box, dtype=np.float32)
            if self.last_box is None:
                smoothed = box_arr
            else:
                smoothed = self.ema_alpha * box_arr + (1.0 - self.ema_alpha) * self.last_box
            self.last_box = smoothed

            cx = (smoothed[0] + smoothed[2]) / 2.0
            cy = (smoothed[1] + smoothed[3]) / 2.0
            bw = (smoothed[2] - smoothed[0]) * 1.35
            bh = (smoothed[3] - smoothed[1]) * 1.35
            side = max(bw, bh)

            x1 = max(0, int(round(cx - side / 2.0)))
            y1 = max(0, int(round(cy - side / 2.0)))
            x2 = min(img_w, int(round(cx + side / 2.0)))
            y2 = min(img_h, int(round(cy + side / 2.0)))
            return (x1, y1, x2, y2)
        else:
            if self.last_box is not None:
                cx = (self.last_box[0] + self.last_box[2]) / 2.0
                cy = (self.last_box[1] + self.last_box[3]) / 2.0
                bw = (self.last_box[2] - self.last_box[0]) * 1.35
                bh = (self.last_box[3] - self.last_box[1]) * 1.35
                side = max(bw, bh)
                x1 = max(0, int(round(cx - side / 2.0)))
                y1 = max(0, int(round(cy - side / 2.0)))
                x2 = min(img_w, int(round(cx + side / 2.0)))
                y2 = min(img_h, int(round(cy + side / 2.0)))
                return (x1, y1, x2, y2)
            return None

    def track_sequence(
        self,
        frame_generator,
        total_frames: int,
        img_w: int,
        img_h: int
    ) -> List[Optional[Tuple[int, int, int, int]]]:
        """
        Tracks the primary face across a sequence of frames, applying temporal EMA smoothing.
        Returns a list of length total_frames with smoothed bounding boxes (x1, y1, x2, y2) or None.
        """
        smoothed_boxes: List[Optional[Tuple[int, int, int, int]]] = []
        last_box: Optional[np.ndarray] = None

        for idx, frame in enumerate(frame_generator):
            raw_box = self.detect_in_frame(frame)
            if raw_box is not None:
                box_arr = np.array(raw_box, dtype=np.float32)
                if last_box is None:
                    smoothed = box_arr
                else:
                    smoothed = self.ema_alpha * box_arr + (1.0 - self.ema_alpha) * last_box
                last_box = smoothed
                
                # Expand box slightly (35% margin) and clamp to bounds
                cx = (smoothed[0] + smoothed[2]) / 2.0
                cy = (smoothed[1] + smoothed[3]) / 2.0
                bw = (smoothed[2] - smoothed[0]) * 1.35
                bh = (smoothed[3] - smoothed[1]) * 1.35
                side = max(bw, bh)
                
                x1 = max(0, int(round(cx - side / 2.0)))
                y1 = max(0, int(round(cy - side / 2.0)))
                x2 = min(img_w, int(round(cx + side / 2.0)))
                y2 = min(img_h, int(round(cy + side / 2.0)))
                
                smoothed_boxes.append((x1, y1, x2, y2))
            else:
                # If undetected for 1-2 frames, keep decaying last_box
                if last_box is not None:
                    # Decay confidence or maintain position
                    smoothed_boxes.append((
                        max(0, int(round(last_box[0]))),
                        max(0, int(round(last_box[1]))),
                        min(img_w, int(round(last_box[2]))),
                        min(img_h, int(round(last_box[3])))
                    ))
                else:
                    smoothed_boxes.append(None)

        return smoothed_boxes

def generate_elliptical_feather_mask(h: int, w: int, feather: float = 0.25, device: str = "cpu") -> torch.Tensor:
    """
    Generates a 2D soft elliptical Gaussian feather mask [1, 1, H, W] for face blending.
    Center is 1.0, tapering smoothly towards the bounding box boundary to 0.0.
    """
    y = torch.linspace(-1.0, 1.0, h, device=device)[:, None]
    x = torch.linspace(-1.0, 1.0, w, device=device)[None, :]
    dist = torch.sqrt(x ** 2 + y ** 2)
    feather_band = max(0.05, min(0.95, float(feather)))
    inner_radius = max(0.0, 1.0 - feather_band)
    mask = torch.clamp((1.0 - dist) / feather_band, 0.0, 1.0)
    smooth_mask = 0.5 - 0.5 * torch.cos(mask * torch.pi)
    return smooth_mask.unsqueeze(0).unsqueeze(0)

