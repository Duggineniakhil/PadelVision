"""Ball detector adapters for the phase 2 bake-off. Each returns a DataFrame(frame, u, v, conf).

- YoloBall:     any Ultralytics YOLO model (V1 padel model, COCO "sports ball")
- TrackNetBall: yastrebksv/TrackNet (tennis, 3-frame heatmap net), using the repo's own code
- RoboflowBall: a model hosted on Roboflow Universe (needs an API key)

All take `frames`: dict frame_index -> BGR image. TrackNet also needs the two previous frames.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["frame", "u", "v", "conf"]


class YoloBall:
    def __init__(self, weights, classes: list[int] | None = None, imgsz: int = 1280,
                 conf: float = 0.05, device=None):  # fmt: skip
        from ultralytics import YOLO

        if device is None:
            import torch

            device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(weights))
        names = self.model.names
        if classes is None:  # every class whose name mentions a ball, else all classes
            classes = [i for i, n in names.items() if "ball" in str(n).lower()] or list(names)
        self.classes, self.names = classes, {i: names[i] for i in classes}
        self.imgsz, self.conf, self.device = imgsz, conf, device

    def detect(self, frames: dict[int, np.ndarray], ids) -> pd.DataFrame:
        rows = []
        for f in ids:
            r = self.model.predict(
                frames[f],
                classes=self.classes,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )[0]
            for (x1, y1, x2, y2), c in zip(
                r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), strict=True
            ):
                rows.append((f, (x1 + x2) / 2, (y1 + y2) / 2, float(c)))
        return pd.DataFrame(rows, columns=COLUMNS)

    def detect_video(
        self, video, stride: int = 1, max_frames: int | None = None, log_every: int = 1000
    ) -> pd.DataFrame:
        """Stream a whole video (frame indices are video frame numbers)."""
        import time

        rows, start = [], time.perf_counter()
        results = self.model.predict(
            source=str(video), stream=True, classes=self.classes, conf=self.conf,
            imgsz=self.imgsz, device=self.device, vid_stride=stride, verbose=False,
        )  # fmt: skip
        for i, r in enumerate(results):
            f = i * stride
            if max_frames is not None and f >= max_frames:
                break
            for (x1, y1, x2, y2), c in zip(
                r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), strict=True
            ):
                rows.append((f, (x1 + x2) / 2, (y1 + y2) / 2, float(c)))
            if i and i % log_every == 0:
                print(f"  ball detect: frame {f} ({i / (time.perf_counter() - start):.0f} fps)")
        return pd.DataFrame(rows, columns=COLUMNS)


class TrackNetBall:
    """Runs yastrebksv/TrackNet exactly as its infer_on_video.py does (640x360 input, current +
    2 previous frames, argmax heatmap -> Hough circle), using the cloned repo's modules."""

    def __init__(self, repo_dir, weights, device=None):
        import torch

        sys.path.insert(0, str(Path(repo_dir)))
        from general import postprocess  # noqa: PLC0415  (from the TrackNet repo)
        from model import BallTrackerNet  # noqa: PLC0415

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj = torch.load(str(weights), map_location=self.device, weights_only=False)
        if isinstance(obj, torch.nn.Module):
            self.model = obj
        else:
            state = (
                obj.get("state_dict", obj.get("model_state_dict", obj))
                if isinstance(obj, dict)
                else obj
            )
            self.model = BallTrackerNet()
            self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.postprocess = postprocess
        self._has_scale = "scale" in inspect.signature(postprocess).parameters

    def detect(self, frames: dict[int, np.ndarray], ids) -> pd.DataFrame:
        import cv2
        import torch

        rows = []
        for f in ids:
            if f - 2 not in frames or f - 1 not in frames:
                continue
            h, w = frames[f].shape[:2]
            imgs = [cv2.resize(frames[k], (640, 360)) for k in (f, f - 1, f - 2)]
            x = np.concatenate(imgs, axis=2).astype(np.float32) / 255.0
            x = torch.from_numpy(np.rollaxis(x, 2, 0)[None]).to(self.device)
            with torch.no_grad():
                out = self.model(x)
            heat = out.argmax(dim=1).detach().cpu().numpy()
            if self._has_scale:  # repo scales both axes by one factor (assumes 16:9)
                xp, yp = self.postprocess(heat, scale=1)
            else:
                xp, yp = self.postprocess(heat)
            if xp is not None and yp is not None:
                rows.append((f, float(xp) * w / 640, float(yp) * h / 360, 1.0))
        return pd.DataFrame(rows, columns=COLUMNS)


class RoboflowBall:
    """Roboflow hosted inference. `model_id` like "padel-ball-xyz/3". Frames are uploaded."""

    def __init__(self, model_id: str, api_key: str, conf: float = 0.05,
                 api_url: str = "https://detect.roboflow.com"):  # fmt: skip
        from inference_sdk import InferenceConfiguration, InferenceHTTPClient

        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self.client.configure(InferenceConfiguration(confidence_threshold=conf))
        self.model_id = model_id

    def detect(self, frames: dict[int, np.ndarray], ids) -> pd.DataFrame:
        rows = []
        for f in ids:
            res = self.client.infer(frames[f], model_id=self.model_id)
            for p in res.get("predictions", []):
                rows.append((f, float(p["x"]), float(p["y"]), float(p["confidence"])))
        return pd.DataFrame(rows, columns=COLUMNS)
