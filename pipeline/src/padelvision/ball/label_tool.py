"""Local ball-labelling tool for a review pack (made on Kaggle by `ball-review-pack`).

    padelvision label-ball data/ball_review

For each frame: Y / ENTER if the yellow ring is on the ball, click the ball if it's
elsewhere, N if no ball is visible. M toggles the motion view (moving things light up,
which makes a tiny ball much easier to spot). Labels are saved to labels.csv after every
answer, so you can quit (ESC) and resume any time.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

WINDOW = "PadelVision ball labelling"
MAX_DISPLAY = (1600, 900)
KEY_ENTER, KEY_ESC, KEY_BACKSPACE = 13, 27, 8
FIELDS = ["frame", "status", "u", "v", "source"]


class BallLabeler:
    def __init__(self, pack_dir: str | Path):
        self.dir = Path(pack_dir)
        self.items = json.loads((self.dir / "pack.json").read_text())["items"]
        self.labels: dict[int, dict] = {}
        self.csv = self.dir / "labels.csv"
        if self.csv.exists():
            with open(self.csv, newline="") as fh:
                for row in csv.DictReader(fh):
                    self.labels[int(row["frame"])] = row
        self.i = next((k for k, it in enumerate(self.items) if it["frame"] not in self.labels), 0)
        self.motion_view = False
        self.mouse = (0.0, 0.0)
        self._cache: dict[tuple[int, bool], np.ndarray] = {}

    # --- state ---------------------------------------------------------------------------
    @property
    def item(self) -> dict:
        return self.items[self.i]

    @property
    def done(self) -> bool:
        return all(it["frame"] in self.labels for it in self.items)

    def image(self, motion: bool | None = None) -> np.ndarray:
        motion = self.motion_view if motion is None else motion
        key = (self.item["frame"], motion)
        if key not in self._cache:
            sub = "motion" if motion else "frames"
            ext = "png" if motion else "jpg"
            img = cv2.imread(str(self.dir / sub / f"{self.item['frame']:06d}.{ext}"))
            self._cache = {key: img}
        return self._cache[key]

    def scale(self) -> float:
        h, w = self.image(False).shape[:2]
        return min(1.0, MAX_DISPLAY[0] / w, MAX_DISPLAY[1] / h)

    # --- actions -------------------------------------------------------------------------
    def label(self, status: str, uv: tuple[float, float] | None = None) -> None:
        it = self.item
        u, v = uv if uv else ("", "")
        self.labels[it["frame"]] = {
            "frame": it["frame"], "status": status, "u": u, "v": v, "source": it["source"],
        }  # fmt: skip
        self._save()
        self.i = min(self.i + 1, len(self.items) - 1)

    def on_mouse(self, event, x, y, *_):
        s = self.scale()
        self.mouse = (x / s, y / s)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.label("ball", (round(x / s, 1), round(y / s, 1)))

    def on_key(self, key: int) -> bool:
        """Returns False to quit."""
        k = chr(key).lower() if 0 <= key < 256 else ""
        if key == KEY_ESC:
            return False
        if key == KEY_ENTER or k == "y":
            if self.item["proposal"]:
                self.label("ball", tuple(round(c, 1) for c in self.item["proposal"]))
        elif k == "n":
            self.label("none")
        elif k == "u" or key == KEY_BACKSPACE:
            self.i = max(0, self.i - 1)
        elif k == "s":
            self.i = min(self.i + 1, len(self.items) - 1)
        elif k == "m":
            self.motion_view = not self.motion_view
        return True

    def _save(self) -> None:
        with open(self.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS)
            w.writeheader()
            for f in sorted(self.labels):
                w.writerow(self.labels[f])

    # --- drawing -------------------------------------------------------------------------
    def render(self) -> np.ndarray:
        img = self.image().copy()
        it = self.item
        if it["proposal"]:
            u, v = (round(c) for c in it["proposal"])
            cv2.circle(img, (u, v), 12, (0, 255, 255), 1, cv2.LINE_AA)
        lab = self.labels.get(it["frame"])
        if lab and lab["status"] == "ball":
            cv2.circle(img, (round(float(lab["u"])), round(float(lab["v"]))), 3, (0, 255, 0), -1)
        s = self.scale()
        if s != 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        self._inset(img, self.mouse, (6, 52), "cursor")
        if it["proposal"]:
            self._inset(img, it["proposal"], (img.shape[1] - 6 - 200, 52), "ring")
        self._banner(img, lab)
        return img

    def _inset(self, canvas, centre, at, title, radius=20, zoom=5):
        src = self.image()
        x, y = round(centre[0]), round(centre[1])
        pad = cv2.copyMakeBorder(src, radius, radius, radius, radius, cv2.BORDER_CONSTANT)
        crop = pad[y : y + 2 * radius, x : x + 2 * radius]
        if crop.shape[:2] != (2 * radius, 2 * radius):
            return
        mag = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        c = radius * zoom
        cv2.circle(mag, (c, c), 12 * zoom // 2, (0, 255, 255), 1)
        cv2.putText(mag, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        h, w = mag.shape[:2]
        x0, y0 = at
        if canvas.shape[0] >= y0 + h and canvas.shape[1] >= x0 + w and x0 >= 0:
            canvas[y0 : y0 + h, x0 : x0 + w] = mag

    def _banner(self, img, lab):
        it = self.item
        n_done = sum(1 for x in self.items if x["frame"] in self.labels)
        status = f"labelled: {lab['status']}" if lab else "not labelled"
        top = (f"{self.i + 1}/{len(self.items)}  frame {it['frame']} ({it['t']:.1f}s)  "
               f"[{it['source']}]  {status}   done {n_done}/{len(self.items)}"
               + ("   MOTION VIEW" if self.motion_view else ""))  # fmt: skip
        bottom = ("Y/ENTER ring is on ball   CLICK ball   N no ball   M motion view   "
                  "U back   S skip   ESC save+quit")  # fmt: skip
        cv2.rectangle(img, (0, 0), (img.shape[1], 46), (0, 0, 0), -1)
        for i, text in enumerate((top, bottom)):
            cv2.putText(img, text, (8, 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)  # fmt: skip


def run(pack_dir) -> None:
    tool = BallLabeler(pack_dir)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, tool.on_mouse)
    while True:
        cv2.imshow(WINDOW, tool.render())
        key = cv2.waitKey(20)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key != -1 and not tool.on_key(key & 0xFF):
            break
    cv2.destroyAllWindows()
    n = len(tool.labels)
    print(f"{n}/{len(tool.items)} frames labelled -> {tool.csv}")
    if tool.done:
        print("All done. Send labels.csv back.")
