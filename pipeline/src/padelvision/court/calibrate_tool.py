"""Interactive court calibration (runs locally, needs only one frame image).

Step 1 - lines: click 3+ points along each straight real-world line (court lines, glass
         frames, vertical posts) to measure lens distortion. Long lines near the image
         edges matter most. Not the net: it sags.
Step 2 - points: click each named court keypoint, or skip it if it's not visible.
Step 3 - review: court lines are re-projected on the image; save if they match.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from padelvision.court.calibration import CourtCalibration
from padelvision.court.draw import MiniCourt, draw_court_overlay
from padelvision.court.geometry import CLICK_ORDER, COURT_KEYPOINTS

WINDOW = "PadelVision court calibration"
MAX_DISPLAY = (1600, 900)
KEY_ENTER, KEY_ESC = 13, 27

HELP = {
    "lines": (
        "STEP 1/3  LINES  - click 3+ points along a straight line (court line, glass frame, post)",
        "ENTER = finish line   U = undo   N = done with lines   ESC = quit",
    ),
    "points": (
        "STEP 2/3  POINTS - click: {name}   ({i}/{n})",
        "S = not visible, skip   U = undo   ESC = quit",
    ),
    "review": (
        "STEP 3/3  REVIEW - mean error {err}   lens line error {rms}",
        "ENTER = save   P = redo points   L = redo lines   ESC = quit",
    ),
}


class CalibrationTool:
    def __init__(self, image: np.ndarray):
        self.image = image
        h, w = image.shape[:2]
        self.size = (w, h)
        self.scale = min(1.0, MAX_DISPLAY[0] / w, MAX_DISPLAY[1] / h)
        self.mode = "lines"
        self.lines: list[list[tuple[float, float]]] = []
        self.current: list[tuple[float, float]] = []
        self.points: dict[str, tuple[float, float]] = {}
        self.idx = 0
        self.history: list[str | None] = []  # keypoint clicked (or None if skipped) per step
        self.cal: CourtCalibration | None = None
        self.error: str | None = None
        self.mouse = (0, 0)
        self.minicourt = MiniCourt(height_px=220)

    # --- events -------------------------------------------------------------------------
    def on_mouse(self, event, x, y, *_):
        px, py = x / self.scale, y / self.scale
        self.mouse = (px, py)
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.mode == "lines":
            self.current.append((px, py))
        elif self.mode == "points":
            name = CLICK_ORDER[self.idx]
            self.points[name] = (px, py)
            self.history.append(name)
            self._advance()

    def on_key(self, key: int) -> str | None:
        """Returns 'save' or 'quit' to end the loop."""
        k = chr(key).lower() if 0 <= key < 256 else ""
        if key == KEY_ESC:
            return "quit"
        if self.mode == "lines":
            if key == KEY_ENTER:
                self._finish_line()
            elif k == "u":
                if self.current:
                    self.current.pop()
                elif self.lines:
                    self.current = self.lines.pop()
            elif k == "n":
                self._finish_line()
                self.mode = "points"
        elif self.mode == "points":
            if k == "s":
                self.history.append(None)
                self._advance()
            elif k == "u" and self.history:
                last = self.history.pop()
                if last:
                    self.points.pop(last, None)
                self.idx -= 1
        elif self.mode == "review":
            if key == KEY_ENTER and self.cal is not None:
                return "save"
            if k == "p":
                self.points, self.history, self.idx = {}, [], 0
                self.mode = "points"
            elif k == "l":
                self.lines, self.current = [], []
                self.mode = "lines"
        return None

    def _finish_line(self):
        if len(self.current) >= 3:
            self.lines.append(self.current)
        self.current = []

    def _advance(self):
        self.idx += 1
        if self.idx >= len(CLICK_ORDER):
            self._fit()
            self.mode = "review"

    def _fit(self):
        self.cal, self.error = None, None
        try:
            self.cal = CourtCalibration.from_points(self.points, self.size, lines=self.lines)
        except ValueError as e:
            self.error = str(e)

    # --- drawing ------------------------------------------------------------------------
    def render(self) -> np.ndarray:
        if self.mode == "review" and self.cal is not None:
            img = draw_court_overlay(self.image, self.cal)
        else:
            img = self.image.copy()
        for line in [*self.lines, self.current]:
            pts = np.array(line).round().astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts], False, (255, 0, 255), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, tuple(p), 3, (255, 0, 255), -1, cv2.LINE_AA)
        for name, (x, y) in self.points.items():
            cv2.circle(img, (round(x), round(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                img,
                name,
                (round(x) + 6, round(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        if self.scale != 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self._draw_magnifier(img)
        if self.mode == "points":
            self._draw_target(img, CLICK_ORDER[self.idx])
        self._draw_banner(img)
        return img

    def _banner_text(self) -> tuple[str, str]:
        top, bottom = HELP[self.mode]
        if self.mode == "points":
            top = top.format(name=CLICK_ORDER[self.idx].upper(), i=self.idx + 1, n=len(CLICK_ORDER))
        elif self.mode == "review":
            if self.cal is None:
                return f"Calibration failed: {self.error}", "P = redo points   L = redo lines"
            rms = f"{self.cal.line_rms_px:.1f}px" if self.cal.line_rms_px is not None else "n/a"
            top = top.format(err=f"{self.cal.reprojection_error_m() * 100:.0f}cm", rms=rms)
        elif self.mode == "lines":
            top += f"   [{len(self.lines)} lines]"
        return top, bottom

    def _draw_banner(self, img):
        top, bottom = self._banner_text()
        cv2.rectangle(img, (0, 0), (img.shape[1], 46), (0, 0, 0), -1)
        for i, text in enumerate((top, bottom)):
            cv2.putText(
                img,
                text,
                (8, 18 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_magnifier(self, img, radius: int = 24, zoom: int = 5):
        x, y = round(self.mouse[0]), round(self.mouse[1])
        pad = cv2.copyMakeBorder(self.image, radius, radius, radius, radius, cv2.BORDER_CONSTANT)
        crop = pad[y : y + 2 * radius, x : x + 2 * radius]
        if crop.shape[:2] != (2 * radius, 2 * radius):
            return
        mag = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        c = radius * zoom
        cv2.line(mag, (c, 0), (c, 2 * c), (0, 255, 0), 1)
        cv2.line(mag, (0, c), (2 * c, c), (0, 255, 0), 1)
        h = mag.shape[0]
        if img.shape[0] > h + 56 and img.shape[1] > h + 10:
            img[52 : 52 + h, 6 : 6 + h] = mag

    def _draw_target(self, img, name: str):
        mini = self.minicourt.canvas()
        cv2.circle(mini, self.minicourt.to_px(COURT_KEYPOINTS[name]), 6, (0, 0, 255), -1)
        cv2.putText(
            mini,
            "camera",
            (mini.shape[1] // 2 - 22, mini.shape[0] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        h, w = mini.shape[:2]
        if img.shape[0] > h + 56 and img.shape[1] > w + 10:
            img[52 : 52 + h, img.shape[1] - w - 6 : img.shape[1] - 6] = mini


def run(image_path: str | Path, out_path: str | Path) -> CourtCalibration | None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    tool = CalibrationTool(image)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, tool.on_mouse)
    result = None
    while True:
        cv2.imshow(WINDOW, tool.render())
        key = cv2.waitKey(20)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            result = "quit"
        elif key != -1:
            result = tool.on_key(key & 0xFF)
        if result:
            break
    cv2.destroyAllWindows()

    if result != "save" or tool.cal is None:
        print("Calibration not saved.")
        return None
    out_path = Path(out_path)
    tool.cal.save(out_path)
    preview = out_path.with_name(out_path.stem + "_preview.png")
    cv2.imwrite(str(preview), draw_court_overlay(image, tool.cal))
    print(f"Saved {out_path} and {preview}")
    print(f"Mean keypoint error: {tool.cal.reprojection_error_m() * 100:.1f} cm")
    for name, err in sorted(tool.cal.point_errors_m().items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {err * 100:6.1f} cm")
    return tool.cal
