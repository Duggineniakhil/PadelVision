"""Local hit/bounce labelling tool for an event review pack (made on Kaggle by
`event-review-pack`).

    padelvision label-events data/event_review

Step through each window and mark EVERY hit and floor bounce:
- hit:    the frame where the racket meets the ball on a real shot (serves too). Not ball
          handling: bouncing the ball between points or tapping it to a partner.
- bounce: the frame where the ball touches the floor (any floor bounce, in or out).
- wall:   optional; glass/fence rebounds (kept for later, not scored yet).
Click the ball first (optional: without a click the tracked ball position is used), then
press H / B / G. A hit is given to the nearest player box; press 1-4 to correct it.
When a window is complete, press V: only finished windows are used for scoring, and in a
finished window "no mark" means "no event". Everything is saved after every change to
labels.csv and labels_windows.json in the pack, so you can quit (ESC) and resume any time.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

WINDOW = "PadelVision event labelling"
MAX_DISPLAY = (1600, 900)
FIELDS = ["frame", "kind", "u", "v", "player"]
KIND_KEYS = {"h": "hit", "b": "bounce", "g": "wall"}
KIND_COLORS = {"hit": (0, 0, 255), "bounce": (0, 200, 0), "wall": (255, 120, 0)}
TRAIL = 10  # tracked-ball frames shown on each side
NEAR_EVENTS = 15  # labelled events within this many frames are drawn
KEY_ESC, KEY_SPACE = 27, 32
KEY_LEFT, KEY_RIGHT = 2424832, 2555904  # cv2.waitKeyEx on Windows


class EventLabeler:
    def __init__(self, pack_dir: str | Path):
        self.dir = Path(pack_dir)
        pack = json.loads((self.dir / "pack.json").read_text())
        self.fps = float(pack["fps"])
        self.windows: list[dict] = pack["windows"]
        self.csv = self.dir / "labels.csv"
        self.windows_json = self.dir / "labels_windows.json"
        self.labels: dict[int, dict] = {}
        if self.csv.exists():
            with open(self.csv, newline="") as fh:
                for row in csv.DictReader(fh):
                    self.labels[int(row["frame"])] = row
        self.done: set[int] = set()
        if self.windows_json.exists():
            self.done = {w["id"] for w in json.loads(self.windows_json.read_text())}
        run = self.dir / "run"
        self.players = _by_frame(run / "players.parquet")
        ball = run / "ball.parquet"
        self.ball = pd.read_parquet(ball).set_index("frame") if ball.exists() else None
        self.wi = next((i for i, w in enumerate(self.windows) if w["id"] not in self.done), 0)
        self.frame = self.windows[self.wi]["start"] if self.windows else 0
        self.click: tuple[float, float] | None = None
        self.mouse = (0.0, 0.0)
        self.playing = False
        self.show_overlay = True
        self.motion_view = False

    # --- state ---------------------------------------------------------------------------
    @property
    def win(self) -> dict:
        return self.windows[self.wi]

    def step(self, n: int) -> None:
        self.frame = int(np.clip(self.frame + n, self.win["start"], self.win["end"]))
        self.click = None

    def goto_window(self, wi: int) -> None:
        self.wi = int(np.clip(wi, 0, len(self.windows) - 1))
        self.frame = self.win["start"]
        self.click = None
        self.playing = False

    def ball_at(self, f: int) -> tuple[float, float] | None:
        if self.ball is None or f not in self.ball.index:
            return None
        row = self.ball.loc[f]
        return float(row.u), float(row.v)

    def mark(self, kind: str) -> None:
        pos = self.click or self.ball_at(self.frame)
        player = ""
        if kind == "hit" and pos is not None:
            player = _nearest_player(self.players.get(self.frame), pos)
        u, v = (round(pos[0], 1), round(pos[1], 1)) if pos else ("", "")
        self.labels[self.frame] = {"frame": self.frame, "kind": kind, "u": u, "v": v,
                                   "player": player}  # fmt: skip
        self.click = None
        self._save()

    def set_player(self, player: int) -> None:
        lab = self.labels.get(self.frame)
        if lab and lab["kind"] == "hit":
            lab["player"] = player
            self._save()

    def delete(self) -> None:
        if self.labels.pop(self.frame, None) is not None:
            self._save()

    def finish_window(self) -> None:
        self.done.add(self.win["id"])
        self._save()
        nxt = next((i for i, w in enumerate(self.windows) if w["id"] not in self.done), None)
        if nxt is not None:
            self.goto_window(nxt)

    def on_key(self, key: int) -> bool:
        """Returns False to quit."""
        if key == KEY_ESC:
            return False
        if key == KEY_RIGHT:
            self.step(1)
        elif key == KEY_LEFT:
            self.step(-1)
        k = chr(key & 0xFF).lower() if 0 <= key < 256 else ""
        if k == "d":
            self.step(1)
        elif k == "a":
            self.step(-1)
        elif k == "e":
            self.step(10)
        elif k == "q":
            self.step(-10)
        elif key == KEY_SPACE:
            self.playing = not self.playing
        elif k in KIND_KEYS:
            self.mark(KIND_KEYS[k])
        elif k in "1234" and k:
            self.set_player(int(k))
        elif k == "x":
            self.delete()
        elif k == "v":
            self.finish_window()
        elif k == "]":
            self.goto_window(self.wi + 1)
        elif k == "[":
            self.goto_window(self.wi - 1)
        elif k == "t":
            self.show_overlay = not self.show_overlay
        elif k == "m":
            self.motion_view = not self.motion_view
        return True

    def on_mouse(self, event, x, y, *_):
        s = self.scale()
        self.mouse = (x / s, y / s)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x / s, y / s)

    def tick(self) -> None:
        if self.playing:
            if self.frame >= self.win["end"]:
                self.playing = False
            else:
                self.step(1)

    def _save(self) -> None:
        with open(self.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS)
            w.writeheader()
            for f in sorted(self.labels):
                w.writerow(self.labels[f])
        done = [{"id": w["id"], "start": w["start"], "end": w["end"]}
                for w in self.windows if w["id"] in self.done]  # fmt: skip
        self.windows_json.write_text(json.dumps(done, indent=1))

    # --- drawing -------------------------------------------------------------------------
    def image(self, f: int | None = None) -> np.ndarray:
        f = self.frame if f is None else f
        img = cv2.imread(str(self.dir / "frames" / f"{f:06d}.jpg"))
        return img if img is not None else np.zeros((720, 1280, 3), np.uint8)

    def scale(self) -> float:
        h, w = self.image().shape[:2]
        return min(1.0, MAX_DISPLAY[0] / w, MAX_DISPLAY[1] / h)

    def render(self) -> np.ndarray:
        img = self._motion() if self.motion_view else self.image().copy()
        if self.show_overlay:
            self._overlay(img)
        for f, lab in self.labels.items():
            if abs(f - self.frame) <= NEAR_EVENTS and lab["u"] != "":
                c = KIND_COLORS[lab["kind"]]
                p = (round(float(lab["u"])), round(float(lab["v"])))
                cv2.circle(img, p, 14 if f == self.frame else 8, c, 2, cv2.LINE_AA)
                if f == self.frame:
                    tag = lab["kind"] + (f" P{lab['player']}" if lab["player"] != "" else "")
                    cv2.putText(img, tag, (p[0] + 16, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        if self.click:
            cx, cy = round(self.click[0]), round(self.click[1])
            cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
        src = img
        s = self.scale()
        if s != 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        _inset(img, src, self.mouse, (6, 52))
        self._banner(img)
        return img

    def _overlay(self, img: np.ndarray) -> None:
        boxes = self.players.get(self.frame)
        if boxes is not None:
            for b in boxes.itertuples():
                p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
                cv2.rectangle(img, p1, p2, (255, 200, 0), 1)
                cv2.putText(img, f"P{int(b.player)}", (p1[0], p1[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)  # fmt: skip
        pts = [self.ball_at(f) for f in range(self.frame - TRAIL, self.frame + TRAIL + 1)]
        pts = np.array([p for p in pts if p is not None]).round().astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)

    def _motion(self) -> np.ndarray:
        f0, f1 = max(self.frame - 1, self.win["start"]), min(self.frame + 1, self.win["end"])
        g = [cv2.cvtColor(self.image(f), cv2.COLOR_BGR2GRAY) for f in (f0, self.frame, f1)]
        m = np.minimum(cv2.absdiff(g[1], g[0]), cv2.absdiff(g[2], g[1]))
        return cv2.cvtColor(np.clip(m.astype(int) * 4, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def _banner(self, img: np.ndarray) -> None:
        w = self.win
        n_ev = sum(w["start"] <= f <= w["end"] for f in self.labels)
        state = "DONE" if w["id"] in self.done else "open"
        flags = ("  PLAYING" if self.playing else "") + ("  MOTION" if self.motion_view else "")
        top = (f"window {self.wi + 1}/{len(self.windows)} [{w['source']}, {state}]  "
               f"frame {self.frame} ({self.frame / self.fps:.2f}s)  "
               f"{self.frame - w['start'] + 1}/{w['end'] - w['start'] + 1}  events here {n_ev}  "
               f"windows done {len(self.done)}/{len(self.windows)}{flags}")  # fmt: skip
        bottom = ("A/D or arrows +-1  Q/E +-10  SPACE play  CLICK ball  H hit  B bounce  "
                  "G wall  1-4 player  X delete  V window done  [ ] window  T overlay  "
                  "M motion  ESC quit")  # fmt: skip
        cv2.rectangle(img, (0, 0), (img.shape[1], 46), (0, 0, 0), -1)
        for i, text in enumerate((top, bottom)):
            cv2.putText(img, text, (8, 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)  # fmt: skip


def _by_frame(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {f: g for f, g in df.groupby("frame")}


def _nearest_player(boxes: pd.DataFrame | None, pos: tuple[float, float]):
    """Player whose box is closest to `pos` (0 inside the box), or "" without boxes."""
    if boxes is None or boxes.empty:
        return ""
    u, v = pos
    dx = np.maximum.reduce([boxes.x1 - u, u - boxes.x2, np.zeros(len(boxes))])
    dy = np.maximum.reduce([boxes.y1 - v, v - boxes.y2, np.zeros(len(boxes))])
    return int(boxes.player.iloc[int(np.argmin(np.hypot(dx, dy)))])


def _inset(canvas, src, centre, at, radius=20, zoom=5) -> None:
    x, y = round(centre[0]), round(centre[1])
    pad = cv2.copyMakeBorder(src, radius, radius, radius, radius, cv2.BORDER_CONSTANT)
    crop = pad[y : y + 2 * radius, x : x + 2 * radius]
    if crop.shape[:2] != (2 * radius, 2 * radius):
        return
    mag = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    h, w = mag.shape[:2]
    x0, y0 = at
    if canvas.shape[0] >= y0 + h and canvas.shape[1] >= x0 + w:
        canvas[y0 : y0 + h, x0 : x0 + w] = mag


def run(pack_dir) -> None:
    tool = EventLabeler(pack_dir)
    if not tool.windows:
        print("The pack has no windows.")
        return
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, tool.on_mouse)
    while True:
        cv2.imshow(WINDOW, tool.render())
        key = cv2.waitKeyEx(round(1000 / tool.fps) if tool.playing else 20)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key != -1 and not tool.on_key(key):
            break
        tool.tick()
    cv2.destroyAllWindows()
    n = len(tool.labels)
    print(f"{n} events, {len(tool.done)}/{len(tool.windows)} windows done -> {tool.csv}")
    if len(tool.done) == len(tool.windows):
        print("All windows done. Send labels.csv and labels_windows.json back.")
