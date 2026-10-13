"""Checking the coarse shot types (stage 5c) by hand: a review pack, a labelling tool, scoring.

    padelvision shot-review-pack <video> --run runs/match1 --out shot_review   (Kaggle: video)
    padelvision label-shots data/shot_review                                    (local GUI)
    padelvision shots-eval data/shot_review                                      (local)

The pack holds a short clip around every rally shot (SHOT_PAD_S each side) and pack.json with
each shot's hitter box and predicted type / side. The tool loops one clip at a time with the
hitter boxed; press the true type and side, or X when it isn't a shot at all. Labels are
saved after every key to shot_labels.csv, so you can stop and continue. Scoring compares
predicted with labelled types (accuracy per type, confusion) and sides.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SHOT_PAD_S = 0.6
MAX_SHOTS = 120
JPEG_QUALITY = 88
WINDOW = "PadelVision shot labelling"
MAX_DISPLAY = (1600, 900)
TYPE_KEYS = {"s": "serve", "g": "groundstroke", "v": "volley", "o": "overhead", "l": "lob",
             "x": "not_a_shot", "u": "unclear"}  # fmt: skip
SIDE_KEYS = {"f": "forehand", "b": "backhand"}
FIELDS = ["frame", "type", "side"]
KEY_ESC = 27
KEY_LEFT, KEY_RIGHT = 2424832, 2555904  # cv2.waitKeyEx on Windows


# --- pack (Kaggle) ------------------------------------------------------------------------------
def make_shot_pack(video, run_dir, out_dir, max_shots: int = MAX_SHOTS, seed: int = 0) -> Path:
    from padelvision.ball.candidates import read_frames

    run, out = Path(run_dir), Path(out_dir)
    info = json.loads((run / "video.json").read_text())
    fps = float(info["fps"])
    shots = json.loads((run / "rallies.json").read_text()).get("stats", {}).get("shots", [])
    if len(shots) > max_shots:  # a sample spread over the match
        idx = np.sort(np.random.default_rng(seed).choice(len(shots), max_shots, replace=False))
        shots = [shots[i] for i in idx]
    players = pd.read_parquet(run / "players.parquet")
    pad = round(SHOT_PAD_S * fps)
    items, wanted = [], set()
    for s in shots:
        f = int(s["frame"])
        b = players[(players.player == s["player"]) & players.frame.between(f - 2, f + 2)]
        box = None if b.empty else [round(float(b.iloc[0][k]), 1) for k in ("x1", "y1", "x2", "y2")]
        items.append({
            "frame": f, "start": f - pad, "end": f + pad, "player": s["player"], "team": s["team"],
            "box": box, "type": s.get("type"), "side": s.get("side"),
        })  # fmt: skip
        wanted.update(range(f - pad, f + pad + 1))
    (out / "frames").mkdir(parents=True, exist_ok=True)
    if wanted:
        for f, img in read_frames(video, max(wanted) + 1):
            if f in wanted:
                cv2.imwrite(str(out / "frames" / f"{f:06d}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])  # fmt: skip
    (out / "pack.json").write_text(
        json.dumps({"video": str(video), "fps": fps, "shots": items}, indent=1)
    )
    print(f"shot review pack: {len(items)} shots, {len(wanted)} frames")
    return out


# --- labelling (local) --------------------------------------------------------------------------
class ShotLabeler:
    def __init__(self, pack_dir: str | Path):
        self.dir = Path(pack_dir)
        pack = json.loads((self.dir / "pack.json").read_text())
        self.fps = float(pack["fps"])
        self.shots: list[dict] = pack["shots"]
        self.csv = self.dir / "shot_labels.csv"
        self.labels: dict[int, dict] = {}
        if self.csv.exists():
            with open(self.csv, newline="") as fh:
                self.labels = {int(r["frame"]): r for r in csv.DictReader(fh)}
        self.i = next((k for k, s in enumerate(self.shots) if s["frame"] not in self.labels), 0)
        self.f = self.shots[self.i]["start"] if self.shots else 0

    @property
    def shot(self) -> dict:
        return self.shots[self.i]

    def key(self, k: int) -> str | None:
        """Handle a key; returns "quit" to stop."""
        c = chr(k).lower() if 0 <= k < 256 else ""
        if k == KEY_ESC:
            return "quit"
        if c in TYPE_KEYS or c in SIDE_KEYS:
            row = self.labels.setdefault(
                self.shot["frame"], {"frame": self.shot["frame"], "type": "", "side": ""}
            )
            if c in TYPE_KEYS:
                row["type"] = TYPE_KEYS[c]
            else:
                row["side"] = SIDE_KEYS[c] if row.get("side") != SIDE_KEYS[c] else ""
            self.save()
        elif c in ("n", " ") or k == KEY_RIGHT:
            self.go(self.i + 1)
        elif c == "p" or k == KEY_LEFT:
            self.go(self.i - 1)
        return None

    def go(self, i: int) -> None:
        self.i = int(np.clip(i, 0, len(self.shots) - 1))
        self.f = self.shot["start"]

    def save(self) -> None:
        with open(self.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for f in sorted(self.labels):
                w.writerow({k: self.labels[f].get(k, "") for k in FIELDS})

    def render(self) -> np.ndarray:
        s = self.shot
        img = cv2.imread(str(self.dir / "frames" / f"{self.f:06d}.jpg"))
        if img is None:
            img = np.zeros((720, 1280, 3), np.uint8)
        if s["box"]:
            x1, y1, x2, y2 = map(round, s["box"])
            color = (0, 0, 255) if self.f == s["frame"] else (0, 255, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        lab = self.labels.get(s["frame"], {})
        lines = [
            f"shot {self.i + 1}/{len(self.shots)}  player {s['player']} ({s['team']})"
            f"  {'CONTACT' if self.f == s['frame'] else ''}",
            f"label: {lab.get('type') or '-'} {lab.get('side') or ''}",
            "S serve  G groundstroke  V volley  O overhead  L lob  X not a shot  U unclear",
            "F forehand  B backhand (toggle)   N / -> next   P / <- back   ESC quit",
        ]
        for k, t in enumerate(lines):
            cv2.putText(
                img, t, (12, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA
            )
            cv2.putText(
                img,
                t,
                (12, 28 + 26 * k),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        h, w = img.shape[:2]
        scale = min(1.0, MAX_DISPLAY[0] / w, MAX_DISPLAY[1] / h)
        return cv2.resize(img, (round(w * scale), round(h * scale))) if scale < 1 else img

    def advance_frame(self) -> None:
        """Loop the clip, pausing briefly on the contact frame."""
        self.f = self.f + 1 if self.f < self.shot["end"] else self.shot["start"]


def run_labeler(pack_dir: str | Path) -> None:
    tool = ShotLabeler(pack_dir)
    if not tool.shots:
        print("No shots in this pack.")
        return
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    delay = max(1, round(1000 / tool.fps * 2))  # half speed
    try:
        while True:
            cv2.imshow(WINDOW, tool.render())
            k = cv2.waitKeyEx(delay * (6 if tool.f == tool.shot["frame"] else 1))
            if k != -1 and tool.key(k) == "quit":
                break
            tool.advance_frame()
    finally:
        cv2.destroyAllWindows()
    print(f"{len(tool.labels)} of {len(tool.shots)} shots labelled -> {tool.csv}")


# --- scoring ------------------------------------------------------------------------------------
def score_shots(pack_dir: str | Path) -> dict:
    """Predicted (pack.json) vs labelled (shot_labels.csv) types and sides."""
    d = Path(pack_dir)
    pred = {s["frame"]: s for s in json.loads((d / "pack.json").read_text())["shots"]}
    with open(d / "shot_labels.csv", newline="") as fh:
        labels = [r for r in csv.DictReader(fh) if r["type"]]
    shots = [r for r in labels if r["type"] not in ("not_a_shot", "unclear")]
    confusion = Counter((r["type"], pred[int(r["frame"])].get("type") or "unknown") for r in shots)
    per_type = {}
    for t in ("serve", "groundstroke", "volley", "overhead", "lob"):
        truth = [r for r in shots if r["type"] == t]
        called = [r for r in shots if pred[int(r["frame"])].get("type") == t]
        right = sum(pred[int(r["frame"])].get("type") == t for r in truth)
        per_type[t] = {"labelled": len(truth), "predicted": len(called),
                       "recall": round(right / len(truth), 2) if truth else None,
                       "precision": round(right / len(called), 2) if called else None}  # fmt: skip
    typed = [r for r in shots if (pred[int(r["frame"])].get("type") or "unknown") != "unknown"]
    sided = [r for r in shots if r["side"] and pred[int(r["frame"])].get("side")]
    return {
        "labelled": len(labels),
        "not_a_shot": sum(r["type"] == "not_a_shot" for r in labels),
        "typed_share": round(len(typed) / len(shots), 2) if shots else None,
        "type_accuracy": round(
            sum(pred[int(r["frame"])]["type"] == r["type"] for r in typed) / len(typed), 2
        )
        if typed
        else None,
        "side_accuracy": round(
            sum(pred[int(r["frame"])]["side"] == r["side"] for r in sided) / len(sided), 2
        )
        if sided
        else None,
        "per_type": per_type,
        "confusion_truth_vs_predicted": {
            f"{a} -> {b}": n for (a, b), n in sorted(confusion.items())
        },
    }
