"""Score detected events (events.parquet) against hand labels from `label-events`.

Labels: CSV with frame, kind ("hit" | "bounce" | "wall"), u, v, player (hits; may be empty).
Walls are not scored: the pipeline doesn't classify them (its unclassified sharp turns are "turn").
Windows: JSON list of {"start", "end"} frame ranges that were labelled exhaustively. Only
predictions inside these windows are scored, and a labelled window with no mark at a frame
means there was no event there.

A prediction matches an unmatched label of the same kind within TOL_S (closest first).
"handling" and "turn" predictions are not hits or bounces, so they never count, right or wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TOL_S = 0.2
KINDS = ("hit", "bounce")


def load_labels(csv_path: str | Path, windows_path: str | Path | None = None):
    csv_path = Path(csv_path)
    if windows_path is None:
        windows_path = csv_path.with_name(csv_path.stem + "_windows.json")
    labels = pd.read_csv(csv_path)
    windows = json.loads(Path(windows_path).read_text())
    return labels, windows


def score_events(
    pred: pd.DataFrame,
    labels: pd.DataFrame,
    windows: list[dict],
    fps: float,
    kinds: tuple[str, ...] = KINDS,
    tol_s: float = TOL_S,
) -> dict:
    if not windows:
        return {"windows": 0, "labelled_s": 0.0}

    def inside(df: pd.DataFrame) -> pd.DataFrame:
        f = df.frame.to_numpy()
        keep = np.zeros(len(f), bool)
        for w in windows:
            keep |= (f >= w["start"]) & (f <= w["end"])
        return df[keep]

    pred, labels = inside(pred), inside(labels)
    tol = tol_s * fps
    out: dict = {
        "windows": len(windows),
        "labelled_s": round(sum(w["end"] - w["start"] + 1 for w in windows) / fps, 1),
        "tolerance_s": tol_s,
    }
    for kind in kinds:
        p = pred[pred.kind == kind]
        lab = labels[labels.kind == kind]
        pairs = _match(p.frame.to_numpy(), lab.frame.to_numpy(), tol)
        tp = len(pairs)
        res = {"labelled": int(len(lab)), "predicted": int(len(p)), "tp": tp,
               "fp": int(len(p) - tp), "fn": int(len(lab) - tp)}  # fmt: skip
        res["precision"] = round(tp / len(p), 3) if len(p) else None
        res["recall"] = round(tp / len(lab), 3) if len(lab) else None
        if tp:
            res["f1"] = round(2 * tp / (len(p) + len(lab)), 3)
        else:
            res["f1"] = 0.0 if len(p) + len(lab) else None
        if kind == "hit" and tp:
            same = [_same_player(p.player.iloc[i], lab.player.iloc[j]) for i, j in pairs]
            known = [s for s in same if s is not None]
            res["player_accuracy"] = round(float(np.mean(known)), 3) if known else None
        if tp and {"u", "v"} <= set(lab.columns):
            d = [np.hypot(p.u.iloc[i] - lab.u.iloc[j], p.v.iloc[i] - lab.v.iloc[j])
                 for i, j in pairs]  # fmt: skip
            d = [x for x in d if np.isfinite(x)]
            res["median_px_error"] = round(float(np.median(d)), 1) if d else None
        out[kind] = res
    return out


def _match(pred_frames: np.ndarray, label_frames: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Greedy one-to-one matching, closest pairs first: [(pred index, label index)]."""
    cand = [
        (abs(int(pf) - int(lf)), i, j)
        for i, pf in enumerate(pred_frames)
        for j, lf in enumerate(label_frames)
        if abs(int(pf) - int(lf)) <= tol
    ]
    used_p, used_l, pairs = set(), set(), []
    for _, i, j in sorted(cand):
        if i not in used_p and j not in used_l:
            used_p.add(i)
            used_l.add(j)
            pairs.append((i, j))
    return pairs


def _same_player(pred_player, label_player) -> bool | None:
    if pd.isna(label_player) or label_player == "":
        return None
    if pd.isna(pred_player):
        return False
    return int(pred_player) == int(label_player)
