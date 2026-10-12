"""Automatic court calibration from a clean background image (players removed).

A padel court is a known shape, and the target camera (low, behind one baseline, wide-angle)
varies mostly in where exactly it stands. So the court is found by fitting that shape:

1. Evidence (`evidence`): painted lines (thin strips brighter than their surroundings, not
   coloured) and the playing floor (the dominant colour at the bottom middle of the image);
   the floor's outline is where it meets the walls (the side lines and the far baseline).
2. Rough placement (`search`): starting from known calibrations of the same kind of camera
   (`ml/calibrations/*.json`), the court is moved by four image points (8 degrees of freedom,
   the lens kept) to put the projected court lines on the evidence: a chamfer score (mean
   distance from projected line points to the nearest evidence pixel, truncated), minimised
   coarse to fine.
3. Precise fit (`refine`): each visible court line is snapped onto the image along its rough
   projection (`snap.py`) and the lens and court are fitted jointly (`CourtCalibration`), twice.
4. Quality gate: fit, lines on both sides of the net, near-half ground error. A failed gate is
   reported, never used silently: the caller falls back to manual calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from padelvision.court.calibration import CourtCalibration
from padelvision.court.geometry import COURT_KEYPOINTS, NAMED_LINES
from padelvision.court.lens import LensModel

# Painted lines: brighter than a TOPHAT_PX neighbourhood by PAINT_MIN, and not coloured.
TOPHAT_PX = 25
PAINT_MIN = 25
PAINT_MAX_SAT = 80
# Floor: colour distance (Lab a/b) to the floor colour sampled at the bottom middle of the image.
FLOOR_SAMPLE = (0.35, 0.65, 0.7, 0.95)  # x0, x1, y0, y1 as fractions of the image
FLOOR_MAX_CHROMA = 9.0

# Lines each kind of evidence explains (the near baseline is usually out of view).
PAINT_LINES = ("near_service_line", "far_service_line", "center_line")
EDGE_LINES = ("left_side", "right_side", "far_baseline")
SNAP_MODE = {**{n: "paint" for n in PAINT_LINES}, **{n: "edge" for n in EDGE_LINES},
             "near_baseline": "edge", "net_base": "net"}  # fmt: skip
CHAMFER_MAX_PX = 12.0
NET_EDGE_PX = 6  # compare the image this far above and below
NET_EDGE_MIN = 18.0  # lightness difference (0-255) for a net-base edge
MAX_NET_MISS_PX = 4.0  # the final fit's net line must lie this close to a net-base edge
EDGE_FINE_PX = 3.0
ANCHORS = ("near_service_left", "near_service_right", "far_service_left", "far_service_right")

MIN_LINES = 4
MIN_LINE_POINTS = 8
MAX_RMS_PX = 2.5
MAX_NEAR_ERROR_M = 0.15


@dataclass
class Evidence:
    paint: np.ndarray  # bool (h, w)
    floor: np.ndarray  # bool (h, w)
    paint_dist: np.ndarray  # px to the nearest painted pixel
    edge_dist: np.ndarray  # px to the nearest floor outline pixel
    net_dist: np.ndarray  # px to the nearest "dark above, floor below" edge (the net's base)
    net_ridge: np.ndarray  # BGR image of that edge strength, to snap the net base on


@dataclass
class AutoResult:
    calibration: CourtCalibration | None
    ok: bool
    reasons: list[str] = field(default_factory=list)
    template: str | None = None
    rough_score_px: float | None = None
    lines: dict[str, int] = field(default_factory=dict)  # snapped points per line
    accuracy: dict | None = None


def evidence(image: np.ndarray) -> Evidence:
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    grey = lab[:, :, 0]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_PX, TOPHAT_PX))
    tophat = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, k)
    paint = (tophat >= PAINT_MIN) & (hsv[:, :, 1] <= PAINT_MAX_SAT)

    x0, x1, y0, y1 = FLOOR_SAMPLE
    sample = lab[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)].reshape(-1, 3)
    sample = sample[
        ~paint[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)].reshape(-1)
    ]
    floor_col = np.median(sample.astype(float), axis=0)
    # Colour (a, b) only: glare and the darker far end change the floor's lightness, not its hue.
    chroma = np.linalg.norm(lab[:, :, 1:].astype(float) - floor_col[1:], axis=2)
    floor = (chroma <= FLOOR_MAX_CHROMA) | paint
    floor = cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    floor = cv2.morphologyEx(floor, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(floor)
    if n > 1:  # the floor is the biggest region touching the sample area
        cx, cy = round((x0 + x1) / 2 * w), round((y0 + y1) / 2 * h)
        lab_id = (
            labels[cy, cx] if labels[cy, cx] else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        )
        floor = (labels == lab_id).astype(np.uint8)
    # Only the outer outline counts: glare and reflections make holes inside the floor.
    contours, _ = cv2.findContours(floor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(floor)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    outline_img = np.zeros_like(floor)
    cv2.drawContours(outline_img, contours, -1, 1, thickness=2)
    outline = outline_img.astype(bool)
    outline[-3:, :] = False  # the image border is not a wall
    outline[:, :3] = outline[:, -3:] = False
    floor = filled
    # Painted court lines lie on the floor: drop bright clutter elsewhere (lights, banners).
    near_floor = cv2.dilate(floor, np.ones((15, 15), np.uint8)).astype(bool)
    paint &= near_floor

    def dist_to(mask: np.ndarray) -> np.ndarray:
        return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)

    paint_dist = dist_to(paint)
    # The net's base: the dark mesh above, the floor below. The net's white top band looks like
    # a painted line, so without this the far lines can all slip by one (band -> far service
    # line -> far baseline) and still fit well.
    blur = cv2.GaussianBlur(grey.astype(np.float32), (5, 5), 0)
    darker_above = np.roll(blur, -NET_EDGE_PX, axis=0) - np.roll(blur, NET_EDGE_PX, axis=0)
    near = cv2.dilate(floor, np.ones((25, 25), np.uint8)).astype(bool)
    usable = near & (paint_dist > 2 * NET_EDGE_PX)
    net_base = (darker_above >= NET_EDGE_MIN) & usable
    net_base[:NET_EDGE_PX] = net_base[-NET_EDGE_PX:] = False
    # A sharp version to snap on (the 2 x NET_EDGE_PX comparison makes a band too wide to snap).
    sharp = np.roll(blur, -2, axis=0) - np.roll(blur, 2, axis=0)
    ridge = np.clip(np.where(usable, 2 * sharp, 0), 0, 255).astype(np.uint8)
    return Evidence(paint, floor.astype(bool), paint_dist, dist_to(outline), dist_to(net_base),
                    np.repeat(ridge[:, :, None], 3, axis=2))  # fmt: skip


# --- rough placement ---------------------------------------------------------------------------
def _samples(names: tuple[str, ...], n: int = 40) -> np.ndarray:
    """Court points (metres) along the named lines, in the parts a low camera usually sees."""
    pts = []
    for name in names:
        a, b, c = NAMED_LINES[name]  # a x + b y + c = 0
        if abs(a) > 0.5:  # along y (side lines, centre line)
            x = -c / a
            ys = np.linspace(-6.95, 6.95, n) if name == "center_line" else np.linspace(-6.95, 10, n)
            pts.append(np.c_[np.full(n, x), ys])
        else:  # across (service lines, baselines)
            pts.append(np.c_[np.linspace(-5, 5, n), np.full(n, -c / b)])
    return np.concatenate(pts)


PAINT_SAMPLES = _samples(PAINT_LINES)
EDGE_SAMPLES = _samples(EDGE_LINES)
NET_SAMPLES = np.c_[np.linspace(-4.5, 4.5, 40), np.zeros(40)]
# Floor check: points inside the court must be floor, points just outside the walls must not.
_gx, _gy = np.meshgrid(np.linspace(-4.5, 4.5, 10), np.linspace(-6.5, 9.5, 17))
INSIDE_SAMPLES = np.c_[_gx.ravel(), _gy.ravel()]
_oy = np.linspace(-6.0, 9.5, 16)
OUTSIDE_SAMPLES = np.r_[np.c_[np.full(16, -5.8), _oy], np.c_[np.full(16, 5.8), _oy],
                        np.c_[np.linspace(-4.5, 4.5, 10), np.full(10, 10.8)]]  # fmt: skip
FLOOR_WEIGHT = 1.0  # a wrongly placed floor sample costs this many caps


def scaled(cal: CourtCalibration, size: tuple[int, int]) -> tuple[LensModel, dict[str, np.ndarray]]:
    """A calibration's lens and anchor points (raw pixels) moved to another image size."""
    s = size[0] / cal.image_size[0]
    lens = LensModel(cal.lens.k1, cal.lens.k2, cal.lens.f * s, cal.lens.cx * s, cal.lens.cy * s)
    anchors = cal.to_image(np.array([COURT_KEYPOINTS[n] for n in ANCHORS])) * s
    return lens, dict(zip(ANCHORS, anchors, strict=True))


def _homography(lens: LensModel, anchors: np.ndarray) -> np.ndarray | None:
    """Court -> undistorted image homography from the 4 anchor points (raw pixels)."""
    court = np.array([COURT_KEYPOINTS[n] for n in ANCHORS], dtype=np.float32)
    und = lens.undistort(np.asarray(anchors, dtype=np.float64)).astype(np.float32)
    if not np.isfinite(und).all():
        return None
    return cv2.getPerspectiveTransform(court, und)


def _project(lens: LensModel, h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homog = np.c_[pts, np.ones(len(pts))] @ h.T
    w = homog[:, 2:]
    px = np.where(w > 1e-9, homog[:, :2] / np.where(w == 0, 1e-12, w), np.nan)
    return lens.distort(px)


def _lookup(dist: np.ndarray, px: np.ndarray, cap: float) -> np.ndarray:
    h, w = dist.shape
    ok = (
        np.isfinite(px).all(axis=1)
        & (px[:, 0] >= 0)
        & (px[:, 0] <= w - 1)
        & (px[:, 1] >= 0)
        & (px[:, 1] <= h - 1)
    )
    out = np.full(len(px), 0.7 * cap)  # out of view: half a miss (lines do leave the image)
    if ok.any():
        xy = px[ok].astype(np.float32)
        d = cv2.remap(dist, xy[:, 0:1], xy[:, 1:2], cv2.INTER_LINEAR)[:, 0]
        out[ok] = np.minimum(d, cap)
    return out


def score(ev: Evidence, lens: LensModel, anchors: np.ndarray, cap: float = CHAMFER_MAX_PX) -> float:
    """Mean truncated distance (px) from the projected court lines to their evidence."""
    h = _homography(lens, anchors)
    if h is None:
        return cap
    d_paint = _lookup(ev.paint_dist, _project(lens, h, PAINT_SAMPLES), cap)
    d_edge = _lookup(ev.edge_dist, _project(lens, h, EDGE_SAMPLES), cap)
    d_net = _lookup(ev.net_dist, _project(lens, h, NET_SAMPLES), cap)
    miss = np.r_[_on_floor(ev, _project(lens, h, INSIDE_SAMPLES), True),
                 _on_floor(ev, _project(lens, h, OUTSIDE_SAMPLES), False)]  # fmt: skip
    return float(np.mean(np.r_[d_paint, d_edge, d_net]) + FLOOR_WEIGHT * cap * np.mean(miss))


def _on_floor(ev: Evidence, px: np.ndarray, want: bool) -> np.ndarray:
    """1 where a visible sample's floor test disagrees with `want` (out of view: 0)."""
    h, w = ev.floor.shape
    ok = (
        np.isfinite(px).all(axis=1)
        & (px[:, 0] >= 0)
        & (px[:, 0] < w)
        & (px[:, 1] >= 0)
        & (px[:, 1] < h)
    )
    out = np.zeros(len(px))
    if ok.any():
        q = px[ok].astype(int)
        out[ok] = ev.floor[q[:, 1], q[:, 0]] != want
    return out


def _similar(anchors: np.ndarray, centre: np.ndarray, dx: float, dy: float, s: float,
             rot_deg: float) -> np.ndarray:  # fmt: skip
    r = np.deg2rad(rot_deg)
    m = s * np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return (anchors - centre) @ m.T + centre + [dx, dy]


def search(ev: Evidence, lens: LensModel, anchors: np.ndarray, starts: int = 3,
           grid: int = 4) -> tuple[np.ndarray, float]:  # fmt: skip
    """Best anchor points near a template's: a coarse grid (shift, scale, and the far pair moved
    and scaled on its own: camera height and tilt differ between courts), then a local
    8-parameter refinement with a shrinking truncation radius from the best few."""
    from scipy.optimize import minimize

    w = ev.paint.shape[1]
    step = w / 32
    centre = anchors.mean(axis=0)
    cands = []
    for dx in np.arange(-grid, grid + 1) * step:
        for dy in np.arange(-grid, grid + 1) * step:
            for s in (0.8, 0.9, 1.0, 1.12, 1.25):
                base = _similar(anchors, centre, dx, dy, s, 0.0)
                fc = base[2:].mean(axis=0)
                for fdy in np.arange(-2, 3) * step:
                    for fs in (0.8, 1.0, 1.25):
                        a = base.copy()
                        a[2:] = (base[2:] - fc) * fs + fc + [0.0, fdy]
                        cands.append((score(ev, lens, a), a))
    cands.sort(key=lambda c: c[0])
    best_a, best_s = cands[0][1], np.inf
    for _, a in cands[:starts]:
        for cap in (CHAMFER_MAX_PX, 6.0, 3.0):
            res = minimize(
                lambda v, c=cap: score(ev, lens, v.reshape(4, 2), c),
                a.ravel(),
                method="Nelder-Mead",
                options={"xatol": 0.2, "fatol": 1e-4, "maxiter": 1500},
            )
            a = res.x.reshape(4, 2)
        s_final = score(ev, lens, a, 3.0)
        if s_final < best_s:
            best_a, best_s = a, s_final
    return best_a, best_s


# --- precise fit -------------------------------------------------------------------------------
def _line_guide(cal: CourtCalibration, name: str, size: tuple[int, int]) -> np.ndarray:
    """The named line projected into the image (visible part, >= 20 px from the border)."""
    a, b, c = NAMED_LINES[name]
    t = np.linspace(-10.5, 10.5, 300)
    pts = np.c_[-c * a - t * b, -c * b + t * a]
    if name == "center_line":
        pts = pts[np.abs(pts[:, 1]) <= 6.95]
    elif name == "net_base":  # between the posts, a little in from the walls
        pts = pts[np.abs(pts[:, 0]) <= 4.6]
    else:
        pts = pts[(np.abs(pts[:, 0]) <= 5.0) & (np.abs(pts[:, 1]) <= 10.0)]
    px = cal.to_image(pts)
    w, h = size
    ok = (
        np.isfinite(px).all(axis=1)
        & (px[:, 0] >= 20)
        & (px[:, 0] <= w - 20)
        & (px[:, 1] >= 20)
        & (px[:, 1] <= h - 20)
    )
    return px[ok]


def refine(
    image: np.ndarray,
    rough: CourtCalibration,
    ev: Evidence | None = None,
    rounds: tuple[float, ...] = (10.0, 5.0),
) -> tuple[CourtCalibration, dict[str, int]]:
    """Snap every visible named line along its projection and refit lens + court, `rounds`
    times with the given search half-widths (px). Wall lines (where the floor meets a wall)
    snap first to the outline of the floor region, which marks on the floor can't fool, and
    then by EDGE_FINE_PX to the image edge there."""
    from padelvision.court.snap import snap_line

    ev = ev or evidence(image)
    floor_img = np.repeat((ev.floor * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    size = (image.shape[1], image.shape[0])
    cal = rough
    lens = rough.lens
    found: dict[str, list] = {}
    for half in rounds:
        found = {}
        for name, mode in SNAP_MODE.items():
            guide = _line_guide(cal, name, size)
            if len(guide) < 2:
                continue
            guide_pts = [tuple(p) for p in guide]
            if mode == "net":  # the strongest dark-above-floor ridge along the net
                pts = snap_line(ev.net_ridge, guide_pts, "paint", half_px=2 * half)
            elif mode == "edge":
                coarse = snap_line(floor_img, guide_pts, "edge", half_px=2 * half)
                pts = (
                    snap_line(image, coarse, "edge", half_px=EDGE_FINE_PX)
                    if len(coarse) >= 2
                    else []
                )
            else:
                pts = snap_line(image, guide_pts, mode, half_px=half)
            if len(pts) >= MIN_LINE_POINTS:
                found[name] = pts
        if len(found) < MIN_LINES:
            break
        # The lens is kept from the template: refitted from partly visible lines it bends the
        # image edges to fit better and overfits (same camera mode = same lens).
        try:
            cal = CourtCalibration.from_points({}, size, named_lines=found, lens=lens)
        except (ValueError, np.linalg.LinAlgError):
            break
    return cal, {n: len(p) for n, p in found.items()}


# --- the whole thing ---------------------------------------------------------------------------
def default_templates() -> list[Path]:
    """Calibrations shipped in the repo (ml/calibrations/*.json), if the repo is there."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        d = parent / "ml" / "calibrations"
        if d.is_dir():
            return sorted(d.glob("*.json"))
    return []


def auto_calibrate(image: np.ndarray, templates: list[str | Path] | None = None) -> AutoResult:
    """Calibrate from a background image (players removed). Tries each template calibration
    as the starting point and keeps the best precise fit; `ok` says whether it passed the gate."""
    size = (image.shape[1], image.shape[0])
    paths = [Path(p) for p in (templates if templates is not None else default_templates())]
    if not paths:
        return AutoResult(None, False, ["no template calibrations to start from"])
    ev = evidence(image)
    best: AutoResult | None = None
    for path in paths:
        tmpl = CourtCalibration.load(path)
        lens, anc = scaled(tmpl, size)
        a, rough_score = search(ev, lens, np.array([anc[n] for n in ANCHORS]))
        try:
            rough = CourtCalibration.from_points(dict(zip(ANCHORS, map(tuple, a), strict=True)),
                                                 size, lens=lens)  # fmt: skip
            cal, lines = refine(image, rough, ev)
        except (ValueError, np.linalg.LinAlgError) as e:
            res = AutoResult(None, False, [f"{path.stem}: {e}"], path.stem, rough_score)
        else:
            res = _judge(cal, lines, path.stem, rough_score)
        if best is None or _rank(res) < _rank(best):
            best = res
    assert best is not None
    return best


def _rank(r: AutoResult) -> tuple:
    rms = (r.accuracy or {}).get("rms_px")
    return (not r.ok, -len(r.lines), rms if rms is not None else np.inf)


def _judge(
    cal: CourtCalibration, lines: dict[str, int], template: str, rough_score: float
) -> AutoResult:
    acc = cal.accuracy()
    reasons = []
    if len(lines) < MIN_LINES:
        reasons.append(f"only {len(lines)} court lines found (need {MIN_LINES})")
    near = {"near_service_line"} & set(lines)
    far = {"far_service_line", "far_baseline"} & set(lines)
    if not near or not far:
        reasons.append("court lines not found on both sides of the net")
    if "net_base" not in lines:  # without it the far lines can slip by one and still fit
        reasons.append("the net's base was not found")
    if acc["rms_px"] is None or acc["rms_px"] > MAX_RMS_PX:
        reasons.append(f"fit {acc['rms_px']} px (max {MAX_RMS_PX})")
    near_err = acc["zones"]["near_half"]["expected_error_m"]
    if near_err is None or near_err > MAX_NEAR_ERROR_M:
        reasons.append(f"near-half ground error {near_err} m (max {MAX_NEAR_ERROR_M})")
    return AutoResult(cal, not reasons, reasons, template, round(rough_score, 2), lines, acc)
