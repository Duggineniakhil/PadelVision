"""Padel court dimensions and named keypoints, in metres.

Coordinate system (single source of truth for the whole project):
- origin at the court centre (middle of the net)
- x runs across the court: -5 (left) .. +5 (right), as seen from the camera
- y runs along the court: -10 (near baseline, closest to camera) .. +10 (far baseline)
- the net is at y = 0
"""

COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0
SERVICE_LINE_FROM_NET_M = 6.95

_HW = COURT_WIDTH_M / 2
_HL = COURT_LENGTH_M / 2
_S = SERVICE_LINE_FROM_NET_M

# Points that are visible on the ground and can be clicked or detected.
COURT_KEYPOINTS: dict[str, tuple[float, float]] = {
    "near_left": (-_HW, -_HL),
    "near_right": (_HW, -_HL),
    "far_left": (-_HW, _HL),
    "far_right": (_HW, _HL),
    "net_left": (-_HW, 0.0),
    "net_right": (_HW, 0.0),
    "near_service_left": (-_HW, -_S),
    "near_service_center": (0.0, -_S),
    "near_service_right": (_HW, -_S),
    "far_service_left": (-_HW, _S),
    "far_service_center": (0.0, _S),
    "far_service_right": (_HW, _S),
}

# The 4 corners, in the order the calibration tool asks for them.
CORNER_NAMES = ("near_left", "near_right", "far_right", "far_left")


def in_court(x: float, y: float, margin_m: float = 0.0) -> bool:
    """True if a ground point (metres) lies inside the court, expanded by `margin_m`."""
    return abs(x) <= _HW + margin_m and abs(y) <= _HL + margin_m
