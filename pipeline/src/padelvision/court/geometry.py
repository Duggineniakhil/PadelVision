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

# The 4 corners.
CORNER_NAMES = ("near_left", "near_right", "far_right", "far_left")

# Order the calibration tool asks for keypoints: far to near, matching how they appear on screen.
CLICK_ORDER = (
    "far_left",
    "far_right",
    "far_service_left",
    "far_service_center",
    "far_service_right",
    "net_left",
    "net_right",
    "near_service_left",
    "near_service_center",
    "near_service_right",
    "near_left",
    "near_right",
)

# Painted lines (plus the net), as (start, end) in metres; used for drawing overlays.
COURT_LINES: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((-_HW, -_HL), (_HW, -_HL)),
    ((-_HW, _HL), (_HW, _HL)),
    ((-_HW, -_HL), (-_HW, _HL)),
    ((_HW, -_HL), (_HW, _HL)),
    ((-_HW, -_S), (_HW, -_S)),
    ((-_HW, _S), (_HW, _S)),
    ((0.0, -_S), (0.0, _S)),
    ((-_HW, 0.0), (_HW, 0.0)),
]

# Named straight court lines usable as calibration constraints: (a, b, c) with a*x + b*y + c = 0.
# The side lines are where the floor meets the side walls.
NAMED_LINES: dict[str, tuple[float, float, float]] = {
    "left_side": (1.0, 0.0, _HW),
    "right_side": (1.0, 0.0, -_HW),
    "center_line": (1.0, 0.0, 0.0),
    "near_service_line": (0.0, 1.0, _S),
    "far_service_line": (0.0, 1.0, -_S),
    "near_baseline": (0.0, 1.0, _HL),
    "far_baseline": (0.0, 1.0, -_HL),
}

# Zones by distance from the net (|y|). Net play happens inside NET_ZONE_M; back-court is
# behind the service line; in between is the transition zone.
NET_ZONE_M = 4.0
BACK_ZONE_M = SERVICE_LINE_FROM_NET_M


def in_court(x, y, margin_m: float = 0.0):
    """True if a ground point (metres) lies inside the court, expanded by `margin_m`.

    Works on scalars or numpy arrays.
    """
    return (abs(x) <= _HW + margin_m) & (abs(y) <= _HL + margin_m)


def zone_of(y):
    """'net' | 'transition' | 'back' for a distance along the court (scalar)."""
    d = abs(y)
    if d <= NET_ZONE_M:
        return "net"
    if d <= BACK_ZONE_M:
        return "transition"
    return "back"
