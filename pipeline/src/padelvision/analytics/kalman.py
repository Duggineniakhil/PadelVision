"""Constant-velocity Kalman filter + Rauch-Tung-Striebel smoother for 2-D court positions.

Each measurement carries its own covariance (from the calibration: a pixel of foot jitter
is a few cm near the camera but ~0.3 m at the far baseline of a low camera), so far-side
positions are trusted less and smoothed more. Missing samples (short gaps) are handled by
prediction only. Measurements that are implausible given the motion model are ignored,
unless several in a row disagree (then the person really moved and the filter re-locks).
"""

from __future__ import annotations

import numpy as np

ACCEL_STD_MPS2 = 4.0  # how hard a padel player can change velocity
INIT_SPEED_STD_MPS = 2.0
GATE_CHI2 = 16.0  # ~4 sigma
MAX_REJECTS = 5


def smooth(z: np.ndarray, r: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """z: (n, 2) positions (NaN rows = missing). r: (n, 2, 2) measurement covariances.

    Returns smoothed positions (n, 2) and velocities (n, 2).
    """
    n = len(z)
    f = np.eye(4)
    f[0, 2] = f[1, 3] = dt
    q1 = ACCEL_STD_MPS2**2 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    q = np.zeros((4, 4))
    q[np.ix_([0, 2], [0, 2])] = q1
    q[np.ix_([1, 3], [1, 3])] = q1
    h = np.zeros((2, 4))
    h[0, 0] = h[1, 1] = 1.0

    first = int(np.argmax(np.isfinite(z).all(axis=1)))
    x = np.r_[z[first], 0.0, 0.0]
    p = np.diag([r[first, 0, 0], r[first, 1, 1], INIT_SPEED_STD_MPS**2, INIT_SPEED_STD_MPS**2])

    xp = np.zeros((n, 4))
    pp = np.zeros((n, 4, 4))
    xf = np.zeros((n, 4))
    pf = np.zeros((n, 4, 4))
    rejects = 0
    for k in range(n):
        if k > 0:
            x = f @ x
            p = f @ p @ f.T + q
        xp[k], pp[k] = x, p
        if k >= first and np.isfinite(z[k]).all():
            s = h @ p @ h.T + r[k]
            innov = z[k] - h @ x
            s_inv = np.linalg.inv(s)
            if innov @ s_inv @ innov <= GATE_CHI2 or rejects >= MAX_REJECTS or k == first:
                gain = p @ h.T @ s_inv
                x = x + gain @ innov
                p = (np.eye(4) - gain @ h) @ p
                rejects = 0
            else:
                rejects += 1
        xf[k], pf[k] = x, p

    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        c = pf[k] @ f.T @ np.linalg.inv(pp[k + 1])
        xs[k] = xf[k] + c @ (xs[k + 1] - xp[k + 1])
    return xs[:, :2], xs[:, 2:]
