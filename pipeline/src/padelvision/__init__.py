"""PadelVision: computer-vision analytics for padel match videos."""

from pathlib import Path

__version__ = "0.1.0"

# pipeline/src/padelvision/__init__.py -> repo root is three levels above the package dir.
REPO_ROOT = Path(__file__).resolve().parents[3]
