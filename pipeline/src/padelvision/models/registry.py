"""Model manifest, download and lookup.

Weights are never committed. `models/manifest.yaml` declares every model (url, sha256,
license); `fetch()` downloads and verifies them into `models/weights/`; pipeline code
gets a model's file only through `weights_path(name)`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

from padelvision import REPO_ROOT

MANIFEST_PATH = REPO_ROOT / "models" / "manifest.yaml"
# Override on Kaggle/Colab to keep weights on persistent storage.
WEIGHTS_DIR = Path(os.environ.get("PADELVISION_WEIGHTS_DIR", REPO_ROOT / "models" / "weights"))

FETCH_HINT = "python scripts/fetch_models.py"


class ModelNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    file: str
    url: str
    sha256: str | None
    task: str
    license: str
    source: str = ""
    notes: str = ""


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    models = raw.get("models") or {}
    return {name: ModelSpec(name=name, **fields) for name, fields in models.items()}


def weights_path(name: str, manifest: dict[str, ModelSpec] | None = None) -> Path:
    """Path to a model's weights. Raises ModelNotFoundError with a fix if missing."""
    manifest = manifest if manifest is not None else load_manifest()
    if name not in manifest:
        raise ModelNotFoundError(
            f"Model '{name}' is not in {MANIFEST_PATH}. Known: {sorted(manifest)}"
        )
    path = WEIGHTS_DIR / manifest[name].file
    if not path.exists():
        raise ModelNotFoundError(
            f"Weights for '{name}' not found at {path}. Run: {FETCH_HINT} {name}"
        )
    return path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(spec: ModelSpec, force: bool = False) -> Path:
    """Download one model (if needed) and verify its sha256."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / spec.file

    if dest.exists() and not force:
        _verify(spec, dest)
        return dest

    with tempfile.NamedTemporaryFile(dir=WEIGHTS_DIR, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(spec.url) as resp:
            shutil.copyfileobj(resp, tmp)
    try:
        _verify(spec, tmp_path)
    except ValueError:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(dest)
    return dest


def _verify(spec: ModelSpec, path: Path) -> None:
    if spec.sha256 is None:
        return
    actual = sha256_of(path)
    if actual != spec.sha256:
        raise ValueError(
            f"sha256 mismatch for '{spec.name}' ({path}): expected {spec.sha256}, got {actual}"
        )
