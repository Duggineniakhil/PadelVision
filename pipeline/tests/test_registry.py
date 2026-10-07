import hashlib

import pytest

from padelvision.models import registry


@pytest.fixture
def weights_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "WEIGHTS_DIR", tmp_path / "weights")
    return tmp_path / "weights"


def _spec(tmp_path, payload: bytes, sha: str | None) -> registry.ModelSpec:
    src = tmp_path / "remote.pt"
    src.write_bytes(payload)
    return registry.ModelSpec(
        name="toy", file="toy.pt", url=src.as_uri(), sha256=sha, task="detect", license="MIT"
    )


def test_repo_manifest_is_valid():
    manifest = registry.load_manifest()
    assert "person-detector" in manifest
    for spec in manifest.values():
        assert spec.url.startswith("https://")
        assert spec.sha256 is None or len(spec.sha256) == 64


def test_fetch_verifies_and_is_idempotent(tmp_path, weights_dir):
    payload = b"fake weights"
    spec = _spec(tmp_path, payload, hashlib.sha256(payload).hexdigest())
    path = registry.fetch(spec)
    assert path.read_bytes() == payload
    assert registry.fetch(spec) == path
    assert registry.weights_path("toy", {"toy": spec}) == path


def test_fetch_rejects_bad_hash(tmp_path, weights_dir):
    spec = _spec(tmp_path, b"tampered", "0" * 64)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        registry.fetch(spec)
    assert not (weights_dir / "toy.pt").exists()
    assert not list(weights_dir.glob("*.part"))


def test_missing_weights_error_says_how_to_fix(tmp_path, weights_dir):
    spec = _spec(tmp_path, b"x", None)
    with pytest.raises(registry.ModelNotFoundError, match="fetch_models.py toy"):
        registry.weights_path("toy", {"toy": spec})
    with pytest.raises(registry.ModelNotFoundError, match="not in"):
        registry.weights_path("nope", {"toy": spec})
