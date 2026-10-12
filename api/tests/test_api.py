import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

VIDEO = {"path": "/kaggle/x/My_match.mp4", "width": 1280, "height": 720, "fps": 30.0,
         "frame_count": 900, "duration_s": 30.0}  # fmt: skip
STATS = {
    "players": {
        "1": {"team": "near", "tracked_fraction": 0.9, "tracked_s": 27.0, "distance_m": 40.0,
              "avg_speed_mps": 1.5, "peak_speed_mps": 4.0, "sprints": 1,
              "zones": {"net": 0.2, "transition": 0.3, "back": 0.5}},
        "2": {"team": "near", "tracked_fraction": 0.1,
              "insufficient_data": "tracked 10% of the time"},
        "3": {"team": "far", "tracked_fraction": 0.9, "tracked_s": 27.0, "distance_m": 30.0,
              "avg_speed_mps": 1.1, "peak_speed_mps": 3.0, "sprints": 0,
              "zones": {"net": 0.5, "transition": 0.3, "back": 0.2}},
        "4": {"team": "far", "tracked_fraction": 0.0,
              "insufficient_data": "tracked 0% of the time"},
    },
    "teams": {"near": {"insufficient_data": "both partners tracked 9% of the time"},
              "far": {"insufficient_data": "both partners tracked 0% of the time"}},
}  # fmt: skip
RALLIES = {
    "hits_by_player": {"1": 2, "3": 1},
    "rallies": [{"id": 1, "start_frame": 30, "end_frame": 300, "start_s": 1.0, "end_s": 10.0,
                 "duration_s": 9.0, "hits": 3, "bounces": 2, "hits_by_player": {"1": 2, "3": 1}}],
    "other_activity": [{"id": 2}],
    "stats": {
        "shot_speed_estimate": {"samples": 1, "insufficient_data": "fewer than 3 measured shots"},
        "per_rally": [{"id": 1, "hits": 3, "measured_shots": 1, "max_shot_kmh": 50.0}],
        "shots": [
            {"frame": 40, "player": 1, "team": "near", "hitter_xy": [0.0, -8.0],
             "bounce": {"frame": 70, "x_m": 1.0, "y_m": 6.0, "zone": "transition"},
             "speed_kmh": 50.0, "speed_range_kmh": [42, 60], "speed_note": None,
             "no_landing": None},
            {"frame": 100, "player": 3, "team": "far", "hitter_xy": [1.0, 7.0], "bounce": None,
             "speed_kmh": None, "speed_note": "probably a volley: no landing",
             "no_landing": "probably_volley"},
            {"frame": 130, "player": 1, "team": "near", "hitter_xy": None, "bounce": None,
             "speed_kmh": None, "speed_note": "landing not seen", "no_landing": "not_seen"},
        ],
    },
}  # fmt: skip


def _zip(files: dict[str, str], folder: str = "run_cache") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(f"{folder}/{name}", text)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PADELVISION_RUNS", str(tmp_path / "runs"))
    from padelvision_api.main import app

    return TestClient(app)


def _import(client, files=None, name=None):
    files = files or {"video.json": json.dumps(VIDEO), "stats.json": json.dumps(STATS),
                      "rallies.json": json.dumps(RALLIES)}  # fmt: skip
    data = {"name": name} if name else {}
    return client.post("/api/runs", files=[("files", ("results.zip", _zip(files)))], data=data)


def _wait_ready(client, run_id):
    for _ in range(100):
        d = client.get(f"/api/runs/{run_id}").json()
        if d["status"] != "processing":
            return d
        time.sleep(0.05)
    raise AssertionError("still processing")


def test_import_list_detail_and_views(client):
    r = _import(client)
    assert r.status_code == 201
    run_id = r.json()["id"]
    assert r.json()["name"] == "My_match" and run_id.startswith("my-match-")
    d = _wait_ready(client, run_id)
    assert d["status"] == "ready" and d["video"]["frame_count"] == 900 and not d["has_video"]
    assert [p["player"] for p in d["players"]] == [1, 2, 3, 4]
    p1 = d["players"][0]
    assert p1["avg_speed_kmh"] == 5.4 and p1["rally_hits"] == 2
    assert d["players"][1]["distance_m"] is None and d["players"][1]["insufficient_data"]
    assert d["rally_summary"] == {
        "count": 1,
        "total_s": 9.0,
        "median_s": 9.0,
        "max_s": 9.0,
        "hits_total": 3,
        "hits_per_rally": 3.0,
        "other_activity": 1,
    }
    titles = [q["title"] for q in d["quality"]]
    assert "No movement stats for player 2" in titles and "No shot speeds" in titles
    assert [x["id"] for x in client.get("/api/runs").json()] == [run_id]

    ra = client.get(f"/api/runs/{run_id}/rallies").json()
    rally = ra["rallies"][0]
    assert rally["landed"] == 1 and rally["probable_volleys"] == 1 and rally["max_shot_kmh"] == 50
    assert [s["index"] for s in ra["shots"]] == [1, 2, 3]
    assert ra["shots"][0]["speed_range_kmh"] == [42, 60] and ra["shots"][0]["bounce"]["t"] == 2.33
    assert client.get(f"/api/runs/{run_id}/events").json() == []
    assert client.get(f"/api/runs/{run_id}/tracks").json()["players"] == {}
    assert client.get(f"/api/runs/{run_id}/highlights").json() == {
        "highlights": None,
        "condensed": None,
    }


def test_rename_delete_and_errors(client):
    run_id = _import(client, name="Final").json()["id"]
    _wait_ready(client, run_id)
    assert client.patch(f"/api/runs/{run_id}", json={"name": " Semi final "}).json()["name"] == (
        "Semi final"
    )
    assert client.patch(f"/api/runs/{run_id}", json={"name": " "}).status_code == 422
    assert client.get(f"/api/runs/{run_id}/files/../meta.json").status_code == 404
    assert client.get(f"/api/runs/{run_id}/files/stats.json").status_code == 404  # not media
    assert client.get(f"/api/runs/{run_id}/video").status_code == 404
    assert client.put(f"/api/runs/{run_id}/video", json={"path": "C:/nope.mp4"}).status_code == 422
    assert client.delete(f"/api/runs/{run_id}").status_code == 204
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    assert client.get("/api/runs/..%2F..").status_code == 404


def test_import_rejects_incomplete_and_unsafe_zips(client):
    r = _import(client, files={"stats.json": json.dumps(STATS)})
    assert r.status_code == 422 and "video.json" in r.json()["detail"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../../video.json", json.dumps(VIDEO))
    r = client.post("/api/runs", files=[("files", ("evil.zip", buf.getvalue()))])
    assert r.status_code == 422 and "Unsafe" in r.json()["detail"]
    assert client.get("/api/runs").json() == []


def test_add_highlight_files_to_an_existing_run(client):
    run_id = _import(client).json()["id"]
    _wait_ready(client, run_id)
    clips = {"total_s": 12.0, "clips": [{"rally_id": 1, "start_s": 0.5, "end_s": 10.5,
                                         "duration_s": 9.0, "hits": 3, "score": 5.0}]}  # fmt: skip
    files = {"highlights.mp4": "fake video", "highlights.json": json.dumps(clips),
             "rallies.json": "{}", "video.json": json.dumps(VIDEO)}  # fmt: skip
    upload = [("files", ("highlights.zip", _zip(files, "highlights_out")))]
    r = client.post(f"/api/runs/{run_id}/files", files=upload)
    assert r.status_code == 200 and r.json() == {"updated": ["highlights.json", "highlights.mp4"]}
    hl = client.get(f"/api/runs/{run_id}/highlights").json()["highlights"]
    assert hl["file"] == "highlights.mp4" and hl["clips"][0]["rally_id"] == 1
    assert client.get(f"/api/runs/{run_id}/rallies").json()["rallies"][0]["id"] == 1  # untouched

    other = dict(VIDEO, frame_count=5000)
    r = client.post(f"/api/runs/{run_id}/files", files=[("files", ("x.zip", _zip(
        {"highlights.mp4": "v", "video.json": json.dumps(other)})))])  # fmt: skip
    assert r.status_code == 422 and "another video" in r.json()["detail"]
    r = client.post(
        f"/api/runs/{run_id}/files", files=[("files", ("x.zip", _zip({"stats.json": "{}"})))]
    )
    assert r.status_code == 422 and "No highlight videos" in r.json()["detail"]


def _png(w=320, h=180) -> bytes:
    import cv2
    import numpy as np

    img = np.full((h, w, 3), (150, 90, 40), np.uint8)
    cv2.line(img, (10, 120), (310, 120), (235, 235, 235), 3)
    return cv2.imencode(".png", img)[1].tobytes()


def _synthetic_constraints(w=1280, h=720):
    """Keypoints projected by a plausible camera homography (no lens)."""
    import numpy as np
    from padelvision.court.geometry import COURT_KEYPOINTS

    hmat = np.array([[60.0, 0.0, 640.0], [0.0, -12.0, 360.0], [0.0, 0.06, 1.0]])
    pts = {}
    for name in ("near_left", "near_right", "net_left", "net_right", "far_left", "far_right",
                 "near_service_center", "far_service_center"):  # fmt: skip
        x, y, s = hmat @ [*COURT_KEYPOINTS[name], 1.0]
        pts[name] = [x / s, y / s]
    return {"image_size": [w, h], "image_points": pts}


def test_calibration_image_snap_fit_and_run_court(client):
    r = client.post("/api/calibration/images", files=[("files", ("frame.png", _png()))])
    assert r.status_code == 201
    img = r.json()
    assert (img["width"], img["height"], img["source"]) == (320, 180, "upload")
    assert client.get(f"/api/calibration/images/{img['id']}").status_code == 200
    assert client.get("/api/calibration/images/..%2Fx").status_code == 404
    zipped = _zip({"background.png": "x"}, "calibration")  # unreadable image inside a zip
    upload = [("files", ("calibration.zip", zipped))]
    assert client.post("/api/calibration/images", files=upload).status_code == 422

    r = client.post("/api/calibration/snap",
                    json={"image_id": img["id"], "guide": [[20, 116], [300, 124]]})  # fmt: skip
    pts = r.json()["points"]
    assert len(pts) > 20 and all(abs(y - 120) < 1 for _, y in pts)

    body = _synthetic_constraints()
    fit = client.post("/api/calibration/fit", json=body).json()
    assert fit["accuracy"]["rms_px"] < 0.01 and len(fit["court_lines"]) >= 6
    assert set(fit["court_json"]) >= {"image_size", "image_points", "image_to_court", "lens"}
    too_few = {"image_size": [1280, 720], "image_points": {"near_left": [1, 2]}}
    bad = client.post("/api/calibration/fit", json=too_few)
    assert bad.status_code == 422 and "at least 4" in bad.json()["detail"]

    run_id = _import(client).json()["id"]
    _wait_ready(client, run_id)
    court = client.get(f"/api/runs/{run_id}/court").json()
    assert court["calibration"] is None and court["video_size"] == [1280, 720]
    assert not court["can_recompute"] and not court["has_video"]
    r = client.put(f"/api/runs/{run_id}/court", json=body)
    assert r.status_code == 422 and "raw detections" in r.json()["detail"]
    r = client.post(f"/api/runs/{run_id}/court/background")
    assert r.status_code == 422 and "video" in r.json()["detail"]
    assert client.post(f"/api/runs/{run_id}/court/restore").status_code == 422
