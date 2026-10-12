"""JSON views of a run folder for the web app. Read-only; nothing here computes new analytics."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from padelvision_api import schemas as S
from padelvision_api.store import MEDIA_FILES, read_meta

BALL_COVERAGE_WARN = 0.5
NEAR_ERROR_WARN_M = 0.15
OVERLAY_MAX_S = 120.0


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


@lru_cache(maxsize=32)
def _parquet_cached(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def _parquet(path: Path) -> pd.DataFrame | None:
    return _parquet_cached(str(path), path.stat().st_mtime) if path.exists() else None


def _num(v) -> float | None:
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


# --- list + detail ---------------------------------------------------------------------------
def list_item(run: Path) -> S.RunListItem:
    meta = read_meta(run)
    video = _json(run / "video.json") or {}
    rallies = _json(run / "rallies.json")
    has_video = bool(meta.get("video_path")) and Path(meta["video_path"]).exists()
    return S.RunListItem(
        id=meta["id"], name=meta["name"], created=meta["created"], status=meta["status"],
        status_message=meta.get("status_message"),
        duration_s=video.get("duration_s"),
        rallies=len(rallies["rallies"]) if rallies else None,
        hits=sum(rallies.get("hits_by_player", {}).values()) if rallies else None,
        has_video=has_video,
        has_highlights=(run / "highlights.mp4").exists(),
    )  # fmt: skip


def detail(run: Path) -> S.RunDetail:
    meta = read_meta(run)
    video = _json(run / "video.json")
    stats = _json(run / "stats.json") or {}
    rallies = _json(run / "rallies.json")
    ball_stats = _json(run / "ball_stats.json")
    rs = (rallies or {}).get("stats", {})
    vpath = meta.get("video_path")
    has_video = bool(vpath) and Path(vpath).exists()

    players = _players(stats, rallies)
    court = court_info(run)
    coverage = ball_stats.get("coverage") if ball_stats else None
    rally_summary = None
    if rallies is not None:
        durations = [r["duration_s"] for r in rallies["rallies"]]
        hits = [r["hits"] for r in rallies["rallies"]]
        rally_summary = S.RallySummary(
            count=len(durations), total_s=round(sum(durations), 1),
            median_s=round(float(np.median(durations)), 1) if durations else None,
            max_s=round(max(durations), 1) if durations else None,
            hits_total=int(sum(hits)),
            hits_per_rally=round(float(np.mean(hits)), 1) if hits else None,
            other_activity=len(rallies.get("other_activity", [])),
        )  # fmt: skip
    sp = rs.get("shot_speed_estimate")
    speed = None
    if sp is not None:
        speed = S.SpeedSummary(samples=sp.get("samples", 0), median_kmh=sp.get("median_kmh"),
                               max_kmh=sp.get("max_kmh"),
                               insufficient_data=sp.get("insufficient_data"),
                               note=sp.get("note"))  # fmt: skip
    return S.RunDetail(
        id=meta["id"], name=meta["name"], created=meta["created"], status=meta["status"],
        status_message=meta.get("status_message"),
        video=S.VideoInfo(**{k: video[k] for k in S.VideoInfo.model_fields}) if video else None,
        has_video=has_video, video_name=Path(vpath).name if has_video else None,
        media=sorted(n for n in MEDIA_FILES if (run / n).exists()),
        court=court, ball_coverage=coverage, players=players,
        teams=_teams(stats), rally_summary=rally_summary, speed_summary=speed,
        quality=_quality(court, coverage, players, rally_summary, speed),
    )  # fmt: skip


def _players(stats: dict, rallies: dict | None) -> list[S.PlayerStats]:
    hits = (rallies or {}).get("hits_by_player", {})
    sp = (rallies or {}).get("stats", {}).get("shot_speed_estimate", {})
    out = []
    for pid, p in sorted(stats.get("players", {}).items(), key=lambda kv: int(kv[0])):
        ok = "insufficient_data" not in p
        out.append(S.PlayerStats(
            player=int(pid), team=p["team"], tracked_fraction=p.get("tracked_fraction", 0.0),
            insufficient_data=p.get("insufficient_data"),
            tracked_s=p.get("tracked_s") if ok else None,
            distance_m=p.get("distance_m") if ok else None,
            avg_speed_kmh=round(p["avg_speed_mps"] * 3.6, 1) if ok and p.get("avg_speed_mps")
            is not None else None,
            peak_speed_kmh=round(p["peak_speed_mps"] * 3.6, 1) if ok and p.get("peak_speed_mps")
            is not None else None,
            sprints=p.get("sprints") if ok else None,
            zones=p.get("zones") if ok else None,
            rally_hits=int(hits.get(pid, 0)),
            shot_speed_median_kmh=sp.get("by_player_median_kmh", {}).get(pid),
            shot_speed_max_kmh=sp.get("by_player_max_kmh", {}).get(pid),
            shot_speed_samples=int(sp.get("by_player_samples", {}).get(pid, 0)),
        ))  # fmt: skip
    return out


def _teams(stats: dict) -> list[S.TeamStats]:
    out = []
    for team in ("near", "far"):
        t = stats.get("teams", {}).get(team)
        if t is None:
            continue
        out.append(S.TeamStats(
            team=team, insufficient_data=t.get("insufficient_data"),
            both_tracked_fraction=t.get("both_tracked_fraction"),
            formation=t.get("formation"), partner_spacing_m=t.get("partner_spacing_m"),
        ))  # fmt: skip
    return out


def _quality(court, coverage, players, rally_summary, speed) -> list[S.QualityNote]:
    notes = []
    if coverage is not None and coverage < BALL_COVERAGE_WARN:
        notes.append(S.QualityNote(
            level="warning", title=f"Ball found on {coverage:.0%} of frames",
            detail="Play where the ball wasn't detected isn't counted: rallies may be split or "
                   "missing, and hit, bounce and shot numbers are lower than the real ones.",
        ))  # fmt: skip
    if court is not None:
        near = court.zones.get("near_half")
        if near and near.expected_error_m is not None and near.expected_error_m > NEAR_ERROR_WARN_M:
            notes.append(S.QualityNote(
                level="warning", title="Court calibration is imprecise",
                detail=f"Positions in the near half may be off by about "
                       f"{near.expected_error_m:.2f} m. Recalibrate this camera position.",
            ))  # fmt: skip
    for p in players:
        if p.insufficient_data:
            notes.append(S.QualityNote(
                level="info", title=f"No movement stats for player {p.player}",
                detail=f"Player {p.player} was {p.insufficient_data}.",
            ))  # fmt: skip
    if rally_summary is not None and rally_summary.count == 0:
        notes.append(
            S.QualityNote(
                level="warning",
                title="No rallies detected",
                detail="No exchange of shots between the teams was found.",
            )
        )
    if speed is not None and speed.insufficient_data:
        notes.append(S.QualityNote(level="info", title="No shot speeds",
                                   detail=f"Shot speeds need at least 3 shots with a clearly "
                                          f"tracked landing ({speed.samples} found)."))  # fmt: skip
    return notes


# --- court -------------------------------------------------------------------------------------
def court_info(run: Path) -> S.CourtInfo | None:
    if not (run / "court.json").exists():
        return None
    return _court_info_cached(str(run / "court.json"), (run / "court.json").stat().st_mtime)


@lru_cache(maxsize=16)
def _court_info_cached(path: str, mtime: float) -> S.CourtInfo:
    from padelvision.court import CourtCalibration
    from padelvision.court.draw import court_polylines

    cal = CourtCalibration.load(path)
    w, h = cal.image_size
    lines = [[(round(float(x), 1), round(float(y), 1)) for x, y in seg]
             for seg in court_polylines(cal, samples=60)]  # fmt: skip
    acc = cal.accuracy()
    zones = {k: S.ZoneAccuracy(**v) for k, v in acc["zones"].items()}
    return S.CourtInfo(image_width=w, image_height=h, rms_px=acc["rms_px"], zones=zones,
                       lines_image=lines)  # fmt: skip


# --- rallies, events, tracks, overlay -------------------------------------------------------------
def rallies(run: Path) -> S.RalliesResponse:
    data = _json(run / "rallies.json")
    if data is None:
        return S.RalliesResponse(rallies=[], shots=[])
    fps = (_json(run / "video.json") or {}).get("fps", 30.0)
    per = {r["id"]: r for r in data.get("stats", {}).get("per_rally", [])}
    shots_raw = data.get("stats", {}).get("shots", [])
    out_r, out_s = [], []
    for r in data["rallies"]:
        mine = [s for s in shots_raw if r["start_frame"] <= s["frame"] <= r["end_frame"]]
        for i, s in enumerate(mine, start=1):
            b = s.get("bounce")
            rng = s.get("speed_range_kmh")
            out_s.append(S.Shot(
                frame=s["frame"], t=round(s["frame"] / fps, 2), rally_id=r["id"], index=i,
                player=s["player"], team=s["team"],
                hitter_xy=tuple(s["hitter_xy"]) if s.get("hitter_xy") else None,
                bounce=S.Bounce(frame=b["frame"], t=round(b["frame"] / fps, 2), x_m=b["x_m"],
                                y_m=b["y_m"], zone=b["zone"]) if b else None,
                speed_kmh=s.get("speed_kmh"), speed_range_kmh=tuple(rng) if rng else None,
                speed_note=s.get("speed_note"), no_landing=s.get("no_landing"),
            ))  # fmt: skip
        out_r.append(S.Rally(
            id=r["id"], start_frame=r["start_frame"], end_frame=r["end_frame"],
            start_s=r["start_s"], end_s=r["end_s"], duration_s=r["duration_s"], hits=r["hits"],
            bounces=r.get("bounces", 0), hits_by_player=r.get("hits_by_player", {}),
            max_shot_kmh=per.get(r["id"], {}).get("max_shot_kmh"),
            measured_shots=per.get(r["id"], {}).get("measured_shots", 0),
            landed=sum(s.get("bounce") is not None for s in mine),
            probable_volleys=sum(s.get("no_landing") == "probably_volley" for s in mine),
        ))  # fmt: skip
    return S.RalliesResponse(rallies=out_r, shots=out_s)


def events(run: Path) -> list[S.Event]:
    ev = _parquet(run / "events.parquet")
    if ev is None:
        return []
    out = []
    for e in ev.itertuples():
        out.append(S.Event(
            frame=int(e.frame), t=round(float(e.t), 3), kind=e.kind, u=float(e.u), v=float(e.v),
            player=int(e.player) if e.player == e.player and e.player is not None else None,
            team=e.team if isinstance(e.team, str) else None,
            x_m=_num(e.x_m), y_m=_num(e.y_m),
            in_court=bool(e.in_court) if isinstance(e.in_court, bool | np.bool_) else None,
        ))  # fmt: skip
    return out


def tracks(run: Path, sample_fps: float = 5.0) -> S.TracksResponse:
    tr = _parquet(run / "tracks.parquet")
    fps = (_json(run / "video.json") or {}).get("fps", 30.0)
    if tr is None:
        return S.TracksResponse(sample_fps=sample_fps, players={})
    step = max(1, round(fps / sample_fps))
    tr = tr[tr.frame % step == 0]
    out = {}
    for p, g in tr.groupby("player"):
        out[str(int(p))] = S.PlayerTrack(t=g.t.round(2).tolist(), x=g.x.round(2).tolist(),
                                         y=g.y.round(2).tolist())  # fmt: skip
    return S.TracksResponse(sample_fps=fps / step, players=out)


def overlay(run: Path, start: int, end: int) -> S.OverlayResponse:
    fps = (_json(run / "video.json") or {}).get("fps", 30.0)
    end = min(end, start + int(OVERLAY_MAX_S * fps))
    pl = _parquet(run / "players.parquet")
    ball = _parquet(run / "ball.parquet")
    p = pl[pl.frame.between(start, end)] if pl is not None else pd.DataFrame()
    b = ball[ball.frame.between(start, end)] if ball is not None else pd.DataFrame()
    r1 = lambda s: s.round(1).tolist()  # noqa: E731
    return S.OverlayResponse(
        start_frame=start, end_frame=end,
        box_frame=p.frame.astype(int).tolist() if len(p) else [],
        box_player=p.player.astype(int).tolist() if len(p) else [],
        box_x1=r1(p.x1) if len(p) else [], box_y1=r1(p.y1) if len(p) else [],
        box_x2=r1(p.x2) if len(p) else [], box_y2=r1(p.y2) if len(p) else [],
        ball_frame=b.frame.astype(int).tolist() if len(b) else [],
        ball_u=r1(b.u) if len(b) else [], ball_v=r1(b.v) if len(b) else [],
        ball_detected=(b.state == "detected").tolist() if len(b) else [],
    )  # fmt: skip


def highlights(run: Path) -> S.HighlightsResponse:
    out = {}
    for name in ("highlights", "condensed"):
        j = _json(run / f"{name}.json")
        if j and (run / f"{name}.mp4").exists():
            out[name] = S.HighlightsVideo(
                file=f"{name}.mp4", version=int((run / f"{name}.mp4").stat().st_mtime),
                total_s=j.get("total_s", 0.0),
                clips=[S.Clip(**{k: c.get(k) for k in S.Clip.model_fields}) for c in j["clips"]],
            )  # fmt: skip
    return S.HighlightsResponse(**out)
