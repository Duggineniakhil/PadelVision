"""Command-line entry point: `padelvision <command>`."""

from __future__ import annotations

import argparse
import json
import sys

from padelvision.models import registry


def _cmd_fetch_models(args: argparse.Namespace) -> int:
    manifest = registry.load_manifest()
    names = args.names or list(manifest)
    unknown = [n for n in names if n not in manifest]
    if unknown:
        print(f"Unknown models: {unknown}. Known: {sorted(manifest)}", file=sys.stderr)
        return 1
    for name in names:
        spec = manifest[name]
        path = registry.fetch(spec, force=args.force)
        status = "verified" if spec.sha256 else "UNPINNED (add sha256 to manifest)"
        print(f"{name}: {path} [{status}]")
        if spec.sha256 is None:
            print(f"  sha256: {registry.sha256_of(path)}")
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    from padelvision.video import probe

    print(json.dumps(probe(args.video).to_dict(), indent=2))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from padelvision.court.calibrate_tool import run

    return 0 if run(args.image, args.out) else 1


def _cmd_analyze(args: argparse.Namespace) -> int:
    from padelvision.run import analyze

    analyze(
        args.video, args.court, args.out, model=args.model, stride=args.stride,
        max_frames=args.max_frames, imgsz=args.imgsz, device=args.device, force=args.force,
    )  # fmt: skip
    return 0


def _cmd_restats(args: argparse.Namespace) -> int:
    from padelvision.run import restats

    restats(args.run, args.court)
    return 0


def _cmd_ball_bootstrap(args: argparse.Namespace) -> int:
    from padelvision.ball.bootstrap import bootstrap

    bootstrap(args.video, args.run, args.max_frames)
    return 0


def _cmd_ball_review_pack(args: argparse.Namespace) -> int:
    from padelvision.ball.bootstrap import make_review_pack

    make_review_pack(args.video, args.run, args.out, args.n_pseudo, args.n_random)
    return 0


def _cmd_label_ball(args: argparse.Namespace) -> int:
    from padelvision.ball.label_tool import run

    run(args.pack)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from pathlib import Path

    import pandas as pd

    from padelvision.court.calibration import CourtCalibration
    from padelvision.render import render_preview

    run_dir = Path(args.run)
    out = render_preview(
        args.video,
        pd.read_parquet(run_dir / "players.parquet"),
        CourtCalibration.load(run_dir / "court.json"),
        args.out or run_dir / "preview.mp4",
        start_s=args.start,
        seconds=args.seconds,
    )
    print(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="padelvision")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-models", help="download and verify model weights")
    p.add_argument("names", nargs="*", help="models to fetch (default: all)")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.set_defaults(func=_cmd_fetch_models)

    p = sub.add_parser("probe", help="print video metadata")
    p.add_argument("video")
    p.set_defaults(func=_cmd_probe)

    p = sub.add_parser("calibrate", help="click court lines/points on a frame (local, GUI)")
    p.add_argument("image", help="a frame from the video, e.g. data/court_frame.png")
    p.add_argument("-o", "--out", default="data/court.json")
    p.set_defaults(func=_cmd_calibrate)

    p = sub.add_parser("analyze", help="run player analytics on a video (GPU recommended)")
    p.add_argument("video")
    p.add_argument("--court", required=True, help="court.json from `padelvision calibrate`")
    p.add_argument("--out", required=True, help="run folder, e.g. runs/match1")
    p.add_argument("--model", default="person-detector")
    p.add_argument("--stride", type=int, default=1, help="analyse every Nth frame")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    p.add_argument("--force", action="store_true", help="recompute cached detections")
    p.set_defaults(func=_cmd_analyze)

    p = sub.add_parser("restats", help="redo identities/stats/heatmaps from cached detections")
    p.add_argument("run", help="run folder with video.json, detections.parquet, stats.json")
    p.add_argument("--court", default=None, help="use a different court.json")
    p.set_defaults(func=_cmd_restats)

    p = sub.add_parser("ball-bootstrap", help="classical ball candidates + pseudo-labels")
    p.add_argument("video")
    p.add_argument("--run", required=True, help="run folder with court.json + detections")
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=_cmd_ball_bootstrap)

    p = sub.add_parser("ball-review-pack", help="sample frames for human ball labelling")
    p.add_argument("video")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-pseudo", type=int, default=150)
    p.add_argument("--n-random", type=int, default=150)
    p.set_defaults(func=_cmd_ball_review_pack)

    p = sub.add_parser("label-ball", help="label a ball review pack (local, GUI)")
    p.add_argument("pack", help="review pack folder with pack.json")
    p.set_defaults(func=_cmd_label_ball)

    p = sub.add_parser("render", help="annotated preview video from a run folder")
    p.add_argument("video")
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--start", type=float, default=0.0, help="start time (s)")
    p.add_argument("--seconds", type=float, default=30.0)
    p.set_defaults(func=_cmd_render)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
