"""
main.py — CLI entry point (unchanged behaviour)

Runs the full PadelVision pipeline on a local video file.
Now delegates all processing to pipeline/processor.py.
"""

import sys
import os
import uuid

sys.path.append(os.path.dirname(__file__))

from pipeline.processor import process_video


def main():
    input_path = "input_videos/input_video1.mp4"
    job_id = "cli_run_" + str(uuid.uuid4())[:8]

    print(f"Starting analysis for: {input_path}  (job: {job_id})")

    use_stubs = "--stubs" in sys.argv

    result = process_video(
        input_path=input_path,
        job_id=job_id,
        player_stub="tracker_stubs/player_detections.pkl" if use_stubs else None,
        ball_stub="tracker_stubs/ball_detections.pkl" if use_stubs else None,
        use_stubs=use_stubs,
        on_progress=lambda stage, pct: print(f"  [{pct:>3}%] {stage}"),
    )

    print("\nDone!")
    print(f"  Video   → {result['video_path']}")
    print(f"  Stats   → {result['analysis_path']}")
    print(f"  Heatmap → {result['heatmap_paths']}")
    print(f"  ShotMap → {result['shot_map_path']}")
    print(f"  Highlights: {len(result['highlights'])} detected")


if __name__ == "__main__":
    main()