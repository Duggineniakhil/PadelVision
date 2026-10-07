# PadelVision V2

Computer-vision analytics for padel match videos: player movement, positioning and
heatmaps, then rallies and shots. A personal project, rebuilt from scratch.
See [docs/V2_PLAN.md](docs/V2_PLAN.md) for the plan.

**Status:** Phase 0 (foundations) done. Next: phase 1, player analytics.

## Local setup (development and tests, no GPU needed)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e "pipeline[dev]"   # macOS/Linux: .venv/bin/python
.venv/Scripts/python -m pytest pipeline
.venv/Scripts/python scripts/fetch_models.py
```

## Running on real videos (Kaggle / Colab)
Open [ml/notebooks/00_setup_check.ipynb](ml/notebooks/00_setup_check.ipynb) in Kaggle or Colab,
turn on a GPU, set `VIDEO_PATH`, and run all cells.

## Layout
```
pipeline/   padelvision package (CV + analytics core) and its tests
models/     manifest.yaml: every model with URL + sha256 (weights are downloaded, never committed)
scripts/    fetch_models.py
ml/         Kaggle/Colab notebooks and the eval-set spec
docs/       plan and design notes
```
