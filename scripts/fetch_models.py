"""Download and verify model weights listed in models/manifest.yaml.

Usage: python scripts/fetch_models.py [name ...] [--force]
Requires the pipeline package: pip install -e pipeline/
"""

import sys

from padelvision.cli import main

if __name__ == "__main__":
    sys.exit(main(["fetch-models", *sys.argv[1:]]))
