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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
