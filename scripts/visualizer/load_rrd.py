"""
Load and view a Rerun .rrd recording file.

Requires: pip install rigid-transform-kit[viz]  (rerun-sdk)

Usage:
  python scripts/visualizer/load_rrd.py output/pallet.rrd
  python scripts/visualizer/load_rrd.py -- path1.rrd path2.rrd
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Load .rrd file(s) in Rerun viewer.",
    )
    p.add_argument(
        "rrd",
        type=Path,
        nargs="+",
        metavar="RRD",
        help=".rrd file path(s) to load",
    )
    args = p.parse_args()

    for path in args.rrd:
        if not path.exists():
            print(f"Error: not found: {path}", file=sys.stderr)
            return 1
        if path.suffix.lower() != ".rrd":
            print(f"Warning: expected .rrd extension: {path}", file=sys.stderr)

    try:
        subprocess.run(["rerun", *[str(p) for p in args.rrd]], check=True)
    except FileNotFoundError:
        print(
            "Error: 'rerun' not found. Install with: pip install rigid-transform-kit[viz]",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as e:
        return e.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
