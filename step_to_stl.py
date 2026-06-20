#!/usr/bin/env python3
"""
Convert STEP (.step / .stp) files to STL in place or to a target directory.
Useful for MuJoCo, which does not load STEP meshes.

Usage:
  python step_to_stl.py [path]
  python step_to_stl.py path/to/file.step
  python step_to_stl.py path/to/dir
  python step_to_stl.py path/to/dir --out path/to/output_dir

Requires: pip install cadquery
"""

import argparse
import os
import sys


def convert_step_to_stl(
    step_path: str,
    stl_path: str | None = None,
    *,
    tolerance: float = 0.001,
    angular_tolerance: float = 0.1,
    ascii_stl: bool = False,
) -> bool:
    """Convert one STEP file to STL. Returns True on success."""
    try:
        import cadquery as cq
    except ImportError:
        print("cadquery is required: pip install cadquery", file=sys.stderr)
        sys.exit(1)

    if stl_path is None:
        base, _ = os.path.splitext(step_path)
        stl_path = base + ".stl"

    step_path = os.path.abspath(step_path)
    stl_path = os.path.abspath(stl_path)

    if not os.path.isfile(step_path):
        print(f"Not a file: {step_path}", file=sys.stderr)
        return False

    try:
        import_step = getattr(cq.importers, "import_step", getattr(cq.importers, "importStep", None))
        if import_step is None:
            raise RuntimeError("cadquery.importers has no import_step / importStep")
        result = import_step(step_path)
        result.export(
            stl_path,
            tolerance=tolerance,
            angularTolerance=angular_tolerance,
            ascii=ascii_stl,
        )
        print(f"  {os.path.basename(step_path)} -> {stl_path}")
        return True
    except Exception as e:
        print(f"  {step_path}: {e}", file=sys.stderr)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert STEP (.step/.stp) files to STL (e.g. for MuJoCo)."
    )
    ap.add_argument(
        "path",
        nargs="?",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimus", "urdf", "half-leg", "meshes"),
        help="STEP file or directory containing .step/.stp files (default: simulation/optimus/urdf/half-leg/meshes)",
    )
    ap.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output directory (default: same as input; only used when path is a directory)",
    )
    ap.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=0.001,
        help="STL mesh linear tolerance (default: 0.001)",
    )
    ap.add_argument(
        "--angular",
        "-a",
        type=float,
        default=0.1,
        help="STL mesh angular tolerance in degrees (default: 0.1)",
    )
    ap.add_argument(
        "--ascii",
        action="store_true",
        help="Export ASCII STL (default: binary)",
    )
    args = ap.parse_args()

    path = os.path.abspath(args.path)
    out_dir = os.path.abspath(args.out) if args.out else None

    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".step", ".stp"):
            print(f"Not a STEP file: {path}", file=sys.stderr)
            sys.exit(1)
        out_path = None
        if out_dir:
            out_path = os.path.join(out_dir, os.path.basename(path))
            out_path = os.path.splitext(out_path)[0] + ".stl"
        ok = convert_step_to_stl(
            path,
            out_path,
            tolerance=args.tolerance,
            angular_tolerance=args.angular,
            ascii_stl=args.ascii,
        )
        sys.exit(0 if ok else 1)
    elif os.path.isdir(path):
        step_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith((".step", ".stp"))
        ]
        if not step_files:
            print(f"No .step/.stp files in {path}", file=sys.stderr)
            sys.exit(1)
        target_dir = out_dir or path
        if out_dir and not os.path.isdir(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        failed = 0
        for step_path in sorted(step_files):
            base = os.path.basename(step_path)
            name = os.path.splitext(base)[0] + ".stl"
            stl_path = os.path.join(target_dir, name)
            if not convert_step_to_stl(
                step_path,
                stl_path,
                tolerance=args.tolerance,
                angular_tolerance=args.angular,
                ascii_stl=args.ascii,
            ):
                failed += 1
        if failed:
            sys.exit(1)
        print(f"Converted {len(step_files)} file(s) to {target_dir}")
    else:
        print(f"Path not found: {path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
