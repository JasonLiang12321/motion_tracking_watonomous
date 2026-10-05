#!/usr/bin/env python3
"""
GMR Xsens PKL -> BeyondMimic CSV converter.

For Xsens-generated GMR PKLs, root_rot is stored as WXYZ.
BeyondMimic's CSV convention is XYZW.

This script converts WXYZ -> XYZW INLINE while writing the CSV.
The input PKL is never modified and no intermediate PKL is created.

Examples:
    python convert_gmr_xsens.py --target_file boxing_retargeted.pkl

    python convert_gmr_xsens.py \
        --target_file boxing_retargeted.pkl \
        --output_file boxing_retargeted.csv
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def validate_input(data) -> None:
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected PKL to contain a dict, got {type(data).__name__}"
        )

    required = ("root_pos", "root_rot", "dof_pos")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing required PKL fields: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an Xsens GMR PKL to BeyondMimic CSV, "
            "reordering root quaternion WXYZ -> XYZW inline."
        )
    )

    parser.add_argument(
        "--target_file",
        type=Path,
        required=True,
        help="Xsens/GMR input PKL, e.g. boxing_retargeted.pkl",
    )

    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help=(
            "Output CSV. If omitted, uses the input filename with "
            "a .csv extension."
        ),
    )

    args = parser.parse_args()

    target_file = args.target_file
    if not target_file.exists():
        raise FileNotFoundError(f"Target file does not exist: {target_file}")

    output_file = args.output_file
    if output_file is None:
        output_file = target_file.with_suffix(".csv")

    data = load_pickle(target_file)
    validate_input(data)

    root_pos = np.asarray(data["root_pos"])
    root_rot_wxyz = np.asarray(data["root_rot"])
    dof_pos = np.asarray(data["dof_pos"])

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(
            f"Expected root_pos shape (N, 3), got {root_pos.shape}"
        )

    if root_rot_wxyz.ndim != 2 or root_rot_wxyz.shape[1] != 4:
        raise ValueError(
            f"Expected root_rot shape (N, 4), got {root_rot_wxyz.shape}"
        )

    if dof_pos.ndim != 2:
        raise ValueError(
            f"Expected dof_pos shape (N, D), got {dof_pos.shape}"
        )

    n_frames = root_pos.shape[0]

    if root_rot_wxyz.shape[0] != n_frames:
        raise ValueError(
            "root_pos and root_rot have different frame counts: "
            f"{root_pos.shape[0]} vs {root_rot_wxyz.shape[0]}"
        )

    if dof_pos.shape[0] != n_frames:
        raise ValueError(
            "root_pos and dof_pos have different frame counts: "
            f"{root_pos.shape[0]} vs {dof_pos.shape[0]}"
        )

    # ------------------------------------------------------------------
    # THE IMPORTANT FIX
    #
    # Input PKL:
    #       [w, x, y, z]
    #
    # BeyondMimic CSV:
    #       [x, y, z, w]
    #
    # Reorder ONLY for the CSV. The source PKL is untouched.
    # ------------------------------------------------------------------
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]

    # Preserve the GMR CSV layout:
    #   root position:      3 values
    #   root quaternion:    4 values
    #   joint DOFs:         remaining values
    motion = np.concatenate(
        (root_pos, root_rot_xyzw, dof_pos),
        axis=1,
    )

    # Verify reordering did not alter quaternion magnitude.
    if not np.allclose(
        np.linalg.norm(root_rot_wxyz, axis=1),
        np.linalg.norm(root_rot_xyzw, axis=1),
        atol=1e-6,
    ):
        raise RuntimeError("Quaternion norm changed during WXYZ -> XYZW conversion.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(motion).to_csv(
        output_file,
        index=False,
        header=False,
    )

    print(f"Input PKL : {target_file}")
    print(f"Output CSV: {output_file}")
    print(f"Frames    : {n_frames}")
    print(f"DOFs      : {dof_pos.shape[1]}")
    print("Root quat : WXYZ (PKL) -> XYZW (CSV) [inline conversion]")

    if n_frames:
        print()
        print(f"First WXYZ: {root_rot_wxyz[0]}")
        print(f"First XYZW: {root_rot_xyzw[0]}")


if __name__ == "__main__":
    main()
