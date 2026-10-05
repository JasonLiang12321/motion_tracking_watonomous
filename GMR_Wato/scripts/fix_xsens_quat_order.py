"""Reorder root_rot in an xsens_bvh_to_robot.py pkl from wxyz to xyzw.

xsens_bvh_to_robot.py saves root_rot straight from MuJoCo qpos (wxyz), but
load_robot_motion (used by vis_robot_motion.py) expects xyzw, like the pkls
from smplx_to_robot.py / bvh_to_robot.py. Everything else is left untouched.
"""
import argparse
import pickle

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="default: <input>_xyzw.pkl")
    args = parser.parse_args()

    output = args.output or args.input.replace(".pkl", "_xyzw.pkl")

    with open(args.input, "rb") as f:
        motion_data = pickle.load(f)

    # wxyz -> xyzw
    motion_data["root_rot"] = np.asarray(motion_data["root_rot"])[:, [1, 2, 3, 0]]

    with open(output, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"Saved to {output}")
