"""Replay a BeyondMimic motion .npz on Wato at TRUE real-time speed.

This shows the reference exactly as the policy is fed it: training advances one npz frame
per 0.02 s env step (50 Hz), whatever the npz's own "fps" field says. Unlike replay_npz.py,
which shows one frame per *rendered* frame (so it plays slower or faster depending on your GPU),
this script picks the frame from the wall clock, dropping frames if rendering is slow.
What you see is what one second of training motion looks like in one real second.

Put this file in scripts/ (next to replay_npz.py).

Usage:
    python scripts/replay_npz_realtime.py --registry_name <entity>-org/wandb-registry-motions/wato_boxing
    python scripts/replay_npz_realtime.py --motion_file /path/to/motion.npz
Options:
    --speed 0.5      play at half speed (still exact: 0.5 s of motion per real second)
    --no_loop        stop at the end instead of looping
"""

import argparse
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a motion npz at true real-time speed.")
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--registry_name", type=str, help="WandB registry name of the motion.")
src.add_argument("--motion_file", type=str, help="Local motion.npz path.")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed factor (1.0 = real time).")
parser.add_argument("--no_loop", action="store_true", default=False, help="Do not loop the motion.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.wato import WATO_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader

# The rate the TRAINING env plays the motion at: sim.dt (0.005) * decimation (4) = 0.02 s per frame.
TRAIN_STEP_DT = 0.02


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
    )
    robot: ArticulationCfg = WATO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def get_motion_file() -> str:
    if args_cli.motion_file:
        return args_cli.motion_file
    import pathlib

    import wandb

    name = args_cli.registry_name if ":" in args_cli.registry_name else args_cli.registry_name + ":latest"
    return str(pathlib.Path(wandb.Api().artifact(name).download()) / "motion.npz")


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=TRAIN_STEP_DT))
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot: Articulation = scene["robot"]

    motion_file = get_motion_file()
    npz_fps = float(np.load(motion_file)["fps"].reshape(-1)[0])
    motion = MotionLoader(motion_file, torch.tensor([0], dtype=torch.long, device=sim.device), sim.device)
    n = motion.time_step_total
    duration = n * TRAIN_STEP_DT

    print("=" * 72)
    print(f"Motion file : {motion_file}")
    print(f"Frames      : {n}")
    print(f"npz 'fps'   : {npz_fps:g}  (ignored by training)")
    print(f"Training plays it at {1 / TRAIN_STEP_DT:g} frames/s  ->  {duration:.2f} s of motion")
    if abs(npz_fps - 1 / TRAIN_STEP_DT) > 1e-3:
        print(f"WARNING: npz fps != {1 / TRAIN_STEP_DT:g}; training plays it {(1 / TRAIN_STEP_DT) / npz_fps:.2f}x "
              f"faster than it was saved. Regenerate with --output_fps 50.")
    print("Compare the duration above with the original recording (BVH: Frames x Frame Time).")
    print("=" * 72)

    t0 = time.perf_counter()
    last_report, rendered, last_frame = t0, 0, -1
    while simulation_app.is_running():
        elapsed = (time.perf_counter() - t0) * args_cli.speed
        frame = int(elapsed / TRAIN_STEP_DT)
        if frame >= n:
            if args_cli.no_loop:
                print(f"[INFO] Reached the end: {duration:.2f} s of motion in "
                      f"{time.perf_counter() - t0:.2f} s real time.")
                break
            t0, frame = time.perf_counter(), 0
        f = torch.tensor([frame], dtype=torch.long, device=sim.device)

        root = robot.data.default_root_state.clone()
        root[:, :3] = motion.body_pos_w[f][:, 0] + scene.env_origins[:, None, :]
        root[:, 3:7] = motion.body_quat_w[f][:, 0]
        root[:, 7:10] = motion.body_lin_vel_w[f][:, 0]
        root[:, 10:] = motion.body_ang_vel_w[f][:, 0]
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(motion.joint_pos[f], motion.joint_vel[f])
        scene.write_data_to_sim()
        sim.render()  # kinematic replay, no physics
        scene.update(TRAIN_STEP_DT)

        look = root[0, :3].cpu().numpy()
        sim.set_camera_view(look + np.array([2.0, 2.0, 0.5]), look)

        rendered += 1
        last_frame = frame
        now = time.perf_counter()
        if now - last_report >= 2.0:
            fps = rendered / (now - last_report)
            note = "" if fps * TRAIN_STEP_DT >= args_cli.speed * 0.95 else "  (frames are being skipped to stay real-time)"
            print(f"[t={last_frame * TRAIN_STEP_DT:6.2f}s / {duration:.2f}s]  render {fps:5.1f} fps{note}")
            last_report, rendered = now, 0


if __name__ == "__main__":
    main()
    simulation_app.close()