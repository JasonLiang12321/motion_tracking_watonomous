"""Play a trained BeyondMimic policy through the WHOLE reference motion, once, start to finish.

Differences from play.py (which is a training-style rollout):
  * starts at frame 0 (or --start_frame), exactly on the reference pose - no random start frame,
    no pose / velocity / joint-position noise at reset
  * no terminations and no timeout reset mid-motion: if the robot falls it keeps trying, and the
    frame where TRAINING would have reset it is reported instead
  * no random pushes unless --pushes (optionally --push_scale / --push_interval)
  * no domain randomization (fixed friction, no CoM shift, no joint-offset noise) unless
    --keep_domain_rand
  * no observation noise unless --keep_noise
  * prints tracking errors for the whole motion and writes them per frame to a CSV

Put this file in scripts/rsl_rl/ (next to play.py and cli_args.py).

Usage:
    python scripts/rsl_rl/play_full_motion.py --task=Tracking-Flat-Wato-v0 \
        --wandb_path=<entity>/<project>/<run_id>
    python scripts/rsl_rl/play_full_motion.py --task=Tracking-Flat-Wato-v0 \
        --load_run <run folder> --motion_file /path/to/motion.npz
Options: --loop, --start_frame N, --stop_on_fail, --video [--headless]
         --pushes [--push_scale 1.0] [--push_interval 1.0 3.0]   random pushes like training
         --keep_domain_rand / --no-keep_domain_rand              (default: keep)
         --keep_noise / --no-keep_noise                          (default: keep)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play an RSL-RL tracking policy through the whole motion.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the whole motion.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file (overrides the run's).")
parser.add_argument("--start_frame", type=int, default=0, help="Motion frame to start from.")
parser.add_argument("--loop", action="store_true", default=False, help="Restart from --start_frame at the end.")
parser.add_argument(
    "--stop_on_fail", action="store_true", default=False,
    help="End the rollout where training would have terminated it (default: keep going).",
)
parser.add_argument("--keep_domain_rand", action=argparse.BooleanOptionalAction, default=True,
                    help="Keep startup randomization (--no-keep_domain_rand to disable).")
parser.add_argument("--keep_noise", action=argparse.BooleanOptionalAction, default=True,
                    help="Keep observation noise (--no-keep_noise to disable).")
parser.add_argument("--pushes", action="store_true", default=False, help="Keep the training's random pushes.")
parser.add_argument("--push_scale", type=float, default=1.0,
                    help="Multiply the training push velocity range (e.g. 0.5 = half as strong, 2.0 = twice).")
parser.add_argument("--push_interval", type=float, nargs=2, default=None, metavar=("MIN_S", "MAX_S"),
                    help="Seconds between pushes (training default: 1.0 3.0).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import numpy as np
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401

# quat_rotate_inverse was renamed quat_apply_inverse in newer Isaac Lab
_quat_apply_inverse = getattr(math_utils, "quat_apply_inverse", None) or math_utils.quat_rotate_inverse


def _load_checkpoint_and_motion(env_cfg, agent_cfg):
    """Same checkpoint/motion lookup as play.py. Returns the checkpoint path."""
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path
        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        wandb_run.file(str(file)).download("./logs/rsl_rl/temp", replace=True)
        resume_path = f"./logs/rsl_rl/temp/{file}"
        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No motion artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if args_cli.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file
    return resume_path


def _configure_for_full_motion(env_cfg, num_steps: int) -> dict:
    """Turn the training env into a clean, single-pass evaluation. Returns the termination settings."""
    ev, obs, term = env_cfg.events, env_cfg.observations, env_cfg.terminations

    # random pushes: off by default, or the training ones (optionally scaled / re-timed)
    if getattr(ev, "push_robot", None) is not None:
        if not args_cli.pushes:
            ev.push_robot = None
        else:
            vr = ev.push_robot.params["velocity_range"]
            ev.push_robot.params["velocity_range"] = {
                k: (lo * args_cli.push_scale, hi * args_cli.push_scale) for k, (lo, hi) in vr.items()
            }
            if args_cli.push_interval is not None:
                ev.push_robot.interval_range_s = tuple(args_cli.push_interval)
            print(f"[INFO]: Pushes ON every {ev.push_robot.interval_range_s} s, "
                  f"velocity range {ev.push_robot.params['velocity_range']}")
    elif args_cli.pushes:
        print("[WARN]: --pushes given but this task has no push_robot event.")

    if not args_cli.keep_domain_rand:
        # fixed friction = same as the ground (1.0), no bounce
        if getattr(ev, "physics_material", None) is not None:
            ev.physics_material.params.update(
                static_friction_range=(1.0, 1.0), dynamic_friction_range=(1.0, 1.0), restitution_range=(0.0, 0.0)
            )
        if getattr(ev, "base_com", None) is not None:
            ev.base_com = None
        # keep this event (it records the nominal default pose) but with zero offset
        if getattr(ev, "add_joint_default_pos", None) is not None:
            ev.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)

    if not args_cli.keep_noise:
        obs.policy.enable_corruption = False

    # start exactly on the reference
    motion = env_cfg.commands.motion
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0.0, 0.0)

    # remember what training would terminate on, then disable it
    fail = {}
    if getattr(term, "anchor_pos", None) is not None:
        fail["anchor_z"] = term.anchor_pos.params["threshold"]
        term.anchor_pos = None
    if getattr(term, "anchor_ori", None) is not None:
        fail["anchor_ori"] = term.anchor_ori.params["threshold"]
        term.anchor_ori = None
    if getattr(term, "ee_body_pos", None) is not None:
        fail["ee_z"] = term.ee_body_pos.params["threshold"]
        fail["ee_bodies"] = list(term.ee_body_pos.params["body_names"])
        term.ee_body_pos = None

    # no timeout before the motion ends
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.episode_length_s = num_steps * step_dt + 5.0
    return fail


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    resume_path = _load_checkpoint_and_motion(env_cfg, agent_cfg)

    # motion length -> number of policy steps to play
    total_frames = int(np.load(env_cfg.commands.motion.motion_file)["joint_pos"].shape[0])
    start = max(0, min(args_cli.start_frame, total_frames - 2))
    num_steps = total_frames - 1 - start
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    print(f"[INFO]: Motion has {total_frames} frames; playing frames {start}..{total_frames - 1} "
          f"({num_steps} steps, {num_steps * step_dt:.1f} s)")

    fail = _configure_for_full_motion(env_cfg, num_steps)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise RuntimeError("Multi-agent environments are not supported.")

    # every reset starts the motion at `start` instead of an adaptively sampled frame
    cmd = env.unwrapped.command_manager.get_term("motion")

    def _fixed_start(env_ids):
        cmd.time_steps[env_ids] = start

    cmd._adaptive_sampling = _fixed_start

    log_dir = os.path.dirname(resume_path)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "full_motion"),
            "step_trigger": lambda step: step == 0,
            "video_length": num_steps,
            "disable_logger": True,
        }
        print(f"[INFO] Recording the whole motion to {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)  # resets the env -> starts at `start`

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    gravity = robot.data.GRAVITY_VEC_W
    ee_idx = [cmd.cfg.body_names.index(n) for n in fail.get("ee_bodies", []) if n in cmd.cfg.body_names]

    csv_path = os.path.join(log_dir, "full_motion_eval.csv")
    run = 0
    while simulation_app.is_running():
        run += 1
        env.reset()
        obs = env.get_observations()
        n = env.num_envs
        first_fail = {k: torch.full((n,), -1, dtype=torch.long, device=env.device) for k in ("anchor_z", "anchor_ori", "ee_z")}
        rows, sums, maxes = [], {}, {}
        stopped_early = False

        for _ in range(num_steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            frame = cmd.time_steps.clone()

            # what training's terminations would have said
            checks = {}
            if "anchor_z" in fail:
                checks["anchor_z"] = (cmd.anchor_pos_w[:, 2] - cmd.robot_anchor_pos_w[:, 2]).abs() > fail["anchor_z"]
            if "anchor_ori" in fail:
                g_ref = _quat_apply_inverse(cmd.anchor_quat_w, gravity)[:, 2]
                g_rob = _quat_apply_inverse(cmd.robot_anchor_quat_w, gravity)[:, 2]
                checks["anchor_ori"] = (g_ref - g_rob).abs() > fail["anchor_ori"]
            if ee_idx:
                ee_err = (cmd.body_pos_relative_w[:, ee_idx, 2] - cmd.robot_body_pos_w[:, ee_idx, 2]).abs()
                checks["ee_z"] = (ee_err > fail["ee_z"]).any(dim=-1)
            for k, hit in checks.items():
                new = hit & (first_fail[k] < 0)
                first_fail[k][new] = frame[new]

            # tracking errors (env 0 to CSV, mean over envs for the summary)
            metrics = {k: v for k, v in cmd.metrics.items() if k.startswith("error_")}
            row = {"frame": int(frame[0]), "time_s": round((int(frame[0]) - start) * step_dt, 3)}
            for k, v in metrics.items():
                row[k] = float(v[0])
                m = float(v.mean())
                sums[k] = sums.get(k, 0.0) + m
                maxes[k] = max(maxes.get(k, 0.0), m)
            row["would_terminate"] = int(any(bool(c[0]) for c in checks.values()))
            rows.append(row)

            if args_cli.stop_on_fail and any(bool(c.all()) for c in checks.values()):
                stopped_early = True
                break
            if bool(dones.any()):  # should not happen, but never play past a reset
                break

        # ---- summary ------------------------------------------------------------
        steps = max(len(rows), 1)
        print("\n" + "=" * 72)
        print(f"Full-motion evaluation, pass {run}: {len(rows)}/{num_steps} steps played"
              + (" (stopped at first failure)" if stopped_early else ""))
        print("-" * 72)
        for k in sorted(sums):
            print(f"  {k:<24} mean {sums[k] / steps:8.4f}   max {maxes[k]:8.4f}")
        print("-" * 72)
        print("  Where training would have reset the episode (first frame per env, -1 = never):")
        for k, v in first_fail.items():
            if k in fail or (k == "ee_z" and ee_idx):
                print(f"    {k:<11} {v.tolist()}")
        ok = [r for r in rows if not r["would_terminate"]]
        print(f"  Frames within all training termination limits (env 0): {len(ok)}/{len(rows)}")
        print("=" * 72 + "\n")

        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[INFO]: Per-frame errors (env 0) written to {csv_path}")

        if args_cli.video or not args_cli.loop:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()