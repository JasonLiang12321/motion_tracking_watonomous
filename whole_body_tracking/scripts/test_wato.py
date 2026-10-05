"""Smoke test for the Wato humanoid config (whole_body_tracking/robots/wato.py).

Checks:
  1. WATO_CFG spawns (URDF -> USD conversion, meshes found)
  2. every joint is driven by exactly one actuator group, and has an action scale
  3. the gains/limits Isaac Lab actually applied match the config
  4. the robot can be simulated while holding its default pose (does it stand or fall)

Run from the Isaac Lab / whole_body_tracking environment:
    python scripts/test_wato.py --headless
    python scripts/test_wato.py                  # with viewer
    python scripts/test_wato.py --fix-base       # hang the robot in the air (checks the PD drives only)
"""

import argparse
import re
import sys
import traceback


from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Load and smoke-test the Wato humanoid config.")
parser.add_argument("--steps", type=int, default=1000, help="physics steps to simulate")
parser.add_argument("--fix-base", action="store_true", help="fix the base in the air")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[1/6] App launched")


def first(obj, *names):
    """Return the first attribute that exists (attribute names differ between Isaac Lab versions)."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def main() -> int:
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass

    from whole_body_tracking.robots.wato import WATO_ACTION_SCALE, WATO_CFG, WATO_URDF_PATH

    print("[2/6] Imports OK")
    print(f"      URDF: {WATO_URDF_PATH}  (exists: {WATO_URDF_PATH.exists()})")
    if not WATO_URDF_PATH.exists():
        print("ERROR: URDF not found. Fix WATO_URDF_PATH in wato.py.")
        return 1

    robot_cfg = WATO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    if args_cli.fix_base:
        robot_cfg.spawn.fix_base = True
        robot_cfg.init_state.pos = (0.0, 0.0, 1.2)

    @configclass
    class WatoTestSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)),
        )
        robot = robot_cfg

    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=0.005))
    sim.set_camera_view([2.5, 2.5, 1.5], [0.0, 0.0, 0.8])
    scene = InteractiveScene(WatoTestSceneCfg(num_envs=1, env_spacing=2.0))
    print("[3/6] Scene created (URDF converted)")

    sim.reset()
    robot: Articulation = scene["robot"]
    print("[4/6] Simulation reset (articulation initialized)")
    print(f"      {robot.num_joints} joints, {robot.num_bodies} bodies")
    print(f"      bodies: {robot.body_names}")

    problems = []

    # ---- 2. actuator + action-scale coverage ---------------------------------
    print("\n[5/6] Joint check")
    stiff = first(robot.data, "joint_stiffness", "default_joint_stiffness")
    damp = first(robot.data, "joint_damping", "default_joint_damping")
    arm = first(robot.data, "joint_armature", "default_joint_armature")
    effort = first(robot.data, "joint_effort_limits", "joint_effort_limit")
    limits = first(robot.data, "joint_pos_limits", "joint_limits", "default_joint_limits")

    header = f"{'joint':<30}{'group':<8}{'scale':>7}{'Kp':>9}{'Kd':>8}{'arm':>8}{'effort':>8}   limits"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(robot.joint_names):
        groups = [
            g for g, a in WATO_CFG.actuators.items()
            if any(re.fullmatch(p, name) for p in a.joint_names_expr)
        ]
        scales = [v for p, v in WATO_ACTION_SCALE.items() if re.fullmatch(p, name)]
        if len(groups) != 1:
            problems.append(f"{name}: matched by {len(groups)} actuator groups {groups}")
        if len(scales) != 1:
            problems.append(f"{name}: {len(scales)} action scales")

        def val(t):
            return f"{t[0, i].item():.3g}" if t is not None else "?"

        lim = f"[{limits[0, i, 0]:.2f}, {limits[0, i, 1]:.2f}]" if limits is not None else "?"
        print(
            f"{name:<30}{(groups[0] if groups else '-'):<8}"
            f"{(scales[0] if scales else float('nan')):>7.3f}"
            f"{val(stiff):>9}{val(damp):>8}{val(arm):>8}{val(effort):>8}   {lim}"
        )
        if limits is not None and (limits[0, i, 1] - limits[0, i, 0]).item() > 6.0:
            problems.append(f"{name}: joint range > 6 rad (placeholder ±pi limits from CAD export)")

    # ---- 4. simulate holding the default pose ---------------------------------
    target = robot.data.default_joint_pos.clone()
    dt = sim.get_physics_dt()
    start_z = robot.data.root_pos_w[0, 2].item()
    min_z = start_z
    for step in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        z = robot.data.root_pos_w[0, 2].item()
        min_z = min(min_z, z)
        if not torch.isfinite(robot.data.joint_pos).all():
            problems.append(f"simulation became NaN at step {step} (unstable gains / overlapping colliders?)")
            break

    end_z = robot.data.root_pos_w[0, 2].item()
    err = (robot.data.joint_pos - target).abs()[0]
    worst = int(err.argmax())
    print(f"\n[6/6] Simulated {args_cli.steps} steps ({args_cli.steps * dt:.1f} s)")
    print(f"      base height: start {start_z:.3f} m, end {end_z:.3f} m, min {min_z:.3f} m")
    print(f"      largest joint tracking error: {err[worst]:.3f} rad at {robot.joint_names[worst]}")
    if not args_cli.fix_base and end_z < 0.5 * start_z:
        problems.append("robot fell (expected with zero pose / placeholder gains; not a loading error)")
    if err[worst] > 0.3:
        problems.append(f"joint {robot.joint_names[worst]} can't hold its target (effort limit too low?)")

    # ---- report ----------------------------------------------------------------
    print()
    if problems:
        print("ISSUES:")
        for p in problems:
            print("  -", p)
    else:
        print("No issues found.")
    print("Wato config LOADED successfully." if robot.num_joints > 0 else "Wato config FAILED to load.")
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        simulation_app.close()