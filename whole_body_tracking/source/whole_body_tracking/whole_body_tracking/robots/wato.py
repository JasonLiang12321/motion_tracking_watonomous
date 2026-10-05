"""Isaac Lab configuration for the Wato humanoid.

Structured like BeyondMimic's robots/g1.py:
  * URDF spawned with UrdfFileCfg (converted to USD on launch)
  * URDF joint drives zeroed; real PD gains come from the actuators
  * per-motor armature -> stiffness/damping for a 10 Hz, damping-ratio-2 joint
  * WATO_ACTION_SCALE = 0.25 * effort_limit / stiffness per joint

Motor specs come from the WATonomous humanoid repo (hardware-confirmed) and the
motor vendors' datasheets. Source for each value is noted next to it:
  [WB]   src/pioneer_humanoid/pioneer_humanoid/whole_body.py  (leg effort/velocity)
  [SP]   .../locomotion/SESSION_PROGRESS_2026-07-13.md       (leg torques, ankle = 60 Nm)
  [BA]   src/pioneer_humanoid/pioneer_humanoid/bimanual_arm.py (arm/gripper limits)
  [MIT]  src/interfacing/can/config/mit_profiles.yaml         (CAN command ranges)
  [CM]   CubeMars product pages (rotor inertia, gear ratio)
  [EST]  ESTIMATE - no published value found; replace when measured / datasheet found

ARMATURE is the reflected rotor inertia at the joint: rotor_inertia * gear_ratio^2.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

WATO_URDF_PATH = (
    Path(__file__).resolve().parents[4]
    / "whole_body_humanoid"
    / "urdf"
    / "whole_body_humanoid_raw_export.urdf" 
)

# -----------------------------------------------------------------------------
# Motor specs
#   ARMATURE [kg m^2]   EFFORT [N m] (N for claws)   VELOCITY [rad/s] (m/s for claws)
# -----------------------------------------------------------------------------
# CubeMars AK10-9: rotor 1002 g cm^2, 9:1  -> 1.002e-4 * 81            [CM]
ARMATURE_AK10 = 0.0081
# CubeMars AK80-9: rotor 607 g cm^2, 9:1   -> 6.07e-5 * 81             [CM]
ARMATURE_AK80 = 0.0049
# RobStride RS03 (60 Nm, 9:1) / RS04 (120 Nm, 9:1): rotor inertia is not
# published. Scaled from the AK10-9 (48 Nm, 9:1 QDD) by peak torque.   [EST]
ARMATURE_RS03 = 0.012
ARMATURE_RS04 = 0.025
# CubeMars AKH70-48 (48:1): rotor inertia not published; the high ratio
# makes the reflected inertia large (~130 g cm^2 * 48^2).               [EST]
ARMATURE_AKH70 = 0.030
# CubeMars GL40 KV70 gimbal motor (low ratio, tiny rotor).              [EST]
ARMATURE_GL40 = 0.0005

EFFORT_AKH70 = 222.0   # [WB][SP] hip pitch + knee, AKH70-48 peak
EFFORT_RS04 = 120.0    # [WB][SP] RS04 peak
EFFORT_RS03 = 60.0     # [WB][SP] RS03 peak; ankle confirmed 60 Nm (not 120)
EFFORT_AK10 = 53.0     # [BA] AK10-9 V3.0 peak (MIT range +-54 [MIT])
EFFORT_AK80 = 18.0     # [MIT] CAN command range is +-18 Nm (datasheet peak 22 [BA]/18 [CM])
EFFORT_GL40 = 0.73     # [BA] GL40 KV70 peak
EFFORT_CLAW = 30.0     # [BA] _GRIPPER_EFFORT_LIMIT (N)

VELOCITY_AKH70 = 3.6652  # [WB] 35 rpm output - SLOW, limits fast knee/hip-pitch motion
VELOCITY_RS04 = 20.944   # [WB] 200 rpm
VELOCITY_RS03 = 20.42    # [WB] 195 rpm (RobStride no-load spec)
VELOCITY_AK10 = 6.0      # [BA] sim cap used by the team (hardware no-load ~33 rad/s [CM])
VELOCITY_AK80 = 6.0      # [BA] sim cap used by the team (hardware no-load ~56 rad/s [CM])
VELOCITY_GL40 = 6.0      # [BA]
VELOCITY_CLAW = 0.2      # [BA] _GRIPPER_VELOCITY_LIMIT (m/s)

# Claws are position-held grippers, not motion-tracked: use the team's
# hand-tuned PD instead of the armature formula.                      [BA]
STIFFNESS_CLAW = 400.0
DAMPING_CLAW = 40.0
ARMATURE_CLAW = 0.001  # [EST]

# Same gain design as g1.py: every joint behaves like a 10 Hz, overdamped spring.
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_RS04 = ARMATURE_RS04 * NATURAL_FREQ**2
STIFFNESS_RS03 = ARMATURE_RS03 * NATURAL_FREQ**2
STIFFNESS_AKH70 = ARMATURE_AKH70 * NATURAL_FREQ**2
STIFFNESS_AK10 = ARMATURE_AK10 * NATURAL_FREQ**2
STIFFNESS_AK80 = ARMATURE_AK80 * NATURAL_FREQ**2
STIFFNESS_GL40 = ARMATURE_GL40 * NATURAL_FREQ**2

DAMPING_RS04 = 2.0 * DAMPING_RATIO * ARMATURE_RS04 * NATURAL_FREQ
DAMPING_RS03 = 2.0 * DAMPING_RATIO * ARMATURE_RS03 * NATURAL_FREQ
DAMPING_AKH70 = 2.0 * DAMPING_RATIO * ARMATURE_AKH70 * NATURAL_FREQ
DAMPING_AK10 = 2.0 * DAMPING_RATIO * ARMATURE_AK10 * NATURAL_FREQ
DAMPING_AK80 = 2.0 * DAMPING_RATIO * ARMATURE_AK80 * NATURAL_FREQ
DAMPING_GL40 = 2.0 * DAMPING_RATIO * ARMATURE_GL40 * NATURAL_FREQ

WATO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=str(WATO_URDF_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Kept True (as in g1.py). Note convex-hull colliders overlap at
            # base_link<->hip_r and calf<->foot; see collision notes.
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # base_link (pelvis) to sole is ~0.84 m in the URDF; spawn slightly above.
        pos=(0.0, 0.0, 0.86),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_a_akh70",
                ".*_hip_r_rs04",
                ".*_thigh_rs03",
                ".*_knee_akh70",
            ],
            effort_limit_sim={
                ".*_hip_a_akh70": EFFORT_AKH70,
                ".*_hip_r_rs04": EFFORT_RS04,
                ".*_thigh_rs03": EFFORT_RS03,
                ".*_knee_akh70": EFFORT_AKH70,
            },
            velocity_limit_sim={
                ".*_hip_a_akh70": VELOCITY_AKH70,
                ".*_hip_r_rs04": VELOCITY_RS04,
                ".*_thigh_rs03": VELOCITY_RS03,
                ".*_knee_akh70": VELOCITY_AKH70,
            },
            stiffness={
                ".*_hip_a_akh70": STIFFNESS_AKH70,
                ".*_hip_r_rs04": STIFFNESS_RS04,
                ".*_thigh_rs03": STIFFNESS_RS03,
                ".*_knee_akh70": STIFFNESS_AKH70,
            },
            damping={
                ".*_hip_a_akh70": DAMPING_AKH70,
                ".*_hip_r_rs04": DAMPING_RS04,
                ".*_thigh_rs03": DAMPING_RS03,
                ".*_knee_akh70": DAMPING_AKH70,
            },
            armature={
                ".*_hip_a_akh70": ARMATURE_AKH70,
                ".*_hip_r_rs04": ARMATURE_RS04,
                ".*_thigh_rs03": ARMATURE_RS03,
                ".*_knee_akh70": ARMATURE_AKH70,
            },
        ),
        # Ankle = parallel linkage driven by two RS03s (the *_ankle_rs03_top/bottom
        # links). Modelled as two serial joints; like g1.py's parallel ankle,
        # each joint gets 2x one motor's gains/armature. Effort stays at one
        # motor's 60 Nm per joint, matching the team's hardware-confirmed ankle [SP].
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=EFFORT_RS03,
            velocity_limit_sim=VELOCITY_RS03,
            joint_names_expr=[".*_foot_joint_simMotor1", ".*_foot_joint_simMotor2"],
            stiffness=2.0 * STIFFNESS_RS03,
            damping=2.0 * DAMPING_RS03,
            armature=2.0 * ARMATURE_RS03,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_ak10_1_pitch",
                ".*_shoulder_ak10_2_roll",
                ".*_elbow_ak80_1_yaw",
                ".*_elbow_ak80_2_bend",
                ".*_elbow_ak80_3_forearm",
                ".*_wrist_gl40",
            ],
            effort_limit_sim={
                ".*_shoulder_ak10_1_pitch": EFFORT_AK10,
                ".*_shoulder_ak10_2_roll": EFFORT_AK10,
                ".*_elbow_ak80_1_yaw": EFFORT_AK80,
                ".*_elbow_ak80_2_bend": EFFORT_AK80,
                ".*_elbow_ak80_3_forearm": EFFORT_AK80,
                ".*_wrist_gl40": EFFORT_GL40,
            },
            velocity_limit_sim={
                ".*_shoulder_ak10_1_pitch": VELOCITY_AK10,
                ".*_shoulder_ak10_2_roll": VELOCITY_AK10,
                ".*_elbow_ak80_1_yaw": VELOCITY_AK80,
                ".*_elbow_ak80_2_bend": VELOCITY_AK80,
                ".*_elbow_ak80_3_forearm": VELOCITY_AK80,
                ".*_wrist_gl40": VELOCITY_GL40,
            },
            stiffness={
                ".*_shoulder_ak10_1_pitch": STIFFNESS_AK10,
                ".*_shoulder_ak10_2_roll": STIFFNESS_AK10,
                ".*_elbow_ak80_1_yaw": STIFFNESS_AK80,
                ".*_elbow_ak80_2_bend": STIFFNESS_AK80,
                ".*_elbow_ak80_3_forearm": STIFFNESS_AK80,
                ".*_wrist_gl40": STIFFNESS_GL40,
            },
            damping={
                ".*_shoulder_ak10_1_pitch": DAMPING_AK10,
                ".*_shoulder_ak10_2_roll": DAMPING_AK10,
                ".*_elbow_ak80_1_yaw": DAMPING_AK80,
                ".*_elbow_ak80_2_bend": DAMPING_AK80,
                ".*_elbow_ak80_3_forearm": DAMPING_AK80,
                ".*_wrist_gl40": DAMPING_GL40,
            },
            armature={
                ".*_shoulder_ak10_1_pitch": ARMATURE_AK10,
                ".*_shoulder_ak10_2_roll": ARMATURE_AK10,
                ".*_elbow_ak80_1_yaw": ARMATURE_AK80,
                ".*_elbow_ak80_2_bend": ARMATURE_AK80,
                ".*_elbow_ak80_3_forearm": ARMATURE_AK80,
                ".*_wrist_gl40": ARMATURE_GL40,
            },
        ),
        # Prismatic grippers (no equivalent in g1.py). Motion data has no claw
        # targets; alternatively make them type="fixed" in the URDF.
        "claws": ImplicitActuatorCfg(
            effort_limit_sim=EFFORT_CLAW,
            velocity_limit_sim=VELOCITY_CLAW,
            joint_names_expr=[".*_claw_1", ".*_claw_2"],
            stiffness=STIFFNESS_CLAW,
            damping=DAMPING_CLAW,
            armature=ARMATURE_CLAW,
        ),
    },
)

WATO_ACTION_SCALE = {}
for a in WATO_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            WATO_ACTION_SCALE[n] = 0.25 * e[n] / s[n]