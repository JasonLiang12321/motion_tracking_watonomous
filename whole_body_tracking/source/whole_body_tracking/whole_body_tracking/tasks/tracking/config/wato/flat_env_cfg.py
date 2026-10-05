"""Wato tracking task, mirroring config/g1/flat_env_cfg.py.

Body names assume UrdfFileCfg's default merge_fixed_joints=True: every link
attached by a fixed joint (Torso_1, baseLink__1__1, motor housings, ankle
motor links) is merged into its parent. The torso is rigidly fixed to
base_link (no waist joints), so base_link is the anchor. Check the real list
with scripts/test_wato.py, which prints robot.body_names.
"""

from isaaclab.utils import configclass

from whole_body_tracking.robots.wato import WATO_ACTION_SCALE, WATO_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

# end effectors: feet and wrists (the claw fingers are separate bodies on the wrists)
WATO_FEET = ["left_foot_1", "right_foot_1"]
WATO_HANDS = ["Mirrorlink6__1__1", "link6__1__1"]  # left wrist, right wrist


@configclass
class WatoFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = WATO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = WATO_ACTION_SCALE

        # ---- motion command (same role as the G1 list) ----------------------
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",           # pelvis + torso (G1: pelvis, torso_link)
            "left_hip_r_1",        # G1: left_hip_roll_link
            "left_calf_1",         # G1: left_knee_link
            "left_foot_1",         # G1: left_ankle_roll_link
            "right_hip_r_1",
            "right_calf_1",
            "right_foot_1",
            "Mirrorlink2__1__1",   # left shoulder roll link   (G1: left_shoulder_roll_link)
            "Mirrorlink4__1__1",   # left elbow bend link      (G1: left_elbow_link)
            "Mirrorlink6__1__1",   # left wrist link           (G1: left_wrist_yaw_link)
            "link2__1__1",         # right shoulder roll link
            "link4__1__1",         # right elbow bend link
            "link6__1__1",         # right wrist link
        ]

        # ---- G1 body names hard-coded in tracking_env_cfg.py -----------------
        # events.base_com randomizes the torso CoM ("torso_link" on G1)
        self.events.base_com.params["asset_cfg"].body_names = "base_link"

        # contacts allowed only on feet and hands (G1 regex names its ankle/wrist links)
        allowed = WATO_FEET + WATO_HANDS
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^" + "".join(f"(?!{n}$)" for n in allowed) + r".+$"
        ]

        # terminate if feet/hands drift too far in z from the reference
        self.terminations.ee_body_pos.params["body_names"] = WATO_FEET + WATO_HANDS


@configclass
class WatoFlatWoStateEstimationEnvCfg(WatoFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
