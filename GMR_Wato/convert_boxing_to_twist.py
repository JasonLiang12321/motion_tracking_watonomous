import pickle
import numpy as np
import torch

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.kinematics_model import KinematicsModel


INPUT_FILE = "retargeting_data/g1/251021_04_boxing_120Hz_cm_3DsMax.pkl"
OUTPUT_FILE = "retargeting_data/g1/251021_04_boxing_120Hz_cm_3DsMax_twist.pkl"

print("Loading:", INPUT_FILE)

# ---------------------------------------------------------
# Load original GMR motion
# ---------------------------------------------------------
with open(INPUT_FILE, "rb") as f:
    motion_data = pickle.load(f)

print("Original keys:", motion_data.keys())

root_pos = np.asarray(motion_data["root_pos"], dtype=np.float32)
root_rot = np.asarray(motion_data["root_rot"], dtype=np.float32)
dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)

fps = int(motion_data["fps"])

print("root_pos:", root_pos.shape)
print("root_rot:", root_rot.shape)
print("dof_pos :", dof_pos.shape)
print("fps     :", fps)

# ---------------------------------------------------------
# Initialize GMR
# ---------------------------------------------------------
retarget = GMR(
    src_human="bvh_xsens",
    tgt_robot="unitree_g1",
    actual_human_height=None,
)

print("G1 XML:", retarget.xml_file)

# ---------------------------------------------------------
# Initialize GMR kinematics
# ---------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

kinematics_model = KinematicsModel(
    retarget.xml_file,
    device=device,
)

# ---------------------------------------------------------
# Convert quaternion format exactly as GMR's
# bvh_to_robot_dataset.py does:
#
# [w, x, y, z] -> [x, y, z, w]
# ---------------------------------------------------------
root_rot = root_rot[:, [1, 2, 3, 0]]

num_frames = root_pos.shape[0]

# ---------------------------------------------------------
# Identity root pose
# ---------------------------------------------------------
identity_root_pos = torch.zeros(
    (num_frames, 3),
    device=device,
    dtype=torch.float32,
)

identity_root_rot = torch.zeros(
    (num_frames, 4),
    device=device,
    dtype=torch.float32,
)

identity_root_rot[:, -1] = 1.0

dof_pos_torch = torch.from_numpy(dof_pos).to(
    device=device,
    dtype=torch.float32,
)

# ---------------------------------------------------------
# Generate local body positions
# ---------------------------------------------------------
print("Computing G1 forward kinematics...")

local_body_pos, _ = kinematics_model.forward_kinematics(
    identity_root_pos,
    identity_root_rot,
    dof_pos_torch,
)

body_names = list(kinematics_model.body_names)

local_body_pos = local_body_pos.detach().cpu().numpy()

print("local_body_pos:", local_body_pos.shape)
print("Number of body links:", len(body_names))

# ---------------------------------------------------------
# IMPORTANT:
# Save ONLY native Python objects.
#
# This avoids NumPy 2.x pickle references such as:
#     numpy._core
#
# TWIST can convert these lists back into tensors/arrays
# when loading them.
# ---------------------------------------------------------
twist_motion_data = {
    "fps": int(fps),

    "root_pos": root_pos.tolist(),

    "root_rot": root_rot.tolist(),

    "dof_pos": dof_pos.tolist(),

    "local_body_pos": local_body_pos.tolist(),

    "link_body_list": body_names,
}

# ---------------------------------------------------------
# Save using standard pickle
# ---------------------------------------------------------
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(
        twist_motion_data,
        f,
        protocol=4,
    )

print()
print("==========================================")
print("TWIST-compatible motion created")
print("==========================================")
print("File:", OUTPUT_FILE)
print("fps:", twist_motion_data["fps"])
print("root_pos:", len(twist_motion_data["root_pos"]))
print("root_rot:", len(twist_motion_data["root_rot"]))
print("dof_pos:", len(twist_motion_data["dof_pos"]))
print("local_body_pos:", len(twist_motion_data["local_body_pos"]))
print("link_body_list:", len(twist_motion_data["link_body_list"]))
