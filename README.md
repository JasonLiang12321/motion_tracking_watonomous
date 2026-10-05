# Wato Motion Tracking Pipeline

Motion tracking pipeline adapted for the **WATonomous (Wato) humanoid robot**:

- **[GMR](https://github.com/YanjieZe/GMR)** (General Motion Retargeting): retargets human motion onto robot motion.
- **[BeyondMimic `whole_body_tracking`](https://github.com/HybridRobotics/whole_body_tracking)**: physics-aware reinforcement learning, with a custom Wato CSV → NPZ configuration and Wato task registration.

```
Human motion (BVH) ──GMR──▶ robot motion (PKL) ──▶ CSV ──▶ NPZ ──▶ BeyondMimic training
```

## Repository layout

```
.
├── convert_motion.sh        # whole pipeline: BVH -> PKL -> CSV -> NPZ
├── GMR_Wato/                # GMR with the watonomous robot
└── whole_body_tracking/     # BeyondMimic with the Wato robot config and tasks
```

## Installation

### GMR
Go into the GMR folder and follow its installation guide:

```bash
cd GMR_Wato
# follow the installation instructions in GMR_Wato/README.md
```

### whole_body_tracking
Use **Isaac Sim 5.1 + Isaac Lab 2.3**, then follow the instructions in `whole_body_tracking` as is:

```bash
cd whole_body_tracking
# follow the installation instructions in whole_body_tracking/README.md
```

## Usage

### Full pipeline: `convert_motion.sh`

For ease of use, `convert_motion.sh` runs the whole pipeline from **BVH → NPZ** for BeyondMimic training. The final NPZ is uploaded to the WandB motion registry.

```bash
./convert_motion.sh --file_path path/to/motion.bvh
```

| Option | Description |
|---|---|
| `--file_path` | **Required.** Path to the input BVH file |
| `--input_fps` | Frame rate of the BVH. Optional: read from the BVH's `Frame Time` if omitted |
| `--name` | Name of the motion (output files and WandB registry entry). Defaults to the BVH file name |
| `--robot` | GMR robot name. Defaults to `watonomous` |

**Different conda environment names:** the script uses `gmr` for GMR and `isaaclab_test` for Isaac Lab. If yours are named differently, set them when running the script:

```bash
GMR_ENV=my_gmr_env ISAAC_ENV=my_isaaclab_env ./convert_motion.sh --file_path path/to/motion.bvh
```

### Step by step

GMR accepts several types of input motion; **currently only Xsens has been tested**.

#### 1. Xsens BVH → PKL (GMR retargeting)

Make sure the Xsens BVH is exported in **3DSM** format.

```bash
python scripts/xsens_bvh_to_robot.py \
    --bvh_file "$FILE_PATH" \
    --robot watonomous \
    --save_path "$PKL_FILE" \
    --scale 0.01 \
    --reset_to_zero \
    --bvh_format 3DSM
```

Outputs a PKL file retargeted for the Wato robot.

#### 2. PKL → CSV

```bash
python "${GMR_DIR}/convert_gmr_xsens.py" --target_file "$PKL_FILE" --output_file "$CSV_FILE"
```

A custom script that converts the PKL to the CSV format BeyondMimic expects, and also fixes the quaternion ordering.
