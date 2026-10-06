#!/usr/bin/env bash
# Xsens BVH -> GMR pkl -> BeyondMimic CSV -> npz (+ WandB motion registry), for the Wato robot.
#
# Layout (this script sits next to both folders):
#   ./GMR_Wato/                  GMR fork with the "watonomous" robot
#   ./whole_body_tracking/       BeyondMimic with wato.py + scripts/csv_to_npz_wato.py
#
# Usage:
#   ./bvh_to_npz_wato.sh --file_path take.bvh                 # fps read from the BVH
#   ./bvh_to_npz_wato.sh --file_path take.bvh --input_fps 120 # force a frame rate
#   ./bvh_to_npz_wato.sh --file_path take.bvh --name wato_boxing --robot watonomous_legs_only
#
# Outputs: GMR_Wato/<name>.pkl, csv/<name>.csv, and WandB registry motions/<name>.
# Env overrides: GMR_ENV (default gmr), ISAAC_ENV (default isaaclab_test).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GMR_DIR="${SCRIPT_DIR}/GMR_Wato"
WBT_DIR="${SCRIPT_DIR}/whole_body_tracking"
CSV_DIR="${SCRIPT_DIR}/csv"
GMR_ENV="${GMR_ENV:-gmr}"
ISAAC_ENV="${ISAAC_ENV:-isaaclab_test}"

FILE_PATH=""
INPUT_FPS=""
NAME=""
ROBOT="watonomous"
OUTPUT_FPS="50"   # BeyondMimic plays motions at 1 / (sim.dt * decimation) = 50 Hz - keep 50

while [[ $# -gt 0 ]]; do
    case "$1" in
        --file_path)  FILE_PATH="$2"; shift 2 ;;
        --input_fps)  INPUT_FPS="$2"; shift 2 ;;
        --name)       NAME="$2"; shift 2 ;;
        --robot)      ROBOT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -n "$FILE_PATH" ]] || { echo "Error: --file_path is required"; exit 1; }
[[ -f "$FILE_PATH" ]] || { echo "Error: BVH file does not exist: $FILE_PATH"; exit 1; }
[[ -d "$GMR_DIR"   ]] || { echo "Error: GMR directory does not exist: $GMR_DIR"; exit 1; }
[[ -d "$WBT_DIR"   ]] || { echo "Error: whole_body_tracking directory does not exist: $WBT_DIR"; exit 1; }

FILE_PATH="$(realpath "$FILE_PATH")"
BASENAME="${NAME:-$(basename "$FILE_PATH" .bvh)}"

# csv -> npz converter for Wato (falls back to csv_to_npz.py if you replaced the original)
CSV2NPZ="${WBT_DIR}/scripts/csv_to_npz_wato.py"
[[ -f "$CSV2NPZ" ]] || CSV2NPZ="${WBT_DIR}/scripts/csv_to_npz.py"
[[ -f "$CSV2NPZ" ]] || { echo "Error: no csv_to_npz(_wato).py in ${WBT_DIR}/scripts"; exit 1; }

# ------------------------------------------------------------
# Frame rate: read it from the BVH ("Frame Time: 0.0083333" -> 120).
# Rounded, NOT truncated (GMR's int(1/frame_time) turns 60 Hz into 59).
# A wrong --input_fps makes the npz play too fast/slow in training.
# ------------------------------------------------------------
BVH_FPS="$(awk '/Frame Time:/ {gsub("\r",""); printf "%d", 1.0/$NF + 0.5; exit}' "$FILE_PATH")"
if [[ -z "$INPUT_FPS" ]]; then
    [[ -n "$BVH_FPS" ]] || { echo "Error: no 'Frame Time' in the BVH; pass --input_fps"; exit 1; }
    INPUT_FPS="$BVH_FPS"
    echo "Input fps from BVH: ${INPUT_FPS}"
elif [[ -n "$BVH_FPS" && "$BVH_FPS" != "$INPUT_FPS" ]]; then
    echo "WARNING: --input_fps ${INPUT_FPS} but the BVH says ${BVH_FPS} fps -> the motion will play at the wrong speed."
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p "$CSV_DIR"

PKL_FILE="${GMR_DIR}/${BASENAME}.pkl"
CSV_FILE="${CSV_DIR}/${BASENAME}.csv"

# ============================================================
# 1. BVH -> PKL (GMR retargeting onto Wato)
# ============================================================
echo "========================================"
echo "Step 1: BVH -> PKL   (robot: ${ROBOT})"
echo "========================================"
conda activate "$GMR_ENV"
cd "$GMR_DIR"
# PYTHONPATH makes sure THIS GMR fork (with the watonomous robot) is imported,
# not another GMR installed in the env.
PYTHONPATH="${GMR_DIR}${PYTHONPATH:+:$PYTHONPATH}" python scripts/xsens_bvh_to_robot.py \
    --bvh_file "$FILE_PATH" \
    --robot "$ROBOT" \
    --save_path "$PKL_FILE" \
    --scale 0.01 \
    --reset_to_zero \
    --start 3 \
    --bvh_format 3DSM
[[ -f "$PKL_FILE" ]] || { echo "Error: GMR did not create $PKL_FILE"; exit 1; }

# ============================================================
# 2. PKL -> CSV (root pos, root quat xyzw, joints)
# ============================================================
echo "========================================"
echo "Step 2: PKL -> CSV"
echo "========================================"
python "${GMR_DIR}/convert_gmr_xsens.py" --target_file "$PKL_FILE" --output_file "$CSV_FILE"
[[ -f "$CSV_FILE" ]] || { echo "Error: CSV conversion failed"; exit 1; }
NCOL="$(head -1 "$CSV_FILE" | awk -F, '{print NF}')"
echo "CSV columns: ${NCOL} (expect 7 + 28 = 35 for the full Wato)"

# ============================================================
# 3. CSV -> NPZ in Isaac Lab, logged to the WandB registry
# ============================================================
echo "========================================"
echo "Step 3: CSV -> NPZ   (${INPUT_FPS} fps -> ${OUTPUT_FPS} fps)"
echo "========================================"
conda activate "$ISAAC_ENV"
cd "$SCRIPT_DIR"

# Runs in the foreground; it ends when csv_to_npz_wato.py ends.
python "$CSV2NPZ" \
    --input_file "$CSV_FILE" \
    --input_fps "$INPUT_FPS" \
    --output_fps "$OUTPUT_FPS" \
    --output_name "$BASENAME"
