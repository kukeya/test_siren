#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_scripts/export_meshes_every_1000.sh [OUT_DIR]
#
# Default OUT_DIR:
#   /home/jym/Repos/Experiments/coral_adaptive/debug/iter_59_debug_ep8000_lr2e-5_20260305

OUT_DIR="${1:-/home/jym/Repos/Experiments/coral_adaptive/debug/iter_59_debug_ep8000_lr2e-5_20260305}"
CKPT_DIR="${OUT_DIR}/siren_checkpoints/checkpoints"
MESH_DIR="${OUT_DIR}/meshes_per_ckpt"

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${CKPT_DIR}" >&2
  exit 1
fi

mkdir -p "${MESH_DIR}"

mapfile -t CKPTS < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name 'model_epoch_*.pth' | sort)

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "No model_epoch_*.pth found in ${CKPT_DIR}" >&2
  exit 1
fi

echo "Found ${#CKPTS[@]} epoch checkpoints under: ${CKPT_DIR}"
echo "Output mesh dir: ${MESH_DIR}"

ok_count=0
fail_count=0

for ckpt in "${CKPTS[@]}"; do
  base="$(basename "${ckpt}")"
  ep="$(sed -E 's/model_epoch_([0-9]+)\.pth/\1/' <<< "${base}")"

  if [[ -z "${ep}" || "${ep}" == "${base}" ]]; then
    echo "Skip unparsable checkpoint name: ${base}"
    continue
  fi

  if (( 10#${ep} % 1000 != 0 )); then
    continue
  fi

  echo "Meshing epoch ${ep} from ${base} ..."
  if conda run -n m_siren python experiment_scripts/test_sdf.py \
    --checkpoint_path "${ckpt}" \
    --output_ply "${MESH_DIR}/epoch_${ep}" \
    --experiment_name meshing_ckpt \
    --resolution 512; then
    ((ok_count+=1))
  else
    ((fail_count+=1))
    echo "Failed meshing ${base}" >&2
  fi
done

echo "Meshing completed. success=${ok_count}, failed=${fail_count}"

if (( fail_count > 0 )); then
  exit 2
fi
