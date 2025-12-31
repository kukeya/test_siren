#!/bin/bash
set -e  # 遇到错误立即退出

eval "$(conda shell.bash hook)"
conda activate m_siren

MODE="all"
RECUR_NUMBER=14
TRAIN_EPOCHS=3000
ORIGIN_EPOCHS=1000

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --recur)
            RECUR_NUMBER="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--mode train|test|all|origin] [--recur NUMBER]"
            exit 1
            ;;
    esac
done

if [[ ! "$MODE" =~ ^(train|test|all|origin)$ ]]; then
    echo "错误: mode 必须是 train, test, all 或 origin"
    echo "用法: $0 [--mode train|test|all|origin] [--recur NUMBER]"
    exit 1
fi

ROOT_NAME="exp11"
BASE_EXP_NAME="${ROOT_NAME}_${RECUR_NUMBER}"
PREV_RECUR=$((RECUR_NUMBER-1))
PREV_EXP_NAME="${ROOT_NAME}_${PREV_RECUR}"
CHECKPOINT_PATH="logs/${ROOT_NAME}/${PREV_EXP_NAME}/checkpoints/model_final.pth"
POINT_CLOUD_PATH="mesh/${ROOT_NAME}/ruyi_recur${RECUR_NUMBER}_n_deformed_w.xyz"
EXP_NAME="${BASE_EXP_NAME}"

run_train_job() {
    local epochs="$1"
    local exp_name="$2"

    local -a cmd=(
        python experiment_scripts/train_sdf_weights.py \
        --point_cloud_path "${POINT_CLOUD_PATH}" \
        --experiment_name "${exp_name}" \
        --num_epochs "${epochs}" \
        --epochs_til_ckpt 500 \
        --steps_til_summary 500 \
        --checkpoint_path "${CHECKPOINT_PATH}"
    )

    echo "执行命令: ${cmd[*]}"

    # if [ -f "${CHECKPOINT_PATH}" ]; then
    #     cmd+=(--checkpoint_path "${CHECKPOINT_PATH}")
    # else
    #     echo "警告: 找不到 checkpoint ${CHECKPOINT_PATH}，将从头训练。"
    # fi

    "${cmd[@]}"
}

relocate_logs() {
    local exp_name="$1"
    local src_dir="logs/${exp_name}"
    local dest_root="logs/${ROOT_NAME}"
    local dest_dir="${dest_root}/${exp_name}"

    if [ -d "${src_dir}" ]; then
        if [ ! -f "${src_dir}/checkpoints/model_final.pth" ]; then
            echo "警告: ${exp_name} 未生成 model_final.pth，训练可能失败。"
        fi
        mkdir -p "${dest_root}"
        rm -rf "${dest_dir}"
        mv "${src_dir}" "${dest_dir}"
    else
        echo "警告: 输出目录 ${src_dir} 不存在。"
    fi
}

relocate_test_logs() {
    local exp_name="$1"
    local src_dir="logs/${exp_name}_rc"
    local dest_root="logs/${ROOT_NAME}"
    local dest_dir="${dest_root}/${exp_name}_rc"

    if [ -d "${src_dir}" ]; then
        mkdir -p "${dest_root}"
        rm -rf "${dest_dir}"
        mv "${src_dir}" "${dest_dir}"
    fi
}

echo "运行模式: $MODE"

echo "训练实验名: ${EXP_NAME}"

echo "点云: ${POINT_CLOUD_PATH}"

if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
    echo "开始训练..."
    run_train_job "${TRAIN_EPOCHS}" "${EXP_NAME}"
    echo "训练完成，整理日志..."
    relocate_logs "${EXP_NAME}"
fi

if [[ "$MODE" == "test" || "$MODE" == "all" ]]; then
    echo "开始测试..."
    CKPT_PATH="logs/${ROOT_NAME}/${EXP_NAME}/checkpoints/model_final.pth"
    if [ -f "${CKPT_PATH}" ]; then
        python experiment_scripts/test_sdf.py \
            --checkpoint_path "${CKPT_PATH}" \
            --experiment_name "${EXP_NAME}_rc"
        echo "测试完成，整理日志..."
        relocate_test_logs "${EXP_NAME}"
    else
        echo "跳过测试: 未找到 checkpoint ${CKPT_PATH}"
    fi
fi

if [[ "$MODE" == "origin" ]]; then
    EXP_NAME="${BASE_EXP_NAME}"
    echo "从原始 checkpoint 开始训练..."
    python experiment_scripts/train_sdf_weights.py \
        --point_cloud_path "mesh/${ROOT_NAME}/ruyi_recur0_n_deformed_w.xyz" \
        --experiment_name "${EXP_NAME}" \
        --checkpoint_path "logs/origin/checkpoints/model_final.pth" \
        --num_epochs ${ORIGIN_EPOCHS} \
        --epochs_til_ckpt 500 \
        --steps_til_summary 500

    if [ -d "logs/${EXP_NAME}" ]; then
        mkdir -p "logs/${ROOT_NAME}"
        rm -rf "logs/${ROOT_NAME}/${EXP_NAME}"
        mv "logs/${EXP_NAME}" "logs/${ROOT_NAME}/${EXP_NAME}"
    fi

    echo "开始测试..."
    python experiment_scripts/test_sdf.py \
        --checkpoint_path "logs/${ROOT_NAME}/${EXP_NAME}/checkpoints/model_final.pth" \
        --experiment_name "${EXP_NAME}_rc"

    if [ -d "logs/${EXP_NAME}_rc" ]; then
        mkdir -p "logs/${ROOT_NAME}"
        rm -rf "logs/${ROOT_NAME}/${EXP_NAME}_rc"
        mv "logs/${EXP_NAME}_rc" "logs/${ROOT_NAME}/${EXP_NAME}_rc"
    fi
fi

conda deactivate

conda activate igr

python experiment_scripts/train_sdf_weights.py \
    --point_cloud_path "mesh/exp14_v2/ruyi_recur0_n_deformed_w.xyz" \
    --experiment_name "exp14_v2_1" \
    --checkpoint_path "logs/exp14_v2/exp14_v2_0/checkpoints/model_final.pth" \
    --num_epochs 4000 \
    --epochs_til_ckpt 500 \
    --steps_til_summary 500

# python experiment_scripts/train_sdf_weights.py \
#     --point_cloud_path "mesh/exp09/ruyi_recur90_n_deformed_w.xyz" \
#     --experiment_name "exp09_test91_e1k" \
#     --checkpoint_path "logs/exp09/exp09_90/checkpoints/model_final.pth" \
#     --num_epochs 1000 \
#     --epochs_til_ckpt 500 \
#     --steps_til_summary 500

# python experiment_scripts/train_sdf_weights.py \
#     --point_cloud_path "mesh/exp09/ruyi_recur90_n_deformed_w.xyz" \
#     --experiment_name "exp09_test91_e3k" \
#     --checkpoint_path "logs/exp09/exp09_90/checkpoints/model_final.pth" \
#     --num_epochs 3000 \
#     --epochs_til_ckpt 500 \
#     --steps_til_summary 500

    
python experiment_scripts/test_sdf.py \
    --checkpoint_path "logs/exp14_v2_1_v2/checkpoints/model_final.pth" \
    --experiment_name "exp14_v2_1_v2_rc"

python experiment_scripts/test_sdf.py \
    --checkpoint_path "logs/exp14_v2_1/checkpoints/model_final.pth" \
    --experiment_name "exp14_v2_1_rc"

# python experiment_scripts/test_sdf.py \
#     --checkpoint_path "logs/exp10_7_v4/checkpoints/model_final.pth" \
#     --experiment_name "exp10_7_v4_rc"


# python experiment_scripts/test_sdf.py \
#     --checkpoint_path "logs/exp09_test91_e1k/checkpoints/model_final.pth" \
#     --experiment_name "exp09_test91_e1k_rc"

# python experiment_scripts/test_sdf.py \
#     --checkpoint_path "logs/exp09_test91_e4k/checkpoints/model_final.pth" \
#     --experiment_name "exp09_test91_e4k_rc"

# python experiment_scripts/check_sdf.py \
#     --checkpoint_path /home/group1/jym/Repos/test_siren/logs/exp14_v2/exp14_v2_0/checkpoints/model_final.pth \
#     --point_cloud_path mesh/exp14_v2/ruyi_recur0_n_deformed_w.xyz


# python experiment_scripts/test_sdf.py \
#     --checkpoint_path "logs/original/checkpoints/model_final.pth" \
#     --experiment_name "original_rc"


# python experiment_scripts/train_sdf.py \
#     --point_cloud_path "mesh/original_pcd_n.xyz" \
#     --experiment_name "origin" \
#     --num_epochs 8000 \
#     --epochs_til_ckpt 1000 \
#     --steps_til_summary 1000

# python experiment_scripts/test_sdf.py \
#     --checkpoint_path "logs/origin/checkpoints/model_final.pth" \
#     --experiment_name "origin_rc"
