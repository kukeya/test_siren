#!/bin/bash
set -e  # 遇到错误立即退出

eval "$(conda shell.bash hook)"
conda activate m_siren

MODE="all"
RECUR_NUMBER=14
TRAIN_EPOCHS=3000
ROOT_NAME="exp11"

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
        --expname)
            ROOT_NAME="$2"
            shift 2
            ;;
        --epochs)
            TRAIN_EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--mode train|test|all] [--recur NUMBER] [--expname NAME] [--epochs 1000|3000]"
            exit 1
            ;;
    esac
done

if [[ ! "$MODE" =~ ^(train|test|all)$ ]]; then
    echo "错误: mode 必须是 train, test 或 all"
    exit 1
fi

if [[ ! "$TRAIN_EPOCHS" =~ ^(1000|3000)$ ]]; then
    echo "错误: epochs 仅支持 1000 或 3000"
    exit 1
fi

BASE_EXP_NAME="${ROOT_NAME}_${RECUR_NUMBER}"
POINT_CLOUD_PATH="mesh/${ROOT_NAME}/ruyi_recur${RECUR_NUMBER}_n_deformed_w.xyz"

# 两个实验名：无衰减 / 有衰减
EXP_NODECAY="${BASE_EXP_NAME}_nodecay_e${TRAIN_EPOCHS}"
EXP_DECAY="${BASE_EXP_NAME}_decay_e${TRAIN_EPOCHS}"

run_train_job() {
    local epochs="$1"
    local exp_name="$2"
    local use_decay="$3"   # "true" or "false"

    local -a cmd=(
        python experiment_scripts/train_sdf.py \
        --point_cloud_path "${POINT_CLOUD_PATH}" \
        --experiment_name "${exp_name}" \
        --num_epochs "${epochs}" \
        --epochs_til_ckpt 500 \
        --steps_til_summary 500 \
        --sdf_weight 1e4
    )

    # 开启学习率衰减
    if [[ "${use_decay}" == "true" ]]; then
        cmd+=(--use_lr_decay)
    fi

    echo "执行命令: ${cmd[*]}"
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
echo "点云: ${POINT_CLOUD_PATH}"
echo "训练轮数: ${TRAIN_EPOCHS}"

if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
    # echo "开始训练（无衰减）..."
    # run_train_job "${TRAIN_EPOCHS}" "${EXP_NODECAY}" "false"
    # echo "整理日志..."
    # relocate_logs "${EXP_NODECAY}"

    echo "开始训练（有衰减）..."
    run_train_job "${TRAIN_EPOCHS}" "${EXP_DECAY}" "true"
    echo "整理日志..."
    relocate_logs "${EXP_DECAY}"
fi

if [[ "$MODE" == "test" || "$MODE" == "all" ]]; then
    for exp_name in "${EXP_NODECAY}" "${EXP_DECAY}"; do
        echo "开始测试: ${exp_name}"
        CKPT_PATH="logs/${ROOT_NAME}/${exp_name}/checkpoints/model_final.pth"
        if [ -f "${CKPT_PATH}" ]; then
            python experiment_scripts/test_sdf.py \
                --checkpoint_path "${CKPT_PATH}" \
                --experiment_name "${exp_name}_rc"
            echo "整理测试日志..."
            relocate_test_logs "${exp_name}"
        else
            echo "跳过测试: 未找到 checkpoint ${CKPT_PATH}"
        fi
    done
fi

conda deactivate