#!/bin/bash
set -e  # 遇到错误立即退出

eval "$(conda shell.bash hook)"
conda activate m_siren


MODE="all"
RECUR_NUMBER=14
# === 设置两个不同的 Epoch ===
EPOCH_1=1000
EPOCH_2=3000
# =========================

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

ROOT_NAME="exp05_ds24_w"
# 基础实验名
BASE_EXP_NAME="${ROOT_NAME}_${RECUR_NUMBER}"

echo "运行模式: $MODE"

if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
    # 定义两个实验的具体名称
    EXP_NAME_1="${BASE_EXP_NAME}_ep${EPOCH_1}"
    EXP_NAME_2="${BASE_EXP_NAME}_ep${EPOCH_2}"

    echo "开始并行训练..."
    echo "任务 1: ${EXP_NAME_1} (Epochs: ${EPOCH_1}) on GPU 0"
    echo "任务 2: ${EXP_NAME_2} (Epochs: ${EPOCH_2}) on GPU 1"

    # train 1 (GPU 0)
    CUDA_VISIBLE_DEVICES=0 python experiment_scripts/train_sdf.py         --point_cloud_path "mesh/${ROOT_NAME}/ruyi_recur$((RECUR_NUMBER))_n_deformed_w.xyz"         --experiment_name "${EXP_NAME_1}"         --checkpoint_path "logs/${ROOT_NAME}/${ROOT_NAME}_$((RECUR_NUMBER-1))/checkpoints/model_final.pth"         --num_epochs $EPOCH_1         --epochs_til_ckpt 500         --steps_til_summary 500 &

    # train 2 (GPU 1)
    CUDA_VISIBLE_DEVICES=1 python experiment_scripts/train_sdf.py         --point_cloud_path "mesh/${ROOT_NAME}/ruyi_recur$((RECUR_NUMBER))_n_deformed_w.xyz"         --experiment_name "${EXP_NAME_2}"         --checkpoint_path "logs/${ROOT_NAME}/${ROOT_NAME}_$((RECUR_NUMBER-1))/checkpoints/model_final.pth"         --num_epochs $EPOCH_2         --epochs_til_ckpt 500         --steps_til_summary 500 &

    # 等待所有后台任务完成
    wait
    echo "训练完成，整理日志..."

    # 移动日志 1
    if [ -d "logs/${EXP_NAME_1}" ]; then
        mkdir -p "logs/${ROOT_NAME}"
        rm -rf "logs/${ROOT_NAME}/${EXP_NAME_1}"
        mv "logs/${EXP_NAME_1}" "logs/${ROOT_NAME}/${EXP_NAME_1}"
    fi

    # 移动日志 2
    if [ -d "logs/${EXP_NAME_2}" ]; then
        mkdir -p "logs/${ROOT_NAME}"
        rm -rf "logs/${ROOT_NAME}/${EXP_NAME_2}"
        mv "logs/${EXP_NAME_2}" "logs/${ROOT_NAME}/${EXP_NAME_2}"
    fi
fi

if [[ "$MODE" == "test" || "$MODE" == "all" ]]; then
    EXP_NAME_1="${BASE_EXP_NAME}_ep${EPOCH_1}"
    EXP_NAME_2="${BASE_EXP_NAME}_ep${EPOCH_2}"
    
    echo "开始并行测试..."

    # test 1 (GPU 0)
    CUDA_VISIBLE_DEVICES=0 python experiment_scripts/test_sdf.py         --checkpoint_path "logs/${ROOT_NAME}/${EXP_NAME_1}/checkpoints/model_final.pth"         --experiment_name "${EXP_NAME_1}_rc" &

    # test 2 (GPU 1)
    CUDA_VISIBLE_DEVICES=1 python experiment_scripts/test_sdf.py         --checkpoint_path "logs/${ROOT_NAME}/${EXP_NAME_2}/checkpoints/model_final.pth"         --experiment_name "${EXP_NAME_2}_rc" &

    wait
    echo "测试完成，整理日志..."

    # 移动测试日志 1
    if [ -d "logs/${EXP_NAME_1}_rc" ]; then
        mkdir -p "logs/${ROOT_NAME}"
        rm -rf "logs/${ROOT_NAME}/${EXP_NAME_1}_rc"
        mv "logs/${EXP_NAME_1}_rc" "logs/${ROOT_NAME}/${EXP_NAME_1}_rc"
    fi

    # 移动测试日志 2
    if [ -d "logs/${EXP_NAME_2}_rc" ]; then
        mkdir -p "logs/${ROOT_NAME}"
        rm -rf "logs/${ROOT_NAME}/${EXP_NAME_2}_rc"
        mv "logs/${EXP_NAME_2}_rc" "logs/${ROOT_NAME}/${EXP_NAME_2}_rc"
    fi
fi

if [[ "$MODE" == "origin" ]]; then
    EXP_NAME="${BASE_EXP_NAME}_origin"
    echo "从原始 checkpoint 开始训练..."
    # train from origin
    python experiment_scripts/train_sdf.py         --point_cloud_path "mesh/${ROOT_NAME}/ruyi_recur0_n_deformed.xyz"         --experiment_name "${EXP_NAME}"         --checkpoint_path "logs/origin/checkpoints/model_final.pth"         --num_epochs $EPOCH_1         --epochs_til_ckpt 500         --steps_til_summary 500

    if [ -d "logs/${EXP_NAME}" ]; then
        mkdir -p "logs/${ROOT_NAME}"
        rm -rf "logs/${ROOT_NAME}/${EXP_NAME}"
        mv "logs/${EXP_NAME}" "logs/${ROOT_NAME}/${EXP_NAME}"
    fi

    echo "开始测试..."
    # test
    python experiment_scripts/test_sdf.py         --checkpoint_path "logs/${ROOT_NAME}/${EXP_NAME}/checkpoints/model_final.pth"         --experiment_name "${EXP_NAME}_rc"

    if [ -d "logs/${EXP_NAME}_rc" ]; then
        mkdir -p "logs/${ROOT_NAME}"
        rm -rf "logs/${ROOT_NAME}/${EXP_NAME}_rc"
        mv "logs/${EXP_NAME}_rc" "logs/${ROOT_NAME}/${EXP_NAME}_rc"
    fi
fi


conda deactivate

conda activate igr


    # python experiment_scripts/train_sdf.py         --point_cloud_path "mesh/exp05_ds24/ruyi_recur88_n_deformed.xyz"         --experiment_name exp07_retrain_2         --num_epochs 3000         --epochs_til_ckpt 500         --steps_til_summary 500
    # python experiment_scripts/test_sdf.py         --checkpoint_path "logs/exp07_retrain/checkpoints/model_final.pth"         --experiment_name "exp07_retrain_rc"
