#!/bin/bash
set -e  # 遇到错误立即退出

eval "$(conda shell.bash hook)"
conda activate m_siren

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

MODE="all"
RECUR_NUMBER=1
EPOCH=1000
NUM_GPUS=7

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
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--mode train|test|all|origin] [--recur NUMBER] [--gpus NUMBER]"
            exit 1
            ;;
    esac
done

if [[ ! "$MODE" =~ ^(train|test|all|origin)$ ]]; then
    echo "错误: mode 必须是 train, test, all 或 origin"
    echo "用法: $0 [--mode train|test|all|origin] [--recur NUMBER] [--gpus NUMBER]"
    exit 1
fi

ROOT_NAME="exp06_hl512_parl"
EXP_NAME="${ROOT_NAME}_${RECUR_NUMBER}"

echo "运行模式: $MODE"
echo "使用 GPU 数量: $NUM_GPUS"

if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
    # train
    python -m torch.distributed.run --nproc_per_node=$NUM_GPUS experiment_scripts/train_sdf_parallel.py \
        --batch_size 15000 \
        --point_cloud_path "mesh/${ROOT_NAME}/ruyi_recur$((RECUR_NUMBER))_n_deformed_w.xyz" \
        --experiment_name "${EXP_NAME}" \
        --checkpoint_path "logs/${ROOT_NAME}/${ROOT_NAME}_$((RECUR_NUMBER-1))/checkpoints/model_final.pth" \
        --num_epochs $EPOCH \
        --epochs_til_ckpt 500 \
        --steps_til_summary 500 \
        --hidden_features 512 \
        --num_hidden_layers 5


    if [ -d "logs/${EXP_NAME}" ]; then
        rm -rf "logs/${ROOT_NAME}/${EXP_NAME}"
        mv "logs/${EXP_NAME}" "logs/${ROOT_NAME}/${EXP_NAME}"
    else
        echo "警告: logs/${EXP_NAME} 不存在,跳过移动"
    fi
fi

if [[ "$MODE" == "test" || "$MODE" == "all" ]]; then
    # test script for experiment
    python experiment_scripts/test_sdf.py \
        --checkpoint_path "logs/${ROOT_NAME}/${EXP_NAME}/checkpoints/model_final.pth" \
        --experiment_name "${EXP_NAME}_rc" \
        --hidden_features 512 \
        --num_hidden_layers 5

    if [ -d "logs/${EXP_NAME}_rc" ]; then
        rm -rf "logs/${ROOT_NAME}/${EXP_NAME}_rc"
        mv "logs/${EXP_NAME}_rc" "logs/${ROOT_NAME}/${EXP_NAME}_rc"
    else
        echo "警告: logs/${EXP_NAME}_rc 不存在,跳过移动"
    fi
fi

if [[ "$MODE" == "origin" ]]; then
    echo "从原始 checkpoint 开始训练..."
    # train from origin
    torchrun --nproc_per_node=$NUM_GPUS experiment_scripts/train_sdf_parallel.py \
        --point_cloud_path "mesh/${ROOT_NAME}/ruyi_recur0_n_deformed.xyz" \
        --experiment_name "${EXP_NAME}" \
        --checkpoint_path "logs/origin/checkpoints/model_final.pth" \
        --num_epochs $EPOCH \
        --epochs_til_ckpt 500 \
        --steps_til_summary 500

    if [ -d "logs/${EXP_NAME}" ]; then
        rm -rf "logs/${ROOT_NAME}/${EXP_NAME}"
        mv "logs/${EXP_NAME}" "logs/${ROOT_NAME}/${EXP_NAME}"
    else
        echo "警告: logs/${EXP_NAME} 不存在,跳过移动"
    fi

    echo "开始测试..."
    # test
    python experiment_scripts/test_sdf.py \
        --checkpoint_path "logs/${ROOT_NAME}/${EXP_NAME}/checkpoints/model_final.pth" \
        --experiment_name "${EXP_NAME}_rc" \
        --hidden_features 512 \
        --num_hidden_layers 5
        
    if [ -d "logs/${EXP_NAME}_rc" ]; then
        rm -rf "logs/${ROOT_NAME}/${EXP_NAME}_rc"
        mv "logs/${EXP_NAME}_rc" "logs/${ROOT_NAME}/${EXP_NAME}_rc"
    else
        echo "警告: logs/${EXP_NAME}_rc 不存在,跳过移动"
    fi
fi

# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

# python -m torch.distributed.run --nproc_per_node=7 experiment_scripts/train_sdf_parallel.py \
# >     --point_cloud_path mesh/ruyi.xyz \
# >     --experiment_name ddp_test \
# >     --batch_size 1000 \
# >     --num_epochs 10 \
# >     --steps_til_summary 5 \
# >     --epochs_til_ckpt 10