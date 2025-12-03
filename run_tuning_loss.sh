#!/bin/bash
set -e

# 激活环境
eval "$(conda shell.bash hook)"
conda activate m_siren

# 数据源配置
SOURCE_ROOT="exp05_ds24"
RECUR_NUMBER=89

# 上一轮 Checkpoint 路径 (用于微调)
PREV_RECUR=$((RECUR_NUMBER-1))
CHECKPOINT_PATH="logs/${SOURCE_ROOT}/${SOURCE_ROOT}_${PREV_RECUR}/checkpoints/model_final.pth"
POINT_CLOUD="mesh/${SOURCE_ROOT}/ruyi_recur${PREV_RECUR}_n_deformed.xyz"

# 输出配置
OUTPUT_ROOT="logs/tuning_loss_${SOURCE_ROOT}_recur${RECUR_NUMBER}"

# 实验名称定义
EXP1_NAME="loss_exp1_sdf_3e4"
EXP2_NAME="loss_exp2_sdf_3e5"
EXP3_NAME="loss_exp3_sdf_3e6"
EXP4_NAME="loss_exp4_grad_1e1"
EXP5_NAME="loss_exp5_grad_1e0"
EXP6_NAME="loss_exp6_no_inter"
EXP7_NAME="loss_exp7_balanced"

echo "=================================================="
echo "SIREN Loss 权重调优测试"
echo "目标点云: ${POINT_CLOUD}"
echo "加载权重: ${CHECKPOINT_PATH}"
echo "结果存储: ${OUTPUT_ROOT}"
echo "=================================================="

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: 找不到 Checkpoint 文件: $CHECKPOINT_PATH"
    exit 1
fi

# 实验 1: 基准 (SDF 3e4, Grad 50)
echo ">>> Launching Exp 1: ${EXP1_NAME} (Baseline)..."
(sleep 2; python experiment_scripts/train_sdf.py \
    --logging_root "${OUTPUT_ROOT}" \
    --point_cloud_path "${POINT_CLOUD}" \
    --experiment_name "${EXP1_NAME}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --num_epochs 2000 \
    --epochs_til_ckpt 1000 \
    --steps_til_summary 500 \
    --lr 1e-4 \
    --sdf_weight 3e4 \
    --grad_weight 50) &

# 实验 2: 增强 SDF (SDF 3e5)
echo ">>> Launching Exp 2: ${EXP2_NAME} (SDF 10x)..."
(sleep 12; python experiment_scripts/train_sdf.py \
    --logging_root "${OUTPUT_ROOT}" \
    --point_cloud_path "${POINT_CLOUD}" \
    --experiment_name "${EXP2_NAME}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --num_epochs 2000 \
    --epochs_til_ckpt 1000 \
    --steps_til_summary 500 \
    --lr 1e-4 \
    --sdf_weight 3e5 \
    --grad_weight 50) &

# 实验 3: 极强 SDF (SDF 3e6)
echo ">>> Launching Exp 3: ${EXP3_NAME} (SDF 100x)..."
(sleep 22; python experiment_scripts/train_sdf.py \
    --logging_root "${OUTPUT_ROOT}" \
    --point_cloud_path "${POINT_CLOUD}" \
    --experiment_name "${EXP3_NAME}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --num_epochs 2000 \
    --epochs_til_ckpt 1000 \
    --steps_til_summary 500 \
    --lr 1e-4 \
    --sdf_weight 3e6 \
    --grad_weight 50) &

# 实验 4: 减弱梯度约束 (Grad 10)
echo ">>> Launching Exp 4: ${EXP4_NAME} (Grad 10)..."
(sleep 32; python experiment_scripts/train_sdf.py \
    --logging_root "${OUTPUT_ROOT}" \
    --point_cloud_path "${POINT_CLOUD}" \
    --experiment_name "${EXP4_NAME}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --num_epochs 2000 \
    --epochs_til_ckpt 1000 \
    --steps_til_summary 500 \
    --lr 1e-4 \
    --sdf_weight 3e4 \
    --grad_weight 10) &

# 实验 5: 极弱梯度约束 (Grad 1)
echo ">>> Launching Exp 5: ${EXP5_NAME} (Grad 1)..."
(sleep 42; python experiment_scripts/train_sdf.py \
    --logging_root "${OUTPUT_ROOT}" \
    --point_cloud_path "${POINT_CLOUD}" \
    --experiment_name "${EXP5_NAME}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --num_epochs 2000 \
    --epochs_til_ckpt 1000 \
    --steps_til_summary 500 \
    --lr 1e-4 \
    --sdf_weight 3e4 \
    --grad_weight 1) &

# 实验 6: 移除 Inter 约束 (No Inter)
echo ">>> Launching Exp 6: ${EXP6_NAME} (No Inter)..."
(sleep 52; python experiment_scripts/train_sdf.py \
    --logging_root "${OUTPUT_ROOT}" \
    --point_cloud_path "${POINT_CLOUD}" \
    --experiment_name "${EXP6_NAME}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --num_epochs 2000 \
    --epochs_til_ckpt 1000 \
    --steps_til_summary 500 \
    --lr 1e-4 \
    --sdf_weight 3e4 \
    --inter_weight 0) &

# 实验 7: 综合调整 (SDF 3e5, Grad 10)
echo ">>> Launching Exp 7: ${EXP7_NAME} (Balanced)..."
(sleep 62; python experiment_scripts/train_sdf.py \
    --logging_root "${OUTPUT_ROOT}" \
    --point_cloud_path "${POINT_CLOUD}" \
    --experiment_name "${EXP7_NAME}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --num_epochs 2000 \
    --epochs_til_ckpt 1000 \
    --steps_til_summary 500 \
    --lr 1e-4 \
    --sdf_weight 3e5 \
    --grad_weight 10) &

echo "=================================================="
echo "所有训练任务已提交后台运行，正在等待完成..."
wait
echo "所有训练任务完成，开始并行测试..."

# 定义测试函数
run_test() {
    local EXP_NAME=$1
    echo ">>> Testing ${EXP_NAME}..."
    # 检查 checkpoint 是否存在
    if [ ! -f "${OUTPUT_ROOT}/${EXP_NAME}/checkpoints/model_final.pth" ]; then
        echo "Warning: Checkpoint not found for ${EXP_NAME}, skipping test."
        return
    fi
    
    python experiment_scripts/test_sdf.py \
        --logging_root "${OUTPUT_ROOT}" \
        --experiment_name "${EXP_NAME}_rc" \
        --checkpoint_path "${OUTPUT_ROOT}/${EXP_NAME}/checkpoints/model_final.pth"
}

# 并行运行测试
run_test "${EXP1_NAME}" &
run_test "${EXP2_NAME}" &
run_test "${EXP3_NAME}" &
run_test "${EXP4_NAME}" &
run_test "${EXP5_NAME}" &
run_test "${EXP6_NAME}" &
run_test "${EXP7_NAME}" &

wait
echo "=================================================="
echo "微调与测试全部完成。"
echo "结果已保存至: ${OUTPUT_ROOT}"
echo "请运行以下命令查看对比结果:"
echo "tensorboard --logdir ${OUTPUT_ROOT}"
echo "=================================================="
