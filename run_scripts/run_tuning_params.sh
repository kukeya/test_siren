#!/bin/bash
set -e

# 激活环境
eval "$(conda shell.bash hook)"
conda activate m_siren

# 数据源配置
SOURCE_ROOT="exp05_ds24"
RECUR_NUMBER=89

# 上一轮 Checkpoint 路径 (用于微调)
# 假设上一轮是 RECUR_NUMBER - 1
PREV_RECUR=$((RECUR_NUMBER-1))
# 注意：这里假设上一轮的 checkpoint 路径格式如下，请根据实际情况确认
CHECKPOINT_PATH="logs/${SOURCE_ROOT}/${SOURCE_ROOT}_${PREV_RECUR}/checkpoints/model_final.pth"
POINT_CLOUD="mesh/${SOURCE_ROOT}/ruyi_recur${PREV_RECUR}_n_deformed.xyz"

# 输出配置
OUTPUT_ROOT="logs/tuning_${SOURCE_ROOT}_recur${RECUR_NUMBER}"

# 实验名称定义
EXP1_NAME="exp1_finetune_base"
EXP2_NAME="exp2_finetune_low_lr"
EXP3_NAME="exp3_finetune_batch_60k"
EXP4_NAME="exp4_finetune_hybrid"
EXP5_NAME="exp5_finetune_long"
EXP6_NAME="exp6_finetune_high_lr"
EXP7_NAME="exp7_finetune_batch_4k"
EXP8_NAME="exp9_finetune_mse_loss_1w_eph"

echo "=================================================="
echo "SIREN 参数微调测试 (基于上一轮 Checkpoint)"
echo "目标点云: ${POINT_CLOUD}"
echo "加载权重: ${CHECKPOINT_PATH}"
echo "结果存储: ${OUTPUT_ROOT}"
echo "=================================================="

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: 找不到 Checkpoint 文件: $CHECKPOINT_PATH"
    echo "请检查路径是否正确，或者上一轮训练是否完成。"
    exit 1
fi

# 实验 1: 基准微调 (Standard Fine-tune)
echo ">>> Launching Exp 8: ${EXP8_NAME}..."
(sleep 2; python experiment_scripts/train_sdf.py \
    --logging_root "${OUTPUT_ROOT}" \
    --point_cloud_path "${POINT_CLOUD}" \
    --experiment_name "${EXP8_NAME}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --num_epochs 10000 \
    --epochs_til_ckpt 1000 \
    --steps_til_summary 500 \
    --lr 1e-4) &

# 实验 2: 低学习率微调 (Low LR Fine-tune)
# echo ">>> Launching Exp 2: ${EXP2_NAME}..."
# (sleep 12; python experiment_scripts/train_sdf.py \
#     --logging_root "${OUTPUT_ROOT}" \
#     --point_cloud_path "${POINT_CLOUD}" \
#     --experiment_name "${EXP2_NAME}" \
#     --checkpoint_path "${CHECKPOINT_PATH}" \
#     --num_epochs 2000 \
#     --epochs_til_ckpt 1000 \
#     --steps_til_summary 500 \
#     --lr 1e-5) &

# # 实验 3: 大 Batch 微调 (Large Batch Fine-tune)
# echo ">>> Launching Exp 3: ${EXP3_NAME}..."
# (sleep 22; python experiment_scripts/train_sdf.py \
#     --logging_root "${OUTPUT_ROOT}" \
#     --point_cloud_path "${POINT_CLOUD}" \
#     --experiment_name "${EXP3_NAME}" \
#     --checkpoint_path "${CHECKPOINT_PATH}" \
#     --num_epochs 2000 \
#     --epochs_til_ckpt 1000 \
#     --steps_til_summary 500 \
#     --batch_size 60000) &

# # 实验 4: 混合策略微调 (Hybrid Fine-tune)
# echo ">>> Launching Exp 4: ${EXP4_NAME}..."
# (sleep 32; python experiment_scripts/train_sdf.py \
#     --logging_root "${OUTPUT_ROOT}" \
#     --point_cloud_path "${POINT_CLOUD}" \
#     --experiment_name "${EXP4_NAME}" \
#     --checkpoint_path "${CHECKPOINT_PATH}" \
#     --num_epochs 3000 \
#     --epochs_til_ckpt 1000 \
#     --steps_til_summary 500 \
#     --lr 1e-5 \
#     --batch_size 60000) &

# # 实验 5: 较长微调 (Longer Fine-tune)
# EXP5_NAME="exp5_finetune_long"
# echo ">>> Launching Exp 5: ${EXP5_NAME}..."
# (sleep 42; python experiment_scripts/train_sdf.py \
#     --logging_root "${OUTPUT_ROOT}" \
#     --point_cloud_path "${POINT_CLOUD}" \
#     --experiment_name "${EXP5_NAME}" \
#     --checkpoint_path "${CHECKPOINT_PATH}" \
#     --num_epochs 5000 \
#     --epochs_til_ckpt 1000 \
#     --steps_til_summary 500 \
#     --lr 1e-4) &

# # 实验 6: 高学习率冲击 (High LR Shock)
# EXP6_NAME="exp6_finetune_high_lr"
# echo ">>> Launching Exp 6: ${EXP6_NAME}..."
# (sleep 52; python experiment_scripts/train_sdf.py \
#     --logging_root "${OUTPUT_ROOT}" \
#     --point_cloud_path "${POINT_CLOUD}" \
#     --experiment_name "${EXP6_NAME}" \
#     --checkpoint_path "${CHECKPOINT_PATH}" \
#     --num_epochs 1000 \
#     --epochs_til_ckpt 500 \
#     --steps_til_summary 200 \
#     --lr 5e-4) &

# # 实验 7: 小 Batch 扰动 (Small Batch Noise)
# EXP7_NAME="exp7_finetune_batch_4k"
# echo ">>> Launching Exp 7: ${EXP7_NAME}..."
# (sleep 62; python experiment_scripts/train_sdf.py \
#     --logging_root "${OUTPUT_ROOT}" \
#     --point_cloud_path "${POINT_CLOUD}" \
#     --experiment_name "${EXP7_NAME}" \
#     --checkpoint_path "${CHECKPOINT_PATH}" \
#     --num_epochs 2000 \
#     --epochs_til_ckpt 1000 \
#     --steps_til_summary 500 \
#     --batch_size 4000) &

echo "=================================================="
echo "所有训练任务已提交后台运行，正在等待完成..."
wait
echo "所有训练任务完成，开始并行测试..."

# 定义测试函数
run_test() {
    local EXP_NAME=$1
    echo ">>> Testing ${EXP_NAME}..."
    python experiment_scripts/test_sdf.py \
        --logging_root "${OUTPUT_ROOT}" \
        --experiment_name "${OUTPUT_ROOT}/${EXP_NAME}_rc" \
        --checkpoint_path "${OUTPUT_ROOT}/${EXP_NAME}/checkpoints/model_final.pth"
}

# 并行运行测试
run_test "${EXP8_NAME}" &
# run_test "${EXP2_NAME}" &
# run_test "${EXP3_NAME}" &
# run_test "${EXP4_NAME}" &
# run_test "${EXP5_NAME}" &
# run_test "${EXP6_NAME}" &
# run_test "${EXP7_NAME}" &

wait
echo "=================================================="
echo "微调与测试全部完成。"
echo "结果已保存至: ${OUTPUT_ROOT}"
echo "请运行以下命令查看对比结果:"
echo "tensorboard --logdir ${OUTPUT_ROOT}"
echo "=================================================="


# python experiment_scripts/test_sdf.py \
#         --experiment_name "/home/group1/jym/Repos/test_siren/logs/tuning_exp05_ds24_recur89/exp8_finetune_mse_loss_rc" \
#         --checkpoint_path "/home/group1/jym/Repos/test_siren/logs/tuning_exp05_ds24_recur89/exp8_finetune_mse_loss/checkpoints/model_final.pth"