#!/bin/bash
# 启动语义标注 vLLM 服务 (MASt3R-SLAM 语义关键帧标注 + 自然语言导航查询)
# 默认模型: Qwen3.5-35B-A3B-GPTQ-Int4 (MoE 256专家/激活3B, 原生多模态,
#   实测延迟中位 0.60s 比 9B 快 2.2 倍; 地标召回略低于 9B, 由用户实际验收决定取舍)
# 备选: ~/Disk/models/Qwen3.5-9B (bf16, 地标召回/精度最高, 延迟 ~1.3s)
# 部署在 L40 (192.168.50.72), 本机通过 http://192.168.50.72:8299/v1 调用
# 用法: bash start_qwen35_vlm_vllm.sh [GPU_IDS] [PORT] [MODEL_PATH] [SERVED_NAME]
# 默认: GPU=0,1 (TP=2, 每卡约 13GB, 给 Isaac Sim 留足缓冲), PORT=8299

set -e

GPU_IDS=${1:-0,1}
PORT=${2:-8299}
MODEL_PATH=${3:-"$HOME/Disk/models/Qwen3.5-35B-A3B-GPTQ-Int4"}
SERVED_NAME=${4:-qwen3.5-35b-a3b}
CONDA_ENV="qwen3"
TP=$(echo "$GPU_IDS" | awk -F, '{print NF}')

echo "[vLLM] Starting ${MODEL_PATH} on GPU ${GPU_IDS} (TP=${TP}), port ${PORT}..."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} vllm serve "${MODEL_PATH}" \
    --port ${PORT} \
    --tensor-parallel-size ${TP} \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.30 \
    --enable-prefix-caching \
    --served-model-name ${SERVED_NAME} \
    --trust-remote-code \
    --no-enable-log-requests
