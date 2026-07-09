#!/bin/bash
# 启动 Qwen3.5-9B vLLM 服务 (MASt3R-SLAM 语义关键帧标注 + 自然语言导航查询)
# Qwen3.5-9B 为原生多模态模型 (config.json 含 vision_config, 支持图像输入)
# 部署在 L40 (192.168.50.72), 本机通过 http://192.168.50.72:8299/v1 调用
# 用法: bash start_qwen35_vlm_vllm.sh [GPU_IDS] [PORT]
# 默认: GPU=0,1 (TP=2, 19GB 权重每卡摊 ~9.5GB, 加 KV 每卡占约 14GB,
#       给 Isaac Sim 等已有任务留足缓冲; 单卡剩余 ~20GB 放不下 19GB 权重 + KV)
# 注: 8199 已被 memory-nav 的 Flask 服务占用, 故用 8299

set -e

GPU_IDS=${1:-0,1}
PORT=${2:-8299}
MODEL_PATH="$HOME/Disk/models/Qwen3.5-9B"
CONDA_ENV="qwen3"
TP=$(echo "$GPU_IDS" | awk -F, '{print NF}')

echo "[vLLM] Starting Qwen3.5-9B on GPU ${GPU_IDS} (TP=${TP}), port ${PORT}..."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} vllm serve "${MODEL_PATH}" \
    --port ${PORT} \
    --dtype bfloat16 \
    --tensor-parallel-size ${TP} \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.30 \
    --enable-prefix-caching \
    --served-model-name qwen3.5-9b \
    --trust-remote-code \
    --no-enable-log-requests
