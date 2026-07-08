#!/bin/bash
cd /home/jx/codes/MASt3R-SLAM/checkpoints
B=https://download.europe.naverlabs.com/ComputerVision/MASt3R
for f in \
  MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
  MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl ; do
  echo ">>> 下载 $f"
  wget -q --show-progress -c "$B/$f" -O "$f" 2>&1 | tail -2
done
echo "=== 下载完成, 文件列表 ==="; ls -lh
