#!/bin/bash
# 安装 MASt3R-SLAM 需编译依赖 —— 全程 --no-build-isolation, 让 curope/asmk/pyimgui/lietorch/backend 用环境内 torch 编译
cd /home/jx/codes/MASt3R-SLAM
ENV=/home/jx/miniconda3/envs/mast3r-slam
PY=$ENV/bin/python
export CUDA_HOME=$ENV
GCC14=/home/jx/codes/MASt3R-SLAM/.gcc14
export PATH=$GCC14:$ENV/bin:$PATH          # gcc/g++ -> 14 (CUDA12.8 要求 gcc<=14, 系统是15)
export CC=$GCC14/gcc CXX=$GCC14/g++ CUDAHOSTCXX=$GCC14/g++
export NVCC_PREPEND_FLAGS="-ccbin $GCC14/g++"   # 强制 nvcc 主机编译器用 gcc-14
export CPATH=$ENV/targets/x86_64-linux/include:$CPATH   # lietorch g++ 需找到 cuda.h (conda 头在此)
export TORCH_CUDA_ARCH_LIST="9.0;12.0"   # Hopper + Blackwell(5090 sm_120)
export MAX_JOBS=8
echo "===== nvcc ====="; $ENV/bin/nvcc --version | tail -2
echo "CUDA_HOME=$CUDA_HOME  ARCH=$TORCH_CUDA_ARCH_LIST"

retry(){ for a in 1 2 3 4; do echo "-- 尝试#$a: $*"; "$@" && return 0; echo "-- #$a 失败,5s重试"; sleep 5; done; return 1; }
PIPI(){ $PY -m pip install --no-build-isolation --retries 6 --timeout 180 "$@"; }

echo "===== [0/3] 预装构建工具 (no-build-isolation 依赖 env 内有) ====="
$PY -m pip install --retries 6 --timeout 180 "setuptools>=68" wheel "cython<3" "numpy==1.26.4" 2>&1 | tail -3

echo "===== [1/3] mast3r (编译 curope/asmk) ====="
retry PIPI -e thirdparty/mast3r || { echo "MAST3R_FAIL"; exit 2; }
echo "===== [2/3] in3d (编译 pyimgui) ====="
retry PIPI -e thirdparty/in3d || { echo "IN3D_FAIL"; exit 3; }
echo "===== [3/3] 主包 (lietorch + mast3r_slam_backends 编译) ====="
retry PIPI -e . || { echo "MAIN_FAIL"; exit 4; }

echo "===== 固定 numpy==1.26.4 ====="; $PY -m pip install "numpy==1.26.4" 2>&1 | tail -2

echo "===== viewer GL 软链 (moderngl 要无版本号 libGL.so/libEGL.so, 系统只有 .so.1) ====="
SYS=/usr/lib/x86_64-linux-gnu
for n in libGL.so.1:libGL.so libEGL.so.1:libEGL.so libGLX.so.0:libGLX.so libOpenGL.so.0:libOpenGL.so; do
  src=${n%:*}; dst=${n#*:}
  [ -e "$SYS/$src" ] && ln -sf "$SYS/$src" "$ENV/lib/$dst"
done
echo "  已建 $ENV/lib/{libGL,libEGL,libGLX,libOpenGL}.so (env python RPATH=\$ORIGIN/../lib 可找到)"
echo "ALL_INSTALL_DONE"
