#!/usr/bin/env bash
# =============================================================================
# 在租用的 GPU 机器（Linux）上一键完成：
#   环境自检 -> 依赖安装 -> 数据集校验 -> 训练 -> 逐类评估 -> 产物汇总
#
# 用法（在仓库根目录执行）：
#   bash training/run_on_gpu.sh --data /root/data/dataset_driver_yolo/data.yaml
#
# 常用参数：
#   --data    <path>   数据 yaml（必填）。需含 nc 与 names，类别顺序会被强制校验
#   --model   <spec>   起始权重，默认 yolo11s.pt（COCO 预训练，首次会自动下载）
#                      想从团队早前的权重继续，就传你本地上传的 .pt 路径
#   --epochs  <n>      默认 100
#   --imgsz   <n>      默认 640（与后端推理尺寸一致，不要随意改小）
#   --batch   <n>      默认 16（2080Ti 11GB 足够；显存不足时降到 8）
#   --device  <id>     默认 0
#   --workers <n>      默认 8
#   --name    <str>    实验名，默认 team_ds_full
#   --skip-install     跳过依赖安装（镜像里已装好时用）
#
# 设计原则：**类别顺序不对就拒绝训练**（以退出码 2 失败），绝不静默继续。
# =============================================================================
set -euo pipefail

DATA=""
MODEL="yolo11s.pt"
EPOCHS=100
IMGSZ=640
BATCH=16
DEVICE=0
WORKERS=8
NAME="team_ds_full"
SKIP_INSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)    DATA="$2"; shift 2 ;;
    --model)   MODEL="$2"; shift 2 ;;
    --epochs)  EPOCHS="$2"; shift 2 ;;
    --imgsz)   IMGSZ="$2"; shift 2 ;;
    --batch)   BATCH="$2"; shift 2 ;;
    --device)  DEVICE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --name)    NAME="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "[错误] 未知参数: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATA" ]]; then
  echo "[错误] 必须指定 --data <数据 yaml>"
  echo
  echo "数据集怎么传上来（任选一种）："
  echo "  A) 从你本机上传（在本机 Windows 的 PowerShell 里执行）："
  echo "     scp -r -P <端口> \"<你本机的数据集目录>\" root@<实例IP>:/root/data/"
  echo "  B) 若该数据集有下载链接，直接在实例上下载并解压"
  echo "  C) 用 training/fetch_dataset.py 从 Roboflow 拉公开数据集（需 ROBOFLOW_API_KEY）"
  exit 1
fi

if [[ ! -f "$DATA" ]]; then
  echo "[错误] 找不到数据 yaml: $DATA"; exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="$(command -v python3 || command -v python)"

echo "================================================================================"
echo "第 1 步：环境自检"
echo "================================================================================"
echo "  仓库根目录 : $REPO_ROOT"
echo "  Python     : $($PY --version 2>&1)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  GPU        : /'
else
  echo "  [警告] 未找到 nvidia-smi —— 这台机器可能不是 GPU 实例"
fi

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  echo
  echo "================================================================================"
  echo "第 2 步：依赖检查与安装"
  echo "================================================================================"
  # torch：如果装了但是 CPU 版，需要换成 CUDA 版
  TORCH_OK=$($PY - <<'EOF' 2>/dev/null || echo "no"
try:
    import torch
    print("yes" if torch.cuda.is_available() else "cpu_only")
except Exception:
    print("no")
EOF
)
  echo "  torch 状态 : $TORCH_OK"
  if [[ "$TORCH_OK" != "yes" ]]; then
    echo "  安装 CUDA 版 PyTorch（2080Ti 是 sm_75，cu124 轮子可用）..."
    $PY -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
  fi
  # ultralytics 固定到本机验证过的版本，避免行为漂移
  if ! $PY -c "import ultralytics; print(ultralytics.__version__)" 2>/dev/null | grep -q '^8\.4\.21$'; then
    echo "  安装 ultralytics==8.4.21 ..."
    YOLO_AUTOINSTALL=false $PY -m pip install --quiet "ultralytics==8.4.21"
  fi
  # 训练链路用到的其它依赖
  $PY -m pip install --quiet pyyaml numpy opencv-python-headless 2>/dev/null || true
else
  echo
  echo "第 2 步：跳过依赖安装（--skip-install）"
fi

$PY - <<'EOF'
import torch
print("  torch      :", torch.__version__)
print("  CUDA 可用  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  设备       :", torch.cuda.get_device_name(0))
    print("  显存       : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))
    print("  算力       : sm_%d%d" % torch.cuda.get_device_capability(0))
EOF

echo
echo "================================================================================"
echo "第 3 步：数据集校验（类别顺序不对会在这里就失败）"
echo "================================================================================"
DATASET_DIR="$(cd "$(dirname "$DATA")" && pwd)"
$PY training/prepare_dataset.py --source "$DATASET_DIR" --format yolo --dry-run --allow-background --max-list 8

echo
echo "================================================================================"
echo "第 4 步：训练"
echo "================================================================================"
echo "  起始权重 : $MODEL"
echo "  epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH device=$DEVICE workers=$WORKERS"
echo "  说明：imgsz 保持 640 —— 后端推理也用 640，尺寸不一致会直接掉精度。"
echo
export YOLO_AUTOINSTALL=false          # 防止 ultralytics 自动 pip 装包（本机踩过这个坑）
set +e
$PY training/train.py \
  --data "$DATA" \
  --model "$MODEL" \
  --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" \
  --device "$DEVICE" --workers "$WORKERS" \
  --name "$NAME"
TRAIN_RC=$?
set -e
echo "训练退出码: $TRAIN_RC"
if [[ $TRAIN_RC -ne 0 ]]; then
  echo "[失败] 训练未成功（2 = 类别顺序不一致，1 = 其它错误）。已停止，不会继续评估。"
  exit $TRAIN_RC
fi

echo
echo "================================================================================"
echo "第 5 步：逐类评估 + 类别顺序把关"
echo "================================================================================"
# train.py 把产物写到它自己的 project 下；这里找出最新一次的 weights
BEST="$(find . -path '*/'"$NAME"'/weights/best.pt' -print 2>/dev/null | head -n 1)"
if [[ -z "$BEST" ]]; then
  echo "[警告] 没找到 $NAME/weights/best.pt，跳过评估"
else
  echo "  权重: $BEST"
  set +e
  $PY training/evaluate.py --weights "$BEST" --data "$DATA" --check-backend \
      --json "${BEST%/weights/best.pt}/metrics.json"
  EVAL_RC=$?
  set -e
  echo "评估退出码: $EVAL_RC （3 = 类别顺序与后端不一致）"
fi

echo
echo "================================================================================"
echo "完成。接下来："
echo "================================================================================"
echo "  1) 把权重取回本机（在本机 PowerShell 执行）："
echo "     scp -P <端口> root@<实例IP>:<远端路径>/best.pt  \"<你本机的项目目录>\\models\\weights\\best_new.pt\""
echo "  2) 别忘了同时取回 metrics.json —— 那是你自己训练的真实指标，软著/答辩要用。"
echo "  3) 本机替换前先备份："
echo "     Copy-Item models\\weights\\best.pt models\\weights\\best.prev.bak.pt"
echo "  4) 替换后跑一遍回归：python -m pytest tests -q"
echo "  5) 关闭实例，别让它空转计费。"
echo
echo "  ⚠️ 记录清楚写进报告：数据集为本团队自建，训练起点为 COCO 预训练 yolo11s.pt，"
echo "     训练轮数 $EPOCHS、imgsz $IMGSZ、batch $BATCH、设备为租用 GPU。"
echo "     AGPL-3.0 义务不因自训而消失（ultralytics 生态要求）。"