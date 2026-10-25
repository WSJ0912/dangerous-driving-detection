#!/usr/bin/env bash
# ============================================================
#  智慧驭行驾驶行为检测系统 - Linux / macOS 启动脚本
#  用法：bash scripts/start.sh
#  行为：优先使用项目内虚拟环境（.venv / backend/.venv），否则回退 python3；
#        检查依赖与模型权重后启动 python backend/app.py
#
#  注意：本文件使用 LF 换行且**不能带 BOM**，否则 shebang 会失效。
# ============================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================"
echo "  智慧驭行驾驶行为检测系统"
echo "================================================"
echo ""

# ---- 1) 选择解释器：能 import flask 的项目内 venv 优先 ----
PY=""
for cand in "$ROOT/.venv/bin/python" "$ROOT/backend/.venv/bin/python"; do
    if [ -x "$cand" ] && "$cand" -c "import flask" >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done

if [ -z "$PY" ]; then
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1; then
            PY="$cand"
            break
        fi
    done
fi

if [ -z "$PY" ]; then
    echo "[错误] 未找到可用的 Python 解释器。"
    echo "       请安装 Python 3.10+，或创建项目内虚拟环境："
    echo "         python3 -m venv .venv"
    echo "         .venv/bin/python -m pip install -r requirements.txt"
    exit 1
fi

echo "[信息] 使用解释器: $PY"

# ---- 2) 检查运行依赖 ----
if ! "$PY" -c "import flask, cv2, numpy" >/dev/null 2>&1; then
    echo "[错误] 运行依赖不完整（需要 flask / opencv-python / numpy）。请执行："
    echo "         $PY -m pip install -r $ROOT/requirements.txt"
    exit 1
fi

# ---- 3) 检查模型权重 ----
WEIGHTS="$ROOT/models/weights/best.pt"
EXPECTED_SIZE=19161683
if [ -f "$WEIGHTS" ]; then
    SIZE="$(wc -c < "$WEIGHTS" | tr -d ' ')"
    echo "[信息] 模型权重: models/weights/best.pt ($SIZE 字节)"
    if [ "$SIZE" != "$EXPECTED_SIZE" ]; then
        echo "[警告] 文件大小与文档记录（$EXPECTED_SIZE 字节）不一致，请核对来源与 SHA256（见 models/README.md）。"
    fi
else
    echo "[警告] 未找到 models/weights/best.pt —— 服务仍可启动，但 /api/detect 无法检测。"
    echo "        请放置 8 类 YOLO11s 权重（说明见 models/README.md），SHA256 前缀 E69680C7。"
fi

echo ""
echo "服务地址: http://127.0.0.1:5000"
echo "按 Ctrl+C 停止服务"
echo ""

# ---- 4) 启动服务 ----
cd "$ROOT/backend" || exit 1
exec "$PY" app.py