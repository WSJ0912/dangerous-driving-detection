# -*- coding: utf-8 -*-
"""真实权重冒烟测试：覆盖 ``/api/detect`` 的完整推理链路

为什么需要这个文件
------------------
其余用例都把检测器桩掉了（为了不依赖 torch 与权重），所以
"letterbox -> 推理 -> 坐标换算回原图 -> 过滤 -> 入库/返回"
这条**核心链路在测试套件里没有任何回归保护**。
历史上它正是被 ``DEFAULT_INPUT_SIZE=416`` + ``MIN_BOX_SIZE=20``
静默打断成"恒定 0 检测、永不告警"的 —— 71 个全绿用例一个都没发现。
独立验证报告（``_审计报告/修复-独立验证.md`` §7-C）把这一点列为最重要的遗留，
本文件即针对它的最小回归保护。

跳过条件
--------
模型权重（``*.pt`` 已 gitignore）或真实样例图像任一缺失时自动 skip，
不会让测试套件失败 —— 因此从 git clone 后即使没有放置 ``best.pt``，
``pytest`` 依旧全绿（只是少跑这一条）。
"""
import base64
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import cv2
    import numpy as np
    import torch  # noqa: F401
    import ultralytics  # noqa: F401
except Exception as _exc:  # pragma: no cover - 环境缺失时整文件跳过
    pytest.skip(
        "未安装 torch / ultralytics / opencv，跳过真实推理冒烟测试: %s" % _exc,
        allow_module_level=True,
    )

from backend import config

# 真实样例图像不入库（体积原因）。需要跑这条回归时，把任意一张含目标的
# 真实图像放到 <项目根>/samples/ 下，或用环境变量 DETECT_SAMPLE_IMAGE 指定。
SAMPLE_CANDIDATES = tuple(p for p in (
    os.environ.get('DETECT_SAMPLE_IMAGE'),
    os.path.join(PROJECT_ROOT, 'samples', 'sample.jpg'),
    os.path.join(PROJECT_ROOT, 'samples', 'val_batch0_labels.jpg'),
) if p)


def _imread_unicode(path):
    """``cv2.imread`` 读不了含非 ASCII 字符的路径（本项目路径即含中文）"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _find_sample():
    for path in SAMPLE_CANDIDATES:
        if os.path.exists(path):
            img = _imread_unicode(path)
            if img is not None:
                return path, img
    return None, None


_SAMPLE_PATH, _SAMPLE_IMG = _find_sample()


@pytest.mark.skipif(
    not os.path.exists(config.MODEL_PATH),
    reason="未找到模型权重 %s（clone 后需自行放置 best.pt，见 README）"
           % config.MODEL_PATH,
)
@pytest.mark.skipif(
    _SAMPLE_IMG is None,
    reason="未找到真实样例图像（放入 <项目根>/samples/sample.jpg 即可启用本测试）",
)
def test_detect_real_image_end_to_end():
    """真实权重 + 真实图像走完整 HTTP 链路，必须产出非 0 检测且坐标在原图内"""
    from backend.app import create_app

    height, width = _SAMPLE_IMG.shape[:2]
    ok, buf = cv2.imencode('.jpg', _SAMPLE_IMG)
    assert ok, "样例图像编码失败"
    image_b64 = base64.b64encode(buf.tobytes()).decode()

    app = create_app()
    app.config['TESTING'] = True
    client = app.test_client()

    resp = client.post('/api/detect', json={'image': image_b64})
    assert resp.status_code == 200, resp.get_data(as_text=True)[:400]
    data = resp.get_json()

    assert data['success'] is True

    # 响应必须回传原图尺寸，供前端做坐标换算
    assert data['image_width'] == width
    assert data['image_height'] == height

    detections = data['detections']
    assert len(detections) > 0, (
        "真实图像上返回 0 个检测 —— 检测链路可能已回归。"
        "历史上 DEFAULT_INPUT_SIZE=416 + MIN_BOX_SIZE=20 曾导致恒定 0 检测、永不告警。"
    )

    # bbox 必须是原图像素坐标，且落在图像范围内
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        assert 0 <= x1 < x2 <= width, det
        assert 0 <= y1 < y2 <= height, det


@pytest.mark.skipif(
    not os.path.exists(config.MODEL_PATH),
    reason="未找到模型权重 %s" % config.MODEL_PATH,
)
def test_health_reports_real_model_state():
    """加载模型后 /api/health 必须反映真实的类别表与设备，而不是硬编码 true"""
    from backend.app import create_app
    from backend.models.detector import get_model, get_model_info

    get_model()  # 显式加载，避免依赖懒加载时机
    app = create_app()
    app.config['TESTING'] = True
    client = app.test_client()

    resp = client.get('/api/health')
    assert resp.status_code == 200
    data = resp.get_json()

    assert data['model_loaded'] is True
    assert data['status'] == 'healthy'
    assert data['num_classes'] == len(config.CLASS_NAMES)
    assert data['classes'] == config.CLASS_NAMES
    assert data['device'] in ('cpu', 'cuda')
    assert data['model_path'] == get_model_info()['model_path']