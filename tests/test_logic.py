# -*- coding: utf-8 -*-
"""逻辑正确性测试：几何/追踪/告警生命周期

注：几何过滤的详细语义（P0-2 包含度）见 test_geometry.py；
本文件只做链路级的回归。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.geometry import calculate_iou, is_valid_detection, filter_detections
from backend.services.tracker import ObjectTracker
from backend.services.behavior import BehaviorAnalyzer


def test_iou():
    iou = calculate_iou([0, 0, 100, 100], [50, 50, 150, 150])
    assert abs(iou - 0.1429) < 0.01
    print('  IOU: OK')


def test_validation():
    # 新签名：(box, img_w, img_h)，坐标是原图像素
    assert is_valid_detection([10, 10, 50, 50], 640, 480)
    assert not is_valid_detection([10, 10, 15, 15], 640, 480)
    print('  Validation: OK')


def test_filter():
    """P0-2 修正：脸内的 close_eye 必须保留（旧断言把错误行为固化为预期）

    旧实现用 IOU>0.1 判断，闭眼框 [20,20,50,50] 对脸框 [0,0,100,100] 的
    IOU 只有 0.09，会被丢弃；新语义按包含度（=1.0）保留。
    """
    dets = [
        {'class': 'face', 'confidence': 0.9, 'bbox': [0, 0, 100, 100]},
        {'class': 'close_eye', 'confidence': 0.8, 'bbox': [20, 20, 50, 50]},
        {'class': 'close_eye', 'confidence': 0.7, 'bbox': [300, 300, 350, 350]},
    ]
    filtered = filter_detections(dets, 640, 480)
    assert sorted(d['class'] for d in filtered) == ['close_eye', 'face']
    # 脸框外的闭眼框仍然被丢弃
    kept_boxes = [d['bbox'] for d in filtered if d['class'] == 'close_eye']
    assert kept_boxes == [[20, 20, 50, 50]]
    print('  Filter: OK')


def test_tracker():
    t = ObjectTracker()
    r1 = t.update([{'class': 'face', 'confidence': 0.9, 'bbox': [0, 0, 100, 100]}])
    assert 'track_id' in r1[0]
    r2 = t.update([{'class': 'face', 'confidence': 0.9, 'bbox': [2, 2, 102, 102]}])
    assert r2[0]['track_id'] == r1[0]['track_id']
    print('  Tracker: OK')


def test_alert_lifecycle():
    """验证告警生命周期: triggered -> ongoing -> resolved -> cooldown"""
    a = BehaviorAnalyzer(cooldown_frames=5)
    d_close = {'class': 'close_eye', 'confidence': 0.9, 'bbox': [20, 20, 50, 50]}
    d_open = {'class': 'open_eye', 'confidence': 0.9, 'bbox': [20, 20, 50, 50]}

    # 累积到阈值
    a.analyze([d_close])
    a.analyze([d_close])
    assert a.state['eye_close_frames'] == 2

    # 第3帧: 达到阈值 -> triggered
    alerts = a.analyze([d_close])
    assert len(alerts) == 1
    assert alerts[0].status == 'triggered'
    assert alerts[0].type == 'eye_close'
    print(f'  Triggered: {alerts[0].message}')

    # 第4帧: 持续中 -> 不产生事件
    alerts = a.analyze([d_close])
    assert len(alerts) == 0
    print('  Ongoing: no event emitted')

    # 眼睛睁开 -> resolved
    alerts = a.analyze([d_open])
    assert len(alerts) == 1
    assert alerts[0].status == 'resolved'
    assert a.state['eye_close_frames'] == 0
    print(f'  Resolved: {alerts[0].message}')

    # 冷却期内: 再次闭眼不触发
    for _ in range(3):
        a.analyze([d_close])
    assert a.state['eye_close_frames'] == 3
    alerts = a.analyze([d_close])
    assert len(alerts) == 0, 'Cooldown should block'
    print('  Cooldown: blocked')

    # 冷却在此帧过期并触发
    alerts = a.analyze([d_close])
    assert len(alerts) == 1 and alerts[0].status == 'triggered'
    print(f'  Re-triggered: {alerts[0].message}')
    print('  Alert lifecycle test: PASSED')


def test_analyzer_thresholds_are_configurable():
    """自定义阈值（api.py 用设置项重建分析器时依赖这个能力）"""
    a = BehaviorAnalyzer(thresholds={'eye_close': 5, 'yawn': 2, 'distracted': 3},
                         cooldown_frames=1)
    d_close = {'class': 'close_eye', 'confidence': 0.9, 'bbox': [20, 20, 50, 50]}
    for _ in range(4):
        assert a.analyze([d_close]) == []
    alerts = a.analyze([d_close])
    assert len(alerts) == 1 and alerts[0].type == 'eye_close'
    assert alerts[0].frame_count == 5