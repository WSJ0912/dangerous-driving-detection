# -*- coding: utf-8 -*-
"""几何工具测试（P0-2 新语义：包含度判定 + 原图像素坐标）

背景：旧实现用 ``calculate_iou(eye, face) > 0.1`` 判断"眼睛在脸内"，
而眼睛框面积通常只占脸框的 3%~8%，IOU 恒小于 0.1，导致真实的
眼睛/哈欠框被**系统性误杀**（且无脸框时 ``any([]) == False`` 全丢）。
新语义用 ``containment_ratio``（交集 / 特征框自身面积）。
"""
import pytest

from backend import config
from backend.utils.geometry import (
    calculate_iou, containment_ratio, is_valid_detection, filter_detections,
)


def _det(cls, conf, bbox):
    return {'class': cls, 'confidence': conf, 'bbox': list(bbox)}


# --------------------------------------------------------------------------
# calculate_iou（回归）
# --------------------------------------------------------------------------
def test_calculate_iou_regression():
    assert abs(calculate_iou([0, 0, 100, 100], [50, 50, 150, 150]) - 0.1429) < 0.01
    assert calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


# --------------------------------------------------------------------------
# containment_ratio
# --------------------------------------------------------------------------
def test_containment_ratio_fully_inside_is_one():
    assert containment_ratio([20, 20, 40, 40], [0, 0, 100, 100]) == 1.0
    assert containment_ratio([50, 50, 80, 80], [0, 0, 100, 100]) == 1.0


def test_containment_ratio_is_relative_to_inner_area_not_union():
    """交集 50x50=2500，inner 自身面积 100x100=10000 -> 0.25

    若误用 IOU 语义会得到 2500/17500≈0.143，这里固定住"包含度"语义。
    """
    assert containment_ratio([50, 50, 150, 150], [0, 0, 100, 100]) == pytest.approx(0.25)


def test_containment_ratio_no_overlap():
    assert containment_ratio([200, 200, 300, 300], [0, 0, 100, 100]) == 0.0


def test_containment_ratio_outer_smaller_than_inner():
    """小框作为 outer：交集 20x20=400 / inner 面积 10000 = 0.04"""
    assert containment_ratio([0, 0, 100, 100], [20, 20, 40, 40]) == pytest.approx(0.04)


def test_containment_ratio_degenerate_box():
    assert containment_ratio([0, 0, 0, 0], [0, 0, 100, 100]) == 0.0


# --------------------------------------------------------------------------
# is_valid_detection（阈值可参数化，坐标是原图像素）
# --------------------------------------------------------------------------
def test_is_valid_detection_basic():
    assert is_valid_detection([10, 10, 50, 50], 640, 480)
    assert not is_valid_detection([10, 10, 15, 15], 640, 480)  # 边长 < MIN_BOX_SIZE


def test_is_valid_detection_uses_original_image_bounds():
    assert not is_valid_detection([10, 10, 700, 50], 640, 480)   # 超出宽度
    assert not is_valid_detection([10, 600, 50, 700], 640, 480)  # 超出高度
    assert not is_valid_detection([-5, 10, 50, 50], 640, 480)    # 负坐标
    assert is_valid_detection([0, 0, 640, 480], 640, 480)        # 正好铺满


def test_is_valid_detection_thresholds_are_configurable():
    """旧实现把 config 阈值写死，改不动；新实现必须支持按调用覆盖"""
    assert not is_valid_detection([0, 0, 15, 15], 640, 480)
    assert is_valid_detection([0, 0, 15, 15], 640, 480, min_size=10)
    assert not is_valid_detection([0, 0, 100, 100], 640, 480, max_ar=0.5)
    assert is_valid_detection([0, 0, 100, 100], 640, 480, min_ar=0.9, max_ar=1.1)


def test_is_valid_detection_defaults_match_config():
    assert is_valid_detection([0, 0, config.MIN_BOX_SIZE, config.MIN_BOX_SIZE], 640, 480)
    assert not is_valid_detection(
        [0, 0, config.MIN_BOX_SIZE - 1, config.MIN_BOX_SIZE - 1], 640, 480)


# --------------------------------------------------------------------------
# filter_detections（核心 P0-2 回归）
# --------------------------------------------------------------------------
def test_filter_keeps_eyes_fully_inside_face():
    """P0-2 验收：20x20 与 30x30 的眼睛框完全落在 100x100 脸框内必须保留"""
    dets = [
        _det('face', 0.9, [0, 0, 100, 100]),
        _det('close_eye', 0.8, [20, 20, 40, 40]),   # 20x20，包含度 1.0
        _det('open_eye', 0.7, [50, 50, 80, 80]),    # 30x30，包含度 1.0
    ]
    kept = filter_detections(dets, 640, 480)
    assert sorted(d['class'] for d in kept) == ['close_eye', 'face', 'open_eye']

    # 同样的框在旧 IOU 语义下 IOU 只有 0.04 / 0.09，会被全部丢弃
    assert calculate_iou([20, 20, 40, 40], [0, 0, 100, 100]) < 0.1
    assert calculate_iou([50, 50, 80, 80], [0, 0, 100, 100]) < 0.1


def test_filter_keeps_yawn_inside_face():
    dets = [
        _det('face', 0.9, [100, 100, 300, 300]),
        _det('Yawn', 0.6, [180, 180, 220, 220]),
        _det('noYawn', 0.5, [150, 150, 190, 190]),
    ]
    kept = filter_detections(dets, 640, 480)
    assert sorted(d['class'] for d in kept) == ['Yawn', 'face', 'noYawn']


def test_filter_drops_features_outside_face():
    dets = [
        _det('face', 0.9, [0, 0, 100, 100]),
        _det('close_eye', 0.8, [300, 300, 350, 350]),  # 与脸框无交集
        _det('Yawn', 0.6, [95, 95, 145, 145]),         # 包含度 25/2500 = 0.01
    ]
    kept = filter_detections(dets, 640, 480)
    assert [d['class'] for d in kept] == ['face']


def test_filter_keeps_features_when_no_face_detected():
    """无脸框时不做约束：无法判定 != 判定为无效（旧实现 any([]) == False 全丢）"""
    dets = [
        _det('close_eye', 0.8, [20, 20, 40, 40]),
        _det('Yawn', 0.6, [60, 60, 100, 100]),
    ]
    kept = filter_detections(dets, 640, 480)
    assert sorted(d['class'] for d in kept) == ['Yawn', 'close_eye']


def test_filter_respects_custom_containment_threshold():
    dets = [
        _det('face', 0.9, [0, 0, 100, 100]),
        _det('close_eye', 0.8, [50, 50, 150, 150]),  # 包含度 0.25
    ]
    assert [d['class'] for d in filter_detections(dets, 640, 480)] == ['face']
    kept = filter_detections(dets, 640, 480, face_containment_min=0.2)
    assert sorted(d['class'] for d in kept) == ['close_eye', 'face']


def test_filter_threshold_default_comes_from_config():
    assert config.FACE_CONTAINMENT_MIN == 0.5
    dets = [
        _det('face', 0.9, [0, 0, 100, 100]),
        _det('close_eye', 0.8, [60, 60, 140, 140]),  # 包含度 1600/6400 = 0.25 < 0.5
    ]
    assert [d['class'] for d in filter_detections(dets, 640, 480)] == ['face']


def test_filter_does_not_constrain_non_face_feature_classes():
    dets = [
        _det('face', 0.9, [0, 0, 100, 100]),
        _det('phone', 0.7, [400, 100, 460, 160]),
        _det('smoke', 0.6, [480, 100, 540, 160]),
        _det('drink', 0.55, [560, 100, 620, 160]),
    ]
    kept = filter_detections(dets, 640, 480)
    assert sorted(d['class'] for d in kept) == ['drink', 'face', 'phone', 'smoke']


def test_filter_applies_original_image_bounds():
    dets = [
        _det('face', 0.9, [0, 0, 100, 100]),
        _det('close_eye', 0.8, [20, 20, 40, 40]),
        _det('phone', 0.7, [700, 100, 760, 160]),   # 超出 640 宽度
    ]
    kept = filter_detections(dets, 640, 480)
    assert sorted(d['class'] for d in kept) == ['close_eye', 'face']

    # 图变小到脸框失效时，眼睛不再受脸框约束，仍然保留（新语义）
    assert [d['class'] for d in filter_detections(dets, 50, 50)] == ['close_eye']


def test_filter_empty_input():
    assert filter_detections([], 640, 480) == []