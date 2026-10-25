# -*- coding: utf-8 -*-
"""几何计算工具：IOU、包含度、边界框验证与过滤

所有坐标均为【原图像素坐标】。
"""
from backend import config


# 需要约束在脸框内的面部特征类别
FACE_FEATURE_CLASSES = ('open_eye', 'close_eye', 'Yawn', 'noYawn')


def calculate_iou(box1, box2):
    """计算两个边界框的 IOU (Intersection over Union)

    box 格式: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def containment_ratio(inner, outer):
    """计算 inner 框有多大比例落在 outer 框内

    返回 交集面积 / inner 面积，取值 0~1。

    为什么不用 IOU：眼睛框面积通常只占脸框的 3%~8%，此时"眼在脸内"的 IOU
    也恒小于 0.1，用 IOU 判定会把真实的眼睛/哈欠框全部误杀。
    本函数衡量的是"被包含程度"，完全落在脸内即为 1.0。
    """
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    inner_area = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])

    return intersection / inner_area if inner_area > 0 else 0.0


def is_valid_detection(box, img_w, img_h, min_size=None, min_ar=None, max_ar=None):
    """检查检测框是否合理

    Args:
        box: [x1, y1, x2, y2]（原图像素坐标）
        img_w, img_h: 原图宽高
        min_size: 最小边长（默认取 config.MIN_BOX_SIZE）
        min_ar / max_ar: 宽高比范围（默认取 config）
    """
    if min_size is None:
        min_size = config.MIN_BOX_SIZE
    if min_ar is None:
        min_ar = config.MIN_ASPECT_RATIO
    if max_ar is None:
        max_ar = config.MAX_ASPECT_RATIO

    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    if width < min_size or height < min_size:
        return False

    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        return False

    aspect_ratio = width / height if height > 0 else 0
    if aspect_ratio < min_ar or aspect_ratio > max_ar:
        return False

    return True


def filter_detections(detections, img_w, img_h, face_containment_min=None):
    """过滤不合理的检测框，并约束面部特征落在脸框内

    Args:
        detections: [{'class', 'confidence', 'bbox'}, ...]
        img_w, img_h: 原图宽高
        face_containment_min: 特征框需落在脸框内的最小面积比例
                              （默认取 config.FACE_CONTAINMENT_MIN）

    Returns:
        list[dict]: 过滤后的检测框（脸框在前）
    """
    if face_containment_min is None:
        face_containment_min = config.FACE_CONTAINMENT_MIN

    valid = [d for d in detections if is_valid_detection(d['bbox'], img_w, img_h)]

    face_detections = [d for d in valid if d['class'] == 'face']
    other_detections = [d for d in valid if d['class'] != 'face']

    filtered_other = []
    for det in other_detections:
        if det['class'] in FACE_FEATURE_CLASSES:
            # 本帧没有检出脸框时不做约束：无法判定不等于判定为无效，
            # 旧实现用 any([]) == False 把这类检测全部丢弃，属系统性误杀。
            if not face_detections:
                filtered_other.append(det)
                continue
            if any(
                containment_ratio(det['bbox'], face['bbox']) >= face_containment_min
                for face in face_detections
            ):
                filtered_other.append(det)
        else:
            filtered_other.append(det)

    return face_detections + filtered_other