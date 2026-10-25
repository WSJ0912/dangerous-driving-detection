# -*- coding: utf-8 -*-
"""检测结果解析与过滤

坐标约定：对外暴露的 bbox 一律是【原图像素坐标】。
"""
from backend import config
from backend.utils.geometry import filter_detections


def parse_results(results, ratio=1.0, pad_x=0, pad_y=0):
    """解析 YOLO 推理结果为统一格式的检测列表，并把坐标换算回原图

    Args:
        results: ultralytics 推理结果列表
        ratio: letterbox 缩放比例（原图 -> 网络输入）
        pad_x, pad_y: letterbox 填充像素数

    Returns:
        list[dict]: [{'class': str, 'confidence': float, 'bbox': [x1,y1,x2,y2]}, ...]
                    坐标为原图像素坐标（尚未裁剪到图像范围内）
    """
    detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            if class_id < 0 or class_id >= len(config.CLASS_NAMES):
                continue

            class_name = config.CLASS_NAMES[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])

            # 反 letterbox：x_orig = (x_net - pad) / ratio
            if ratio and ratio > 0:
                x1 = (x1 - pad_x) / ratio
                x2 = (x2 - pad_x) / ratio
                y1 = (y1 - pad_y) / ratio
                y2 = (y2 - pad_y) / ratio

            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': [int(round(x1)), int(round(y1)),
                         int(round(x2)), int(round(y2))],
            })
    return detections


def _clamp_box(box, img_w, img_h):
    """把检测框裁剪到图像范围内；完全出界时返回 None"""
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), img_w))
    y1 = max(0, min(int(y1), img_h))
    x2 = max(0, min(int(x2), img_w))
    y2 = max(0, min(int(y2), img_h))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return [x1, y1, x2, y2]


def process_detections(detections, img_w, img_h):
    """裁剪到图像范围、过滤无效框，并附加危险等级

    Args:
        detections: parse_results 的输出（原图坐标，可能略越界）
        img_w, img_h: 原图宽高

    Returns:
        list[dict]: 附带 danger_level 的检测结果
    """
    clamped = []
    for d in detections:
        box = _clamp_box(d['bbox'], img_w, img_h)
        if box is None:
            continue
        clamped.append({
            'class': d['class'],
            'confidence': d['confidence'],
            'bbox': box,
        })

    filtered = filter_detections(clamped, img_w, img_h)

    return [
        {
            'class': d['class'],
            'confidence': round(d['confidence'], 2),
            'bbox': d['bbox'],
            'danger_level': config.DANGER_LEVELS.get(d['class'], 'low'),
        }
        for d in filtered
    ]