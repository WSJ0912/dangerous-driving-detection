# -*- coding: utf-8 -*-
"""IOU 目标追踪器"""
from backend import config
from backend.utils.geometry import calculate_iou


class ObjectTracker:
    """基于 IOU 的简单目标追踪器"""

    def __init__(self, iou_threshold=None):
        self._objects = {}
        self._next_id = 0
        self._iou_threshold = iou_threshold or config.TRACK_IOU_THRESHOLD

    def update(self, detections):
        """对当前帧检测结果进行追踪匹配

        Args:
            detections: [{'class': str, 'confidence': float, 'bbox': [4]}, ...]

        Returns:
            list[dict]: 附带 track_id 的检测结果
        """
        new_tracked = {}
        for det in detections:
            bbox = det['bbox']
            class_name = det['class']
            best_track_id = self._match(bbox, class_name)
            if best_track_id is not None:
                new_tracked[best_track_id] = {
                    'class': class_name,
                    'bbox': bbox,
                    'confidence': det['confidence'],
                    'last_seen': 0,
                }
                det['track_id'] = best_track_id
            else:
                new_id = self._next_id
                self._next_id += 1
                new_tracked[new_id] = {
                    'class': class_name,
                    'bbox': bbox,
                    'confidence': det['confidence'],
                    'last_seen': 0,
                }
                det['track_id'] = new_id
        self._objects = new_tracked
        return detections

    def _match(self, bbox, class_name):
        best_iou = self._iou_threshold
        best_id = None
        for tid, tdata in self._objects.items():
            if tdata['class'] == class_name:
                iou = calculate_iou(bbox, tdata['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
        return best_id
