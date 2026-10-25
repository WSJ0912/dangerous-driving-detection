# -*- coding: utf-8 -*-
"""YOLO 模型加载与推理"""
import os
import sys

import torch

from backend import config


_model = None
_model_path = None
_device = 'unknown'


def _resolve_model_path():
    """按 主路径 -> 回退路径 顺序解析模型文件"""
    for path in (config.MODEL_PATH, config.MODEL_PATH_FALLBACK):
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"模型文件不存在: {config.MODEL_PATH}\n"
        f"回退路径也不存在: {config.MODEL_PATH_FALLBACK}\n"
        f"请将 best.pt 放到 models/weights/ 目录下（详见 README 的“模型来源”章节）"
    )


def get_model():
    """获取全局单例 YOLO 模型，延迟加载"""
    global _model, _model_path, _device
    if _model is not None:
        return _model

    model_path = _resolve_model_path()

    sys.path.append(config.YOLO_PROJECT_PATH)
    from ultralytics import YOLO

    torch.serialization.add_safe_globals([torch.nn.Module])

    _model = YOLO(model_path)
    _model_path = model_path

    try:
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _model.to(_device)
        print(f"模型加载成功: {model_path}")
        print(f"推理设备: {'GPU' if _device == 'cuda' else 'CPU'}")
    except Exception as e:
        _device = 'cpu'
        print(f"模型设备设置失败: {e}, 使用 CPU")

    return _model


def is_model_loaded():
    """模型是否已成功加载（供健康检查使用，不会触发加载）"""
    return _model is not None


def get_model_info():
    """返回真实的模型状态，供 /api/health 使用（不会触发加载）"""
    info = {
        'loaded': False,
        'model_path': None,
        'device': 'unknown',
        'num_classes': 0,
        'classes': [],
    }
    if _model is None:
        return info

    names = getattr(_model, 'names', None) or {}
    try:
        ordered = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    except Exception:
        ordered = list(names.values())

    info.update({
        'loaded': True,
        'model_path': _model_path,
        'device': _device,
        'num_classes': len(ordered),
        'classes': ordered,
    })
    return info


def detect_objects(image, conf=None, iou=None, max_det=None, input_size=None):
    """执行 YOLO 目标检测

    Args:
        image: numpy array (H, W, C)，应为 letterbox 之后的正方形输入
        conf: 置信度阈值
        iou: NMS IOU 阈值
        max_det: 单帧最大检测数
        input_size: 网络输入尺寸；None 表示沿用模型自带配置

    Returns:
        ultralytics results 列表
    """
    model = get_model()
    use_half = _device == 'cuda'

    kwargs = {
        'source': image,
        'conf': conf if conf is not None else config.DETECTION_CONF_THRESHOLD,
        'iou': iou if iou is not None else config.DETECTION_IOU_THRESHOLD,
        'verbose': False,
        'half': use_half,
        'max_det': max_det if max_det is not None else config.MAX_DETECTIONS,
    }
    if input_size:
        kwargs['imgsz'] = int(input_size)

    return model.predict(**kwargs)