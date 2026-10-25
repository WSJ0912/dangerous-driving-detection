# -*- coding: utf-8 -*-
"""图像编解码与预处理工具"""
import base64

import cv2
import numpy as np


def decode_base64_image(b64_string: str) -> np.ndarray:
    """Base64 字符串解码为 OpenCV 图像

    兼容两种输入：
    - 纯 base64（前端 detect.html 使用 `.split(',')[1]` 后的结果）
    - 带 data URL 前缀的完整字符串（如 `data:image/jpeg;base64,....`）

    Returns:
        numpy array (H, W, C) BGR 格式

    Raises:
        ValueError: 解码失败或格式无效
    """
    if not isinstance(b64_string, str) or not b64_string.strip():
        raise ValueError("Invalid base64 image data")

    payload = b64_string.strip()
    # 容错：剥离 data URL 前缀。若直接对整串做 b64decode，
    # 前缀里的字符恰好都在 base64 字母表内，会静默解出损坏数据。
    if payload.startswith('data:'):
        comma = payload.find(',')
        if comma == -1:
            raise ValueError("Invalid data URL image data")
        payload = payload[comma + 1:]

    try:
        image_data = base64.b64decode(payload, validate=False)
    except Exception:
        raise ValueError("Invalid base64 image data")

    if not image_data:
        raise ValueError("Invalid base64 image data")

    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image format")

    return img


def letterbox_image(img: np.ndarray, input_size: int):
    """等比缩放并居中填充到 input_size x input_size

    与直接 `cv2.resize(img, (input_size, input_size))` 不同，本函数保持长宽比，
    避免把 16:9 的驾驶画面压成 1:1 导致人脸纵向变形、小目标丢失。

    Args:
        img: numpy array (H, W, C)
        input_size: 目标正方形边长

    Returns:
        (letterboxed_img, ratio, pad_x, pad_y)
        ratio  缩放比例（原图 -> 缩放后）
        pad_x  左侧填充像素数
        pad_y  顶部填充像素数

    坐标反算（把网络输出框还原到原图）：
        x_orig = (x_letterboxed - pad_x) / ratio
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image shape")

    ratio = min(float(input_size) / float(w), float(input_size) / float(h))
    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, int(round(h * ratio)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((input_size, input_size, 3), 114, dtype=img.dtype)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, ratio, pad_x, pad_y


def preprocess_image(img: np.ndarray, input_size: int) -> np.ndarray:
    """兼容旧调用：只返回 letterbox 后的图像"""
    letterboxed, _, _, _ = letterbox_image(img, input_size)
    return letterboxed