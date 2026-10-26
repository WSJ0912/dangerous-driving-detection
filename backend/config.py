# -*- coding: utf-8 -*-
"""
全局配置管理
"""
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


def resource_path(relative_path):
    """获取资源文件的绝对路径（兼容 PyInstaller 打包）"""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = BASE_DIR
    return os.path.normpath(os.path.join(base_path, relative_path))


DB_PATH = os.path.join(BASE_DIR, 'data', 'detection.db')
ALERT_SCREENSHOT_DIR = os.path.join(BASE_DIR, 'uploads', 'alerts')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ---- 模型配置 ----
# 生产模型路径。权重文件不入库（*.pt 被 .gitignore 排除），放置方式见 models/README.md
MODEL_PATH = resource_path("../models/weights/best.pt")
# 回退路径：替换前的备份权重（同样不入库）
MODEL_PATH_FALLBACK = resource_path("../models/weights/best.prev.bak.pt")

# ---- 推理参数 ----
# 输入尺寸需与模型训练尺寸一致（当前权重训练时使用 imgsz=640）。
# 注意：图像会做 letterbox 等比缩放后再送入模型，检测框会换算回原图像素坐标，
# 因此该值只影响精度，不影响返回给前端的坐标系。
DEFAULT_INPUT_SIZE = 640
DETECTION_CONF_THRESHOLD = 0.3
DETECTION_IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 10

# ---- 检测过滤 ----
# 这些阈值作用在【原图像素坐标】上（不是缩放后的网络输入坐标）
MIN_BOX_SIZE = 20
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0

# 眼/哈欠等面部特征框需有多少比例的面积落在脸框内才认为有效。
# 语义是"包含度"（交集面积 / 特征框面积），不是 IOU：
# 眼睛框远小于脸框，用 IOU 判定会把真实的眼睛框全部误杀。
FACE_CONTAINMENT_MIN = 0.5

# ---- 目标追踪 ----
TRACK_IOU_THRESHOLD = 0.3

# ---- 行为分析阈值 ----
WARNING_THRESHOLDS = {
    'eye_close': 3,
    'yawn': 2,
    'distracted': 5,
}
WARNING_COOLDOWN_FRAMES = 30

# ---- 运行时可调设置（可通过 GET/PUT /api/settings 读写，持久化在 SQLite）----
DEFAULT_SETTINGS = {
    'detect_interval_ms': 300,
    'conf_threshold': DETECTION_CONF_THRESHOLD,
    'iou_threshold': DETECTION_IOU_THRESHOLD,
    'input_size': DEFAULT_INPUT_SIZE,
    'max_detections': MAX_DETECTIONS,
    'eye_close_frames': WARNING_THRESHOLDS['eye_close'],
    'yawn_frames': WARNING_THRESHOLDS['yawn'],
    'distracted_frames': WARNING_THRESHOLDS['distracted'],
    'cooldown_frames': WARNING_COOLDOWN_FRAMES,
}

# 每个可调设置的取值范围（用于后端校验）
SETTINGS_RANGES = {
    'detect_interval_ms': (100, 5000),
    'conf_threshold': (0.01, 1.0),
    'iou_threshold': (0.01, 1.0),
    'input_size': (320, 1280),
    'max_detections': (1, 100),
    'eye_close_frames': (1, 100),
    'yawn_frames': (1, 100),
    'distracted_frames': (1, 100),
    'cooldown_frames': (0, 600),
}

# ---- 类别定义 ----
CLASS_NAMES = [
    'Yawn',
    'close_eye',
    'noYawn',
    'open_eye',
    'face',
    'smoke',
    'drink',
    'phone',
]

DANGER_LEVELS = {
    'Yawn': 'medium',
    'close_eye': 'high',
    'noYawn': 'low',
    'open_eye': 'low',
    'face': 'low',
    'smoke': 'high',
    'drink': 'high',
    'phone': 'high',
}

# ---- 告警消息 ----
ALERT_MESSAGES = {
    'eye_close': '驾驶员眼睛闭合，请注意安全！',
    'yawn': '驾驶员打哈欠，可能疲劳驾驶！',
    'distracted': '驾驶员分心，请注意专注驾驶！',
}

# ---- Flask 配置 ----
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
STATIC_FOLDER = resource_path('../frontend_web')
TEMPLATE_FOLDER = resource_path('../frontend_web')