# -*- coding: utf-8 -*-
"""Flask API 路由

坐标约定：所有返回给前端的 bbox 一律是【原图像素坐标】。
检测链路：decode -> letterbox -> detect -> parse_results(反 letterbox 换算)
          -> process_detections(裁剪到原图范围并过滤)

其它：
- 帧去重：检测结果无变化则跳过写库
- 告警生命周期：只记录 triggered / resolved 事件
- 设置项（GET/PUT /api/settings）持久化在 SQLite 的 app_settings 表，
  默认值来自 config.DEFAULT_SETTINGS，范围校验来自 config.SETTINGS_RANGES
"""
import logging
import os
import uuid
from datetime import datetime

import cv2
from flask import Blueprint, jsonify, request, send_from_directory

from backend import config
from backend import database
from backend.models import detector
from backend.services import detection as detection_service
from backend.services.behavior import BehaviorAnalyzer
from backend.services.tracker import ObjectTracker
from backend.utils import image as image_utils


logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

# 跨请求持久化状态（单进程内存态；多 worker 部署下不共享，见修复报告“遗留问题”）
_tracker = ObjectTracker()
_analyzer = BehaviorAnalyzer()
_last_detection_hash = ""  # 用于帧去重

# 告警截图只允许这些扩展名
ALLOWED_SCREENSHOT_EXTENSIONS = ('.jpg', '.jpeg', '.png')

_HISTORY_DELETE_SCOPES = ('all', 'detections', 'alerts')
_HISTORY_LIMIT_MAX = 1000


# --------------------------------------------------------------------------
# 运行时状态
# --------------------------------------------------------------------------
def _detection_hash(detections):
    """生成检测结果的短哈希，用于去重判断"""
    if not detections:
        return ""
    parts = []
    for d in sorted(detections, key=lambda x: (x.get('class', ''), x.get('track_id', 0))):
        parts.append(f"{d.get('class','')}:{d.get('track_id',0)}")
    return "|".join(parts)


def _build_analyzer(settings):
    """按设置构造 BehaviorAnalyzer"""
    return BehaviorAnalyzer(
        thresholds={
            'eye_close': settings['eye_close_frames'],
            'yawn': settings['yawn_frames'],
            'distracted': settings['distracted_frames'],
        },
        cooldown_frames=settings['cooldown_frames'],
    )


def _apply_settings(settings):
    """把设置同步到运行时

    阈值/冷却帧数存在分析器实例里，改了设置必须重建 _analyzer 才生效。
    """
    global _analyzer
    _analyzer = _build_analyzer(settings)


def reset_runtime_state(settings=None):
    """重置跨请求内存状态（tracker / 去重哈希 / 分析器）

    ``settings`` 省略时从数据库读取。应用启动与测试隔离时调用。
    """
    global _tracker, _last_detection_hash
    _tracker = ObjectTracker()
    _last_detection_hash = ""
    _apply_settings(settings if settings is not None else database.get_settings())


# --------------------------------------------------------------------------
# 参数校验
# --------------------------------------------------------------------------
def _clamp_unit_float(value, default, name):
    """校验可选数值参数并夹取到 [0, 1]

    None 用默认值；类型不是数值时报 ValueError（交给上层返回 400）。
    """
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} 必须是 0~1 之间的数值')
    return min(1.0, max(0.0, float(value)))


def _clamp_int(value, lower, upper, default):
    """把可选的整数查询参数夹取到 [lower, upper]"""
    if value is None:
        return default
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return max(lower, min(upper, value))


def _validate_settings_payload(payload):
    """设置的白名单 + 类型 + 范围校验

    Returns:
        (cleaned, error): 校验通过时 error 为 None
    """
    if not isinstance(payload, dict) or not payload:
        return {}, '请求体必须是非空的 JSON 对象'

    defaults = database.default_settings()
    ranges = getattr(config, 'SETTINGS_RANGES', {})

    cleaned = {}
    for key, value in payload.items():
        if key not in defaults:
            return {}, f'不支持的设置项: {key}'
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return {}, f'{key} 必须是数值'

        default_value = defaults[key]
        if isinstance(default_value, float):
            value = float(value)
        else:
            if float(value) != int(value):
                return {}, f'{key} 必须是整数'
            value = int(value)

        lower, upper = ranges.get(key, (None, None))
        if lower is not None and value < lower:
            return {}, f'{key} 不能小于 {lower}'
        if upper is not None and value > upper:
            return {}, f'{key} 不能大于 {upper}'

        cleaned[key] = value

    return cleaned, None


# --------------------------------------------------------------------------
# 检测
# --------------------------------------------------------------------------
@api_bp.route('/api/detect', methods=['POST'])
def detect():
    """接收 Base64 图像，返回检测与告警结果（bbox 为原图像素坐标）

    请求体：{"image": "<base64 或 dataURL>", "conf"?: 0~1, "iou"?: 0~1}
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data.get('image'):
        return jsonify({'error': 'No image provided'}), 400

    try:
        settings = database.get_settings()
        conf = _clamp_unit_float(data.get('conf'), settings['conf_threshold'], 'conf')
        iou = _clamp_unit_float(data.get('iou'), settings['iou_threshold'], 'iou')
        input_size = int(settings['input_size'])
        max_det = int(settings['max_detections'])

        img = image_utils.decode_base64_image(data['image'])
        img_h, img_w = img.shape[:2]

        # letterbox 等比缩放 + 居中填充；坐标随后由 parse_results 反算回原图
        lb_img, ratio, pad_x, pad_y = image_utils.letterbox_image(img, input_size)
        results = detector.detect_objects(
            lb_img, conf=conf, iou=iou, max_det=max_det, input_size=input_size)
        raw_detections = detection_service.parse_results(results, ratio, pad_x, pad_y)
        detections = detection_service.process_detections(raw_detections, img_w, img_h)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception('检测失败')
        return jsonify({'error': f'Internal error: {str(e)}'}), 500

    detections = _tracker.update(detections)
    alerts = _analyzer.analyze(detections)

    # 帧去重：检测结果无变化则跳过写入
    global _last_detection_hash
    dh = _detection_hash(detections)
    if dh != _last_detection_hash:
        try:
            database.insert_detections(detections)
            _last_detection_hash = dh
        except Exception:
            logger.exception('检测记录写入失败')

    # 告警事件处理：triggered / resolved 时保存截图并落库
    triggered = [e for e in alerts if e.status == 'triggered']
    resolved = [e for e in alerts if e.status == 'resolved']
    if triggered or resolved:
        screenshot_path = _save_alert_screenshot(img)
        try:
            if triggered:
                database.insert_alerts(triggered, _analyzer.state, screenshot_path)
            if resolved:
                database.insert_alerts(resolved, _analyzer.state, screenshot_path)
        except Exception:
            logger.exception('告警记录写入失败')

    return jsonify({
        'success': True,
        'detections': detections,
        'alerts': [e.to_dict() for e in alerts],
        'behavior_state': _analyzer.state,
        'input_size': input_size,
        'image_width': img_w,
        'image_height': img_h,
    })


@api_bp.route('/api/health', methods=['GET'])
def health_check():
    """健康检查：返回真实模型状态（不会触发模型加载）"""
    info = detector.get_model_info()
    model_loaded = bool(info.get('loaded'))
    return jsonify({
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'model_path': info.get('model_path'),
        'device': info.get('device', 'unknown'),
        'num_classes': info.get('num_classes', 0),
        'classes': info.get('classes', []),
    })


# --------------------------------------------------------------------------
# 告警截图
# --------------------------------------------------------------------------
def _save_alert_screenshot(image):
    """保存告警截图，返回文件名；写入失败返回空字符串并记录日志

    旧实现忽略 ``cv2.imwrite`` 的返回值，磁盘满/无写权限时会静默把
    一个不存在的截图路径写进数据库。
    """
    try:
        os.makedirs(config.ALERT_SCREENSHOT_DIR, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(config.ALERT_SCREENSHOT_DIR, filename)
        written = cv2.imwrite(filepath, image)
        if not written or not os.path.isfile(filepath):
            logger.error('告警截图写入失败（cv2.imwrite 返回 %r）: %s', written, filepath)
            return ""
        return filename
    except Exception:
        logger.exception('告警截图保存异常')
        return ""


def _screenshot_url(screenshot_path):
    """由数据库中的截图路径/文件名推导可访问 URL（形如 /api/screenshots/<文件名>）"""
    if not screenshot_path:
        return ''
    name = os.path.basename(str(screenshot_path).replace('\\', '/'))
    return f'/api/screenshots/{name}' if name else ''


def _with_screenshot_url(rows):
    """给告警记录补充 screenshot_url（P0-4：截图只写不读）"""
    result = []
    for row in rows:
        item = dict(row)
        item['screenshot_url'] = _screenshot_url(item.get('screenshot_path'))
        result.append(item)
    return result


def _safe_screenshot_path(filename):
    """校验截图文件名并解析为绝对路径；非法（含路径穿越）返回 None

    三重防线：① 只允许纯 basename；② 只允许 jpg/jpeg/png；
    ③ 解析后的绝对路径必须仍位于 config.ALERT_SCREENSHOT_DIR 内。
    """
    if not isinstance(filename, str) or not filename:
        return None

    normalized = filename.replace('\\', '/')
    if normalized in ('.', '..') or os.path.basename(normalized) != normalized:
        return None
    if os.path.splitext(normalized)[1].lower() not in ALLOWED_SCREENSHOT_EXTENSIONS:
        return None

    base_dir = os.path.abspath(config.ALERT_SCREENSHOT_DIR)
    full_path = os.path.abspath(os.path.join(base_dir, normalized))
    if os.path.dirname(full_path) != base_dir:
        return None
    return full_path


@api_bp.route('/api/screenshots/<path:filename>', methods=['GET'])
def get_screenshot(filename):
    """安全返回告警截图"""
    full_path = _safe_screenshot_path(filename)
    if full_path is None:
        return jsonify({'error': 'Invalid screenshot filename'}), 400
    if not os.path.isfile(full_path):
        return jsonify({'error': 'Screenshot not found'}), 404
    return send_from_directory(
        os.path.abspath(config.ALERT_SCREENSHOT_DIR), os.path.basename(full_path))


# --------------------------------------------------------------------------
# 历史记录
# --------------------------------------------------------------------------
@api_bp.route('/api/history', methods=['GET'])
def get_history():
    """查询检测与告警历史（支持时间范围过滤和按天统计）

    告警行额外带 ``screenshot_url``，前端可直接用作缩略图地址。
    """
    limit = _clamp_int(request.args.get('limit', 50, type=int), 1, _HISTORY_LIMIT_MAX, 50)
    offset = _clamp_int(request.args.get('offset', 0, type=int), 0, 10 ** 9, 0)
    filter_class = request.args.get('class', None)
    filter_alert = request.args.get('alert_type', None)
    stats_only = request.args.get('stats', '0') == '1'
    start = request.args.get('start', None)
    end = request.args.get('end', None)
    trend = request.args.get('trend', '0') == '1'

    if stats_only:
        return jsonify(database.get_detection_stats())

    if trend:
        days = _clamp_int(request.args.get('days', 7, type=int), 1, 365, 7)
        return jsonify(database.get_daily_stats(days))

    if start and end:
        return jsonify({
            'detections': database.query_detections_by_time(start, end, limit, offset),
            'alerts': _with_screenshot_url(
                database.query_alerts_by_time(start, end, limit, offset)),
        })

    return jsonify({
        'detections': database.query_detections(limit, offset, filter_class),
        'alerts': _with_screenshot_url(
            database.query_alerts(limit, offset, filter_alert)),
    })


@api_bp.route('/api/history', methods=['DELETE'])
def delete_history():
    """按范围清空历史数据：``?scope=all|detections|alerts``（默认 all）"""
    scope = (request.args.get('scope') or 'all').strip().lower()
    if scope not in _HISTORY_DELETE_SCOPES:
        return jsonify({
            'error': f'scope 必须是 {list(_HISTORY_DELETE_SCOPES)} 之一'
        }), 400

    deleted = database.delete_records(scope)

    # 检测记录被清空后去重哈希同时失效：否则画面不变时下一帧会被
    # “结果无变化”永久跳过，用户会以为清空后不再记录任何数据。
    if scope in ('all', 'detections'):
        global _last_detection_hash
        _last_detection_hash = ""

    return jsonify({'success': True, 'deleted': deleted})


# --------------------------------------------------------------------------
# 设置
# --------------------------------------------------------------------------
@api_bp.route('/api/settings', methods=['GET'])
def get_settings():
    """读取运行时可调设置"""
    return jsonify({'success': True, 'settings': database.get_settings()})


@api_bp.route('/api/settings', methods=['PUT'])
def update_settings():
    """更新运行时可调设置（白名单 + 类型/范围校验），并立即对 /api/detect 生效"""
    payload = request.get_json(silent=True)
    cleaned, error = _validate_settings_payload(payload)
    if error:
        return jsonify({'error': error}), 400

    settings = database.set_settings(cleaned)
    _apply_settings(settings)
    return jsonify({'success': True, 'settings': settings})