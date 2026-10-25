# -*- coding: utf-8 -*-
"""API 层测试：Flask test_client 打真实路由

只把最外层的 YOLO 推理（``detector.detect_objects``）换成桩以避免加载 torch，
**其余全部是真实代码**：真实 base64 解码、真实 letterbox、真实 parse_results
坐标反算、真实 process_detections 过滤、真实 BehaviorAnalyzer、真实 SQLite。

覆盖：
- /api/health 返回真实模型状态（且不触发模型加载）
- /api/detect 返回原图像素坐标 + conf/iou 生效与夹取 + 帧去重 + 告警落库
- /api/history 告警行的 screenshot_url（P0-4 截图只写不读）
- DELETE /api/history 三个 scope
- GET/PUT /api/settings（白名单 + 范围校验 + 立即生效）
- GET /api/screenshots/<file>（防路径穿越）
- /api/* 未知路径 / 错误方法返回 JSON 404 / 405，且不影响非 API 的静态回落
"""
import base64
import json
import os

import cv2
import numpy as np
import pytest

from backend import config
from backend import database
from backend import app as app_module
from backend.models import detector
from backend.routes import api as api_module
from backend.routes.api import _safe_screenshot_path
from backend.services.behavior import AlertEvent


# 测试用原图 1280x720；letterbox 到 640 时 ratio=0.5、pad=(0,140)
IMG_W, IMG_H = 1280, 720

# 期望的【原图坐标】检测框
FACE_BOX = [100, 100, 400, 500]        # 300x400
CLOSE_EYE_BOX = [120, 140, 140, 160]   # 20x20，完全落在脸框内
PHONE_BOX = [900, 100, 1000, 300]


def _letterbox_params(input_size):
    """镜像 letterbox_image 的几何，用于把期望的原图框翻译成网络输入坐标

    ratio = min(size/w, size/h)；pad = (size - round(边长*ratio)) // 2
    """
    ratio = min(float(input_size) / IMG_W, float(input_size) / IMG_H)
    new_w = max(1, int(round(IMG_W * ratio)))
    new_h = max(1, int(round(IMG_H * ratio)))
    return ratio, (input_size - new_w) // 2, (input_size - new_h) // 2


def _to_input_space(box, input_size):
    """原图坐标 -> letterbox 后的网络输入坐标"""
    ratio, pad_x, pad_y = _letterbox_params(input_size)
    return [box[0] * ratio + pad_x, box[1] * ratio + pad_y,
            box[2] * ratio + pad_x, box[3] * ratio + pad_y]


class _FakeBox:
    def __init__(self, class_id, confidence, xyxy):
        self.cls = np.array([class_id], dtype=float)
        self.conf = np.array([confidence], dtype=float)
        self.xyxy = np.array([xyxy], dtype=float)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def _image_base64(width=IMG_W, height=IMG_H):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 1] = 160
    ok, buf = cv2.imencode('.jpg', img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode('ascii')


def _assert_box_close(actual, expected, tol=1):
    assert len(actual) == 4, actual
    assert all(abs(a - e) <= tol for a, e in zip(actual, expected)), (actual, expected)


def _write_test_jpeg(name='shot.jpg'):
    os.makedirs(config.ALERT_SCREENSHOT_DIR, exist_ok=True)
    path = os.path.join(config.ALERT_SCREENSHOT_DIR, name)
    assert cv2.imwrite(path, np.zeros((8, 8, 3), dtype=np.uint8))
    return path


@pytest.fixture()
def detect_calls(monkeypatch):
    """把 YOLO 推理换成桩：记录入参，并按"网络输入坐标"返回固定的三个框

    class_id 取自 config.CLASS_NAMES 的下标：1=close_eye 4=face 7=phone。
    桩按实际收到的画布边长换算坐标，因此在 input_size 改变时依然自洽。
    """
    calls = []

    def fake_detect_objects(image, conf=None, iou=None, max_det=None, input_size=None):
        net_size = int(image.shape[0])
        calls.append({
            'shape': tuple(image.shape),
            'conf': conf, 'iou': iou,
            'max_det': max_det, 'input_size': input_size,
        })
        return [_FakeResult([
            _FakeBox(4, 0.95, _to_input_space(FACE_BOX, net_size)),
            _FakeBox(1, 0.90, _to_input_space(CLOSE_EYE_BOX, net_size)),
            _FakeBox(7, 0.85, _to_input_space(PHONE_BOX, net_size)),
        ])]

    monkeypatch.setattr(detector, 'detect_objects', fake_detect_objects)
    return calls


@pytest.fixture()
def client(detect_calls):
    """test_client；每次重置跨请求内存状态（tracker / 去重哈希 / 分析器）"""
    app = app_module.create_app()
    app.config['TESTING'] = True
    api_module.reset_runtime_state()
    with app.test_client() as c:
        yield c


# --------------------------------------------------------------------------
# /api/health
# --------------------------------------------------------------------------
def test_health_reports_real_model_state(client, monkeypatch):
    monkeypatch.setattr(detector, 'get_model_info', lambda: {
        'loaded': True,
        'model_path': 'D:\\models\\weights\\best.pt',
        'device': 'cpu',
        'num_classes': 8,
        'classes': list(config.CLASS_NAMES),
    })
    r = client.get('/api/health')
    assert r.status_code == 200
    data = r.get_json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True
    assert data['model_path'].endswith('best.pt')
    assert data['device'] == 'cpu'
    assert data['num_classes'] == 8
    assert data['classes'] == list(config.CLASS_NAMES)


def test_health_degraded_when_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(detector, 'get_model_info', lambda: {
        'loaded': False, 'model_path': None, 'device': 'unknown',
        'num_classes': 0, 'classes': [],
    })
    r = client.get('/api/health')
    assert r.status_code == 200, '模型未加载也必须返回 200，而不是 5xx'
    data = r.get_json()
    assert data['status'] == 'degraded'
    assert data['model_loaded'] is False
    assert data['classes'] == []


def test_health_does_not_trigger_model_loading(client):
    """健康检查必须是纯读：测试进程从未加载模型，调用后也不能变"""
    assert detector.is_model_loaded() is False
    assert client.get('/api/health').status_code == 200
    assert detector.is_model_loaded() is False


# --------------------------------------------------------------------------
# /api/detect
# --------------------------------------------------------------------------
def test_detect_returns_original_image_coordinates(client, detect_calls):
    r = client.post('/api/detect', json={'image': _image_base64()})
    assert r.status_code == 200
    data = r.get_json()

    assert data['success'] is True
    assert data['image_width'] == IMG_W
    assert data['image_height'] == IMG_H
    assert data['input_size'] == config.DEFAULT_INPUT_SIZE == 640

    by_class = {d['class']: d for d in data['detections']}
    assert sorted(by_class) == ['close_eye', 'face', 'phone']

    for cls, expected in (('face', FACE_BOX),
                          ('close_eye', CLOSE_EYE_BOX),
                          ('phone', PHONE_BOX)):
        bbox = by_class[cls]['bbox']
        _assert_box_close(bbox, expected)
        x1, y1, x2, y2 = bbox
        # 关键断言：bbox 必须是原图坐标，不能是 640 空间坐标
        assert 0 <= x1 < x2 <= IMG_W
        assert 0 <= y1 < y2 <= IMG_H

    assert by_class['face']['danger_level'] == 'low'
    assert by_class['phone']['danger_level'] == 'high'
    assert 'track_id' in by_class['face']

    # 送进模型的是 letterbox 后的 640x640 输入
    assert detect_calls[-1]['shape'] == (640, 640, 3)
    assert detect_calls[-1]['input_size'] == 640


def test_detect_forwards_and_clamps_conf_iou(client, detect_calls):
    client.post('/api/detect', json={'image': _image_base64(), 'conf': 0.42, 'iou': 0.33})
    assert detect_calls[-1]['conf'] == pytest.approx(0.42)
    assert detect_calls[-1]['iou'] == pytest.approx(0.33)

    # 超范围夹取到 [0, 1]
    client.post('/api/detect', json={'image': _image_base64(), 'conf': 5, 'iou': -3})
    assert detect_calls[-1]['conf'] == 1.0
    assert detect_calls[-1]['iou'] == 0.0

    # conf=0 必须忠实生效（不能被默认值顶掉）
    client.post('/api/detect', json={'image': _image_base64(), 'conf': 0})
    assert detect_calls[-1]['conf'] == 0.0


def test_detect_rejects_bad_params_and_images(client, detect_calls):
    assert client.post('/api/detect', json={}).status_code == 400
    assert client.post('/api/detect', json={'image': ''}).status_code == 400
    assert client.post('/api/detect', json={'image': 'not-base64!!'}).status_code == 400
    assert client.post('/api/detect', json=[1, 2, 3]).status_code == 400

    r = client.post('/api/detect', json={'image': _image_base64(), 'conf': 'high'})
    assert r.status_code == 400
    assert 'error' in r.get_json()

    r = client.post('/api/detect', json={'image': _image_base64(), 'iou': True})
    assert r.status_code == 400

    # 被拒绝的请求一律不得触达模型
    assert detect_calls == []


def test_detect_writes_detections_once_per_change(client, detect_calls):
    """帧去重：检测结果无变化时只写一次库"""
    for _ in range(3):
        assert client.post('/api/detect', json={'image': _image_base64()}).status_code == 200
    assert len(database.query_detections(100)) == 3  # 3 个框，仅第一帧写入
    assert len(detect_calls) == 3


def test_detect_max_detections_follows_settings(client, detect_calls):
    assert client.put('/api/settings', json={'max_detections': 3}).status_code == 200
    client.post('/api/detect', json={'image': _image_base64()})
    assert detect_calls[-1]['max_det'] == 3


# --------------------------------------------------------------------------
# 告警 + 截图（P0-4）
# --------------------------------------------------------------------------
def test_detect_writes_readable_alert_screenshot(client, detect_calls):
    """P0-4 回归：告警截图写盘后必须能通过 /api/screenshots 读回"""
    responses = []
    for _ in range(3):  # eye_close 默认阈值 3 帧
        responses.append(client.post(
            '/api/detect', json={'image': _image_base64()}).get_json())

    triggered = [a for r in responses for a in r['alerts'] if a['status'] == 'triggered']
    assert len(triggered) == 1
    assert triggered[0]['type'] == 'eye_close'
    assert triggered[0]['frame_count'] == 3

    alerts = client.get('/api/history').get_json()['alerts']
    assert len(alerts) == 1
    url = alerts[0]['screenshot_url']
    assert url.startswith('/api/screenshots/')
    assert alerts[0]['screenshot_path']

    shot = client.get(url)
    assert shot.status_code == 200
    assert shot.mimetype == 'image/jpeg'
    assert shot.data[:2] == b'\xff\xd8'  # JPEG SOI
    assert os.path.isfile(os.path.join(config.ALERT_SCREENSHOT_DIR,
                                       os.path.basename(url)))


def test_save_alert_screenshot_returns_empty_on_failure(monkeypatch):
    """cv2.imwrite 失败必须返回空字符串（不能把不存在的路径静默写库）"""
    monkeypatch.setattr(api_module.cv2, 'imwrite', lambda *a, **k: False)
    assert api_module._save_alert_screenshot(np.zeros((4, 4, 3), dtype=np.uint8)) == ''


# --------------------------------------------------------------------------
# /api/history
# --------------------------------------------------------------------------
def test_history_alert_rows_expose_screenshot_url(client):
    database.insert_alerts(
        [AlertEvent('eye_close', 'triggered', '眼睛闭合', 3)],
        {'eye_status': 'close'},
        os.path.join(config.ALERT_SCREENSHOT_DIR, 'shot_1.jpg'),
    )
    database.insert_alerts([AlertEvent('yawn', 'triggered', '哈欠', 2)], {}, '')

    rows = client.get('/api/history').get_json()['alerts']
    by_type = {r['alert_type']: r for r in rows}
    assert len(rows) == 2
    assert by_type['eye_close']['screenshot_url'] == '/api/screenshots/shot_1.jpg'
    # 没有截图的历史行不编造 URL
    assert by_type['yawn']['screenshot_url'] == ''


def test_history_time_range_and_filters(client):
    database.insert_detections([
        {'class': 'face', 'confidence': 0.9, 'danger_level': 'low',
         'track_id': 1, 'bbox': [0, 0, 100, 100]},
    ], '2026-01-01T00:00:00')
    database.insert_detections([
        {'class': 'phone', 'confidence': 0.8, 'danger_level': 'high',
         'track_id': 2, 'bbox': [10, 10, 60, 90]},
    ], '2026-06-01T00:00:00')
    database.insert_alerts(
        [AlertEvent('eye_close', 'triggered', '眼睛闭合', 3)],
        {'eye_status': 'close'},
        '/x/shot_2.jpg',
    )

    data = client.get('/api/history?start=2025-12-31T00:00:00&end=2026-01-02T00:00:00'
                      ).get_json()
    assert [d['class_name'] for d in data['detections']] == ['face']
    assert data['alerts'] == []

    all_data = client.get('/api/history?limit=100').get_json()
    assert {d['class_name'] for d in all_data['detections']} == {'face', 'phone'}
    assert all_data['alerts'][0]['screenshot_url'] == '/api/screenshots/shot_2.jpg'

    filtered = client.get('/api/history?class=phone').get_json()
    assert [d['class_name'] for d in filtered['detections']] == ['phone']

    filtered = client.get('/api/history?alert_type=yawn').get_json()
    assert filtered['alerts'] == []


def test_history_stats_and_trend(client):
    database.insert_detections([{
        'class': 'face', 'confidence': 0.9, 'danger_level': 'low',
        'track_id': 1, 'bbox': [0, 0, 100, 100]}])
    stats = client.get('/api/history?stats=1').get_json()
    assert stats['total_detections'] == 1
    assert stats['detections_by_class'] == {'face': 1}

    trend = client.get('/api/history?trend=1&days=7').get_json()
    assert set(trend) == {'detection_trend', 'alert_trend'}


def test_delete_history_resets_frame_dedup(client, detect_calls):
    """清空历史后必须恢复写入

    否则 ``_last_detection_hash`` 仍是清空前的值，画面不变时下一帧被判为
    “结果无变化”而永久跳过，用户会以为清空后系统不再记录任何数据。
    """
    client.post('/api/detect', json={'image': _image_base64()})
    assert len(database.query_detections(100)) == 3

    # 同一画面再来一帧：被去重，不重复写
    client.post('/api/detect', json={'image': _image_base64()})
    assert len(database.query_detections(100)) == 3

    assert client.delete('/api/history?scope=all').status_code == 200
    assert database.query_detections(100) == []

    # 清空之后同一画面必须重新入库
    client.post('/api/detect', json={'image': _image_base64()})
    assert len(database.query_detections(100)) == 3


def test_delete_history_scopes(client):
    def seed(times=1):
        for _ in range(times):
            database.insert_detections([{
                'class': 'face', 'confidence': 0.9, 'danger_level': 'low',
                'track_id': 1, 'bbox': [0, 0, 100, 100]}])
            database.insert_alerts([AlertEvent('yawn', 'triggered', '哈欠', 2)], {}, '')

    def reset():
        client.delete('/api/history?scope=all')

    # scope=detections：只清检测，告警保留
    reset()
    seed(1)
    r = client.delete('/api/history?scope=detections')
    assert r.status_code == 200
    assert r.get_json() == {'success': True, 'deleted': {'detections': 1, 'alerts': 0}}
    assert client.get('/api/history').get_json()['detections'] == []
    assert len(client.get('/api/history').get_json()['alerts']) == 1

    # scope=alerts：只清告警，检测保留
    reset()
    seed(2)
    r = client.delete('/api/history?scope=alerts')
    assert r.get_json() == {'success': True, 'deleted': {'detections': 0, 'alerts': 2}}
    assert client.get('/api/history').get_json()['alerts'] == []
    assert len(client.get('/api/history').get_json()['detections']) == 2

    # 默认 scope=all
    reset()
    seed(3)
    r = client.delete('/api/history')
    assert r.get_json() == {'success': True, 'deleted': {'detections': 3, 'alerts': 3}}
    assert client.get('/api/history').get_json() == {'detections': [], 'alerts': []}

    # 非法 scope：400，且不得删除任何数据
    reset()
    seed(1)
    r = client.delete('/api/history?scope=everything')
    assert r.status_code == 400
    assert 'error' in r.get_json()
    assert len(client.get('/api/history').get_json()['detections']) == 1
    assert len(client.get('/api/history').get_json()['alerts']) == 1


# --------------------------------------------------------------------------
# /api/settings
# --------------------------------------------------------------------------
def test_settings_defaults(client):
    r = client.get('/api/settings')
    assert r.status_code == 200
    data = r.get_json()
    assert data['success'] is True
    assert data['settings'] == dict(config.DEFAULT_SETTINGS)
    assert data['settings'] == database.default_settings()


def test_settings_put_then_get_roundtrip(client):
    r = client.put('/api/settings', json={'conf_threshold': 0.55, 'cooldown_frames': 12})
    assert r.status_code == 200
    body = r.get_json()
    assert body['success'] is True
    assert body['settings']['conf_threshold'] == pytest.approx(0.55)
    assert body['settings']['cooldown_frames'] == 12

    again = client.get('/api/settings').get_json()['settings']
    assert again['conf_threshold'] == pytest.approx(0.55)
    assert again['cooldown_frames'] == 12
    assert again['yawn_frames'] == config.DEFAULT_SETTINGS['yawn_frames']


def test_settings_rejects_invalid_payloads(client):
    bad_payloads = [
        {'evil': 1},                    # 不在白名单
        {'conf_threshold': 'high'},     # 类型错误
        {'conf_threshold': True},       # bool 不是数值
        {'conf_threshold': 1.5},        # 超范围
        {'eye_close_frames': 0},        # 低于下限
        {'input_size': 99999},          # 高于上限
        {'input_size': 640.5},          # 整数字段给了小数
        {},
        [],
    ]
    for payload in bad_payloads:
        r = client.put('/api/settings', json=payload)
        assert r.status_code == 400, payload
        assert 'error' in r.get_json()

    # 非法请求不能留下任何痕迹
    assert client.get('/api/settings').get_json()['settings'] == dict(config.DEFAULT_SETTINGS)


def test_settings_conf_threshold_becomes_detect_default(client, detect_calls):
    assert client.put('/api/settings', json={'conf_threshold': 0.77}).status_code == 200
    client.post('/api/detect', json={'image': _image_base64()})
    assert detect_calls[-1]['conf'] == pytest.approx(0.77)

    # 请求里显式传的值优先
    client.post('/api/detect', json={'image': _image_base64(), 'conf': 0.11})
    assert detect_calls[-1]['conf'] == pytest.approx(0.11)


def test_settings_rebuilds_analyzer_thresholds(client, detect_calls):
    """改 eye_close_frames 后 /api/detect 必须真的按新阈值告警（重建 _analyzer）"""
    assert client.put('/api/settings', json={'eye_close_frames': 5}).status_code == 200

    for _ in range(4):
        r = client.post('/api/detect', json={'image': _image_base64()})
        assert r.get_json()['alerts'] == [], '阈值 5 时第 4 帧不应告警'

    r = client.post('/api/detect', json={'image': _image_base64()})
    alerts = r.get_json()['alerts']
    triggered = {a['type']: a for a in alerts if a['status'] == 'triggered'}
    # 桩里还带一个 phone 框，第 5 帧 distracted 也会触发，因此按类型取
    assert 'eye_close' in triggered
    assert triggered['eye_close']['type'] == 'eye_close'
    assert triggered['eye_close']['status'] == 'triggered'
    assert triggered['eye_close']['frame_count'] == 5
    assert r.get_json()['behavior_state']['eye_status'] == 'close'


def test_settings_input_size_changes_letterbox_input(client, detect_calls):
    assert client.put('/api/settings', json={'input_size': 320}).status_code == 200
    r = client.post('/api/detect', json={'image': _image_base64()})
    data = r.get_json()
    assert data['input_size'] == 320
    assert detect_calls[-1]['shape'] == (320, 320, 3)
    # 坐标仍然是原图坐标（与 input_size 无关）
    by_class = {d['class']: d for d in data['detections']}
    _assert_box_close(by_class['face']['bbox'], FACE_BOX)
    _assert_box_close(by_class['close_eye']['bbox'], CLOSE_EYE_BOX)


# --------------------------------------------------------------------------
# /api/screenshots
# --------------------------------------------------------------------------
def test_screenshot_endpoint_serves_image(client):
    _write_test_jpeg('shot.jpg')
    r = client.get('/api/screenshots/shot.jpg')
    assert r.status_code == 200
    assert r.mimetype == 'image/jpeg'
    assert r.data[:2] == b'\xff\xd8'


def test_screenshot_missing_file_returns_json_404(client):
    r = client.get('/api/screenshots/nope.jpg')
    assert r.status_code == 404
    assert r.mimetype == 'application/json'
    assert 'error' in r.get_json()


def test_screenshot_path_traversal_is_rejected(client):
    """三重防线：basename / 扩展名 / 目录归属"""
    for bad in ('../config.py', '..\\config.py', 'sub/shot.jpg', '..%2Fconfig.py',
                '/etc/passwd', 'shot.txt', 'shot.jpg.exe', '..', '.', ''):
        assert _safe_screenshot_path(bad) is None, bad

    # 合法文件名仍能解析
    good = _safe_screenshot_path('shot.jpg')
    assert good is not None
    assert os.path.dirname(good) == os.path.abspath(config.ALERT_SCREENSHOT_DIR)

    # HTTP 层同样被拒（400 或路由层 404，都不得真的返回文件）
    _write_test_jpeg('shot.jpg')
    for bad_url in ('/api/screenshots/..%2Fconfig.py',
                    '/api/screenshots/../backend/config.py',
                    '/api/screenshots/shot.txt',
                    '/api/screenshots/..%5C..%5Cconfig.py'):
        r = client.get(bad_url)
        assert r.status_code in (400, 404), (bad_url, r.status_code)
        assert r.mimetype == 'application/json', (bad_url, r.mimetype)


# --------------------------------------------------------------------------
# /api/* 的 JSON 错误语义 + 非 API 静态回落
# --------------------------------------------------------------------------
def test_unknown_api_path_returns_json_404(client):
    r = client.get('/api/does_not_exist')
    assert r.status_code == 404
    assert r.mimetype == 'application/json'
    assert 'error' in r.get_json()

    r = client.get('/api')
    assert r.status_code == 404
    assert r.mimetype == 'application/json'


def test_wrong_method_on_api_returns_json_405(client):
    r = client.get('/api/detect')       # 只允许 POST
    assert r.status_code == 405
    assert r.mimetype == 'application/json'
    assert 'error' in r.get_json()

    r = client.post('/api/health')      # 只允许 GET
    assert r.status_code == 405
    assert r.mimetype == 'application/json'

    r = client.put('/api/history')      # 只允许 GET / DELETE
    assert r.status_code == 405
    assert r.mimetype == 'application/json'


def test_non_api_paths_still_fall_back_to_static_pages(client):
    r = client.get('/')
    assert r.status_code == 200
    assert r.mimetype == 'text/html'

    r = client.get('/detect.html')
    assert r.status_code == 200
    assert r.mimetype == 'text/html'

    # 未命中的非 API 路径仍然回落 index.html（原有行为，未被 /api 的 JSON 404 影响）
    r = client.get('/definitely/not/a/page')
    assert r.status_code == 200
    assert r.mimetype == 'text/html'


def test_settings_get_output_can_always_be_put_back(client):
    """不变量回归（Lead 批准方案 A）：伪造越界值入库 -> GET 回退默认 ->
    把 GET 的**完整输出原样** PUT 回去 -> 必须 200，而不是 400。
    """
    db = database.get_db()
    dirty = {
        'conf_threshold': 9.0,
        'input_size': 99999,
        'cooldown_frames': -5,
        'max_detections': 0,
    }
    for key, value in dirty.items():
        db.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
                   (key, json.dumps(value)))
    db.commit()

    got = client.get('/api/settings').get_json()['settings']
    for key in dirty:
        assert got[key] == config.DEFAULT_SETTINGS[key], key

    # 前端就是这么做的：把 GET 到的 9 键完整回传
    r = client.put('/api/settings', json=got)
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['settings'] == dict(config.DEFAULT_SETTINGS)

    # 回传后仍然稳定（幂等）
    again = client.get('/api/settings').get_json()['settings']
    assert client.put('/api/settings', json=again).status_code == 200
