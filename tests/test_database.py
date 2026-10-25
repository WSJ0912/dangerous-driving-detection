# -*- coding: utf-8 -*-
"""数据库持久化测试

所有用例都运行在 conftest.py 的 ``isolate_paths`` 隔离出的临时库上，
不会写入生产库 backend/data/detection.db。
（旧版把一个 sqlite 文件写在 tmp_path 却从未使用，并且直接往生产库写测试数据。）
"""
import json
import os

import pytest

from backend import config
from backend import database
from backend.database import (
    get_db, insert_detections, insert_alerts,
    query_detections, query_alerts, get_detection_stats,
    delete_records, get_settings, set_settings, default_settings,
)
from backend.services.behavior import AlertEvent


def _detection(cls='close_eye', conf=0.95, box=(10, 20, 50, 60), track_id=1):
    return {
        'class': cls, 'confidence': conf, 'danger_level': 'high',
        'track_id': track_id, 'bbox': list(box),
    }


def test_init():
    db = get_db()
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = [t['name'] for t in tables]
    assert 'detection_records' in names
    assert 'alert_records' in names
    assert 'app_settings' in names
    print(f'  Schema: OK (tables: {names})')


def test_db_is_isolated_from_production_db():
    """测试库必须落在 tmp_path，绝不能是生产库"""
    assert config.DB_PATH != os.path.join(
        config.BASE_DIR, 'data', 'detection.db')
    get_db()
    assert os.path.exists(config.DB_PATH)
    assert os.path.abspath(config.DB_PATH).startswith(
        os.path.abspath(str(config.DATA_DIR)))


def test_insert_detections():
    insert_detections([
        _detection('close_eye', 0.95, (10, 20, 50, 60), 1),
        _detection('face', 0.99, (0, 0, 100, 100), 0),
    ], '2026-01-01T00:00:00')

    rows = query_detections(10)
    assert len(rows) == 2

    eyes = query_detections(10, class_name='close_eye')
    assert len(eyes) == 1
    assert eyes[0]['bbox_x1'] == 10
    assert eyes[0]['bbox_y2'] == 60
    assert eyes[0]['danger_level'] == 'high'

    print(f'  Insert detections: OK ({len(rows)} rows, {len(eyes)} close_eye)')


def test_insert_detections_empty_is_noop():
    insert_detections([])
    assert query_detections(10) == []


def test_insert_alerts():
    insert_alerts(
        [AlertEvent('eye_close', 'triggered', 'eyes closed!', 5)],
        {'eye_status': 'close', 'eye_close_frames': 5},
        '/tmp/screenshot.jpg',
    )

    rows = query_alerts(10)
    assert len(rows) == 1
    assert rows[0]['alert_type'] == 'eye_close'
    assert rows[0]['screenshot_path'] == '/tmp/screenshot.jpg'

    state = json.loads(rows[0]['behavior_state'])
    assert state['eye_close_frames'] == 5

    print(f'  Insert alerts: OK ({len(rows)} rows)')


def test_insert_alerts_empty_is_noop():
    insert_alerts([], {}, '/tmp/screenshot.jpg')
    assert query_alerts(10) == []


def test_stats():
    insert_detections([_detection()], '2026-01-01T00:00:00')
    insert_alerts([AlertEvent('yawn', 'triggered', '哈欠', 2)], {}, '')
    stats = get_detection_stats()
    assert stats['total_detections'] == 1
    assert stats['total_alerts'] == 1
    assert stats['detections_by_class'] == {'close_eye': 1}
    assert stats['alerts_by_type'] == {'yawn': 1}
    print(f'  Stats: OK (dets={stats["total_detections"]}, alerts={stats["total_alerts"]})')


def test_daily_stats_shape():
    insert_detections([_detection()])
    trend = database.get_daily_stats(7)
    assert set(trend) == {'detection_trend', 'alert_trend'}


def test_delete_records_scopes():
    def seed_alerts():
        insert_alerts([AlertEvent('yawn', 'triggered', '哈欠', 2)], {}, '')

    # 仅删检测记录：告警保留
    insert_detections([_detection()], '2026-01-01T00:00:00')
    seed_alerts()
    assert delete_records('detections') == {'detections': 1, 'alerts': 0}
    assert query_detections(10) == []
    assert len(query_alerts(10)) == 1

    # 仅删告警记录：检测保留
    insert_detections([_detection()], '2026-01-01T00:00:00')
    assert delete_records('alerts') == {'detections': 0, 'alerts': 1}
    assert query_alerts(10) == []
    assert len(query_detections(10)) == 1

    # 全部删除
    seed_alerts()
    assert delete_records('all') == {'detections': 1, 'alerts': 1}
    assert query_detections(10) == []
    assert query_alerts(10) == []

    # 空库删除返回 0，不报错
    assert delete_records('all') == {'detections': 0, 'alerts': 0}


def test_delete_records_invalid_scope_raises():
    with pytest.raises(ValueError):
        delete_records('everything')


def test_settings_defaults_come_from_config():
    defaults = default_settings()
    assert defaults == dict(config.DEFAULT_SETTINGS)
    assert get_settings() == defaults


def test_settings_roundtrip_is_persisted_as_json():
    updated = set_settings({'conf_threshold': 0.66, 'eye_close_frames': 7})
    assert updated['conf_threshold'] == pytest.approx(0.66)
    assert updated['eye_close_frames'] == 7
    # 未改动的项保持默认值
    assert updated['yawn_frames'] == config.DEFAULT_SETTINGS['yawn_frames']

    # 值以 JSON 文本落库（保留类型）
    db = get_db()
    raw = db.execute(
        "SELECT value FROM app_settings WHERE key = ?", ('conf_threshold',)
    ).fetchone()['value']
    assert json.loads(raw) == pytest.approx(0.66)

    # 重新读取仍然生效（不是只存在于内存里）
    assert get_settings()['conf_threshold'] == pytest.approx(0.66)
    assert get_settings()['eye_close_frames'] == 7


def test_settings_overwrite():
    set_settings({'cooldown_frames': 11})
    assert get_settings()['cooldown_frames'] == 11
    set_settings({'cooldown_frames': 22})
    assert get_settings()['cooldown_frames'] == 22


def test_settings_ignores_legacy_unknown_keys():
    db = get_db()
    db.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
               ('not_a_real_setting', '123'))
    db.commit()
    assert 'not_a_real_setting' not in get_settings()
    assert set(get_settings()) == set(config.DEFAULT_SETTINGS)


def test_settings_recovers_from_dirty_values():
    """脏数据不能让 get_settings 抛异常或把类型带坏"""
    db = get_db()
    db.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
               ('conf_threshold', 'not-json'))
    db.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
               ('input_size', '"640"'))
    db.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
               ('max_detections', '3.5'))
    db.commit()

    settings = get_settings()
    assert settings['conf_threshold'] == config.DEFAULT_SETTINGS['conf_threshold']
    assert settings['input_size'] == config.DEFAULT_SETTINGS['input_size']
    assert settings['max_detections'] == config.DEFAULT_SETTINGS['max_detections']


def test_reset_connection_switches_database(tmp_path):
    """reset_connection 后 DB_PATH 切换必须真的换库"""
    insert_detections([_detection()], '2026-01-01T00:00:00')
    assert len(query_detections(10)) == 1

    other_dir = tmp_path / 'other'
    config.DATA_DIR = str(other_dir)
    config.DB_PATH = str(other_dir / 'detection.db')
    try:
        # 未重置单例：仍连在旧库上（这正是旧测试污染生产库的机制）
        assert len(query_detections(10)) == 1
        database.reset_connection()
        assert query_detections(10) == []
    finally:
        database.reset_connection()


def test_settings_get_output_is_always_valid_put_input():
    """不变量：GET /api/settings 的输出必须永远是合法 PUT 输入

    前置：绕过 API 直接往 app_settings 写越界值（模拟外部篡改，或未来
    SETTINGS_RANGES 收窄使历史合法值变成越界值）。GET 必须回退到默认值，
    而不是把越界值原样返回 —— 否则前端「保存时回传全量设置」会被
    PUT 一律 400，设置页永久保存失败。
    """
    db = get_db()
    dirty = {
        'conf_threshold': 9.0,      # 上限 1.0
        'input_size': 99999,        # 上限 1280
        'cooldown_frames': -5,      # 下限 0
        'max_detections': 0,        # 下限 1
    }
    for key, value in dirty.items():
        db.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
                   (key, json.dumps(value)))
    db.commit()

    settings = get_settings()
    for key in dirty:
        assert settings[key] == config.DEFAULT_SETTINGS[key], key
    # GET 的完整输出必须与默认值等价（即一定落在范围内、可被 PUT 接受）
    assert settings == dict(config.DEFAULT_SETTINGS)


def test_settings_range_boundary_values_are_kept():
    """只回退越界值，边界值必须原样保留（不能顺手误伤合法配置）"""
    ranges = config.SETTINGS_RANGES
    db = get_db()

    boundary = {
        'conf_threshold': ranges['conf_threshold'][0],   # 下界 0.01
        'cooldown_frames': ranges['cooldown_frames'][1],  # 上界 600
        'eye_close_frames': ranges['eye_close_frames'][0],  # 下界 1
    }
    for key, value in boundary.items():
        assert ranges[key][0] <= value <= ranges[key][1]
        db.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
                   (key, json.dumps(value)))
    db.commit()

    settings = get_settings()
    for key, value in boundary.items():
        assert settings[key] == value, key
