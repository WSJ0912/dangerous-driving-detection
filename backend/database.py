# -*- coding: utf-8 -*-
"""SQLite 数据库：检测记录、告警记录与应用设置的持久化

约定与注意事项：
- 连接是模块级单例（``_conn``）。若要把 ``config.DB_PATH`` 切到别的库
  （测试隔离、备份切换等），**必须先调用** :func:`reset_connection` 丢弃旧连接，
  否则 ``get_db()`` 仍会返回指向旧库的连接。
- ``app_settings`` 表存放 ``GET/PUT /api/settings`` 的可调项，键与
  ``config.DEFAULT_SETTINGS`` 保持一致，值以 JSON 文本存储（保留类型信息）。
"""
import json
import os
import sqlite3
from datetime import datetime

from backend import config


_conn = None

# delete_records 允许的范围
DELETE_SCOPES = ('all', 'detections', 'alerts')


def get_db():
    """获取数据库连接（懒初始化，自动建表）"""
    global _conn
    if _conn is not None:
        return _conn

    os.makedirs(config.DATA_DIR, exist_ok=True)
    _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL")
    _init_schema(_conn)
    return _conn


def reset_connection():
    """关闭并丢弃模块级连接单例

    测试把 ``config.DB_PATH`` 重定向到临时目录后必须调用本函数，
    否则会继续读写重定向之前的数据库（这正是旧测试污染生产库的原因）。
    """
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None


def _init_schema(conn):
    """创建表结构"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detection_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL,
            danger_level TEXT,
            track_id INTEGER,
            bbox_x1 INTEGER,
            bbox_y1 INTEGER,
            bbox_x2 INTEGER,
            bbox_y2 INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alert_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            message TEXT NOT NULL,
            screenshot_path TEXT,
            behavior_state TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()


def insert_detections(detections, timestamp=None):
    """批量插入检测记录"""
    ts = timestamp or datetime.now().isoformat()
    db = get_db()
    rows = []
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        rows.append((
            ts, d['class'], d['confidence'],
            d.get('danger_level', 'low'), d.get('track_id'),
            x1, y1, x2, y2,
        ))
    db.executemany(
        """INSERT INTO detection_records
           (timestamp, class_name, confidence, danger_level, track_id,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    db.commit()


def insert_alerts(alerts, behavior_state, screenshot_path=""):
    """批量插入告警记录（AlertEvent list）

    Args:
        alerts: AlertEvent 列表（只应包含 triggered / resolved 事件）
        behavior_state: BehaviorAnalyzer.state
        screenshot_path: 截图文件名或路径；截图写入失败时传空字符串
    """
    if not alerts:
        return
    ts = datetime.now().isoformat()
    db = get_db()
    rows = [
        (ts, a.type, a.message, screenshot_path,
         json.dumps(behavior_state, ensure_ascii=False))
        for a in alerts
    ]
    db.executemany(
        """INSERT INTO alert_records
           (timestamp, alert_type, message, screenshot_path, behavior_state)
           VALUES (?, ?, ?, ?, ?)""",
        rows,
    )
    db.commit()


def query_detections(limit=50, offset=0, class_name=None):
    """查询检测记录，支持按类别过滤"""
    db = get_db()
    if class_name:
        rows = db.execute(
            "SELECT * FROM detection_records WHERE class_name = ? "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            (class_name, limit, offset),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM detection_records ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def query_alerts(limit=50, offset=0, alert_type=None):
    """查询告警记录，支持按类型过滤"""
    db = get_db()
    if alert_type:
        rows = db.execute(
            "SELECT * FROM alert_records WHERE alert_type = ? "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            (alert_type, limit, offset),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM alert_records ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def query_alerts_by_time(start_ts, end_ts, limit=100, offset=0):
    """按时间范围查询告警"""
    db = get_db()
    rows = db.execute(
        "SELECT * FROM alert_records WHERE timestamp >= ? AND timestamp <= ? "
        "ORDER BY id DESC LIMIT ? OFFSET ?",
        (start_ts, end_ts, limit, offset),
    ).fetchall()
    return [dict(r) for r in rows]


def query_detections_by_time(start_ts, end_ts, limit=100, offset=0):
    """按时间范围查询检测记录"""
    db = get_db()
    rows = db.execute(
        "SELECT * FROM detection_records WHERE timestamp >= ? AND timestamp <= ? "
        "ORDER BY id DESC LIMIT ? OFFSET ?",
        (start_ts, end_ts, limit, offset),
    ).fetchall()
    return [dict(r) for r in rows]


def get_daily_stats(days=7):
    """获取按天统计的检测趋势数据"""
    db = get_db()
    rows = db.execute(
        "SELECT DATE(timestamp) as day, COUNT(*) as cnt "
        "FROM detection_records "
        "WHERE timestamp >= DATE('now', ?) "
        "GROUP BY day ORDER BY day ASC",
        (f'-{days} days',),
    ).fetchall()
    detection_trend = {r['day']: r['cnt'] for r in rows}

    rows = db.execute(
        "SELECT DATE(timestamp) as day, COUNT(*) as cnt "
        "FROM alert_records "
        "WHERE timestamp >= DATE('now', ?) "
        "GROUP BY day ORDER BY day ASC",
        (f'-{days} days',),
    ).fetchall()
    alert_trend = {r['day']: r['cnt'] for r in rows}

    return {
        'detection_trend': detection_trend,
        'alert_trend': alert_trend,
    }


def get_detection_stats():
    """获取检测统计摘要"""
    db = get_db()
    total = db.execute("SELECT COUNT(*) as cnt FROM detection_records").fetchone()['cnt']
    total_alerts = db.execute("SELECT COUNT(*) as cnt FROM alert_records").fetchone()['cnt']
    by_class = {}
    for row in db.execute(
        "SELECT class_name, COUNT(*) as cnt FROM detection_records GROUP BY class_name"
    ).fetchall():
        by_class[row['class_name']] = row['cnt']
    by_alert_type = {}
    for row in db.execute(
        "SELECT alert_type, COUNT(*) as cnt FROM alert_records GROUP BY alert_type"
    ).fetchall():
        by_alert_type[row['alert_type']] = row['cnt']
    return {
        'total_detections': total,
        'total_alerts': total_alerts,
        'detections_by_class': by_class,
        'alerts_by_type': by_alert_type,
    }


def delete_records(scope='all'):
    """按范围清空历史记录（DELETE /api/history 的持久化实现）

    Args:
        scope: ``'all'`` 全部 | ``'detections'`` 仅检测记录 | ``'alerts'`` 仅告警记录

    Returns:
        dict: 实际删除的行数，形如 ``{'detections': 12, 'alerts': 3}``

    Raises:
        ValueError: scope 不在允许范围内
    """
    if scope not in DELETE_SCOPES:
        raise ValueError(f"Invalid scope: {scope}")

    db = get_db()
    deleted = {'detections': 0, 'alerts': 0}
    if scope in ('all', 'detections'):
        deleted['detections'] = db.execute("DELETE FROM detection_records").rowcount
    if scope in ('all', 'alerts'):
        deleted['alerts'] = db.execute("DELETE FROM alert_records").rowcount
    db.commit()
    return deleted


def default_settings():
    """设置的默认值（唯一来源：config.DEFAULT_SETTINGS）"""
    return dict(config.DEFAULT_SETTINGS)


def _settings_ranges():
    """config.SETTINGS_RANGES（冻结契约）；缺失时退化为无范围限制"""
    ranges = getattr(config, 'SETTINGS_RANGES', None)
    return ranges if isinstance(ranges, dict) else {}


def _clamp_to_range(key, value, default_value):
    """越界即回退默认值

    不变量：``GET /api/settings`` 的输出必须**永远**是合法的 ``PUT`` 输入。
    若 app_settings 被绕过 API 改写为超范围值（或未来 SETTINGS_RANGES 收窄），
    把原值原样返回会让前端「保存时回传全量设置」被 PUT 一律 400，
    设置页将永久保存失败。
    """
    lower, upper = _settings_ranges().get(key, (None, None))
    if lower is not None and value < lower:
        return default_value
    if upper is not None and value > upper:
        return default_value
    return value


def _decode_setting(key, default_value, raw):
    """把 app_settings 里的 JSON 文本还原成设置值

    两道回退，任何一道不通过都退回默认值：
    1) 类型/格式不符（脏数据、手工改库、旧版本遗留）；
    2) 超出 ``config.SETTINGS_RANGES``（保证 GET 的输出一定是合法 PUT 输入）。
    """
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return default_value

    if isinstance(default_value, bool):
        if not isinstance(value, bool):
            return default_value
    elif isinstance(default_value, int):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default_value
        if not float(value).is_integer():
            return default_value
        value = int(value)
    elif isinstance(default_value, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default_value
        value = float(value)
    elif not isinstance(value, type(default_value)):
        return default_value

    return _clamp_to_range(key, value, default_value)


def get_settings():
    """读取应用设置：config 默认值 + app_settings 表的覆盖值

    白名单之外的遗留键直接忽略，因此返回的键集合恒等于 DEFAULT_SETTINGS。
    """
    settings = default_settings()
    db = get_db()
    for row in db.execute("SELECT key, value FROM app_settings").fetchall():
        key = row['key']
        if key not in settings:
            continue
        settings[key] = _decode_setting(key, settings[key], row['value'])
    return settings


def set_settings(updates):
    """写入应用设置并返回写入后的完整设置

    调用方（API 层）负责白名单与范围校验；本函数只负责持久化。
    """
    if updates:
        db = get_db()
        db.executemany(
            "INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
            [(k, json.dumps(v, ensure_ascii=False)) for k, v in updates.items()],
        )
        db.commit()
    return get_settings()