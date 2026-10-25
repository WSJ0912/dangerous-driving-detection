# -*- coding: utf-8 -*-
"""pytest 全局配置：测试隔离

``isolate_paths`` 是 autouse fixture，会把**所有**可写路径重定向到 pytest 的
``tmp_path``：

- ``config.DATA_DIR`` / ``config.DB_PATH``      -> 临时目录下的独立 sqlite 库
- ``config.ALERT_SCREENSHOT_DIR``               -> 临时目录下的截图目录

并丢弃 ``backend.database`` 的模块级连接单例。因此跑 ``pytest`` 不会读写生产库
``backend/data/detection.db``，也不会往 ``backend/uploads/alerts`` 里塞测试图片。

（旧版测试直接调用 ``get_db()``，而 ``_conn`` 是模块级单例，
 结果每次跑测试都往生产库里插数据。）
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend import config
from backend import database


@pytest.fixture(autouse=True)
def isolate_paths(tmp_path, monkeypatch):
    """把数据库与截图目录重定向到 tmp_path（每个测试都隔离）"""
    data_dir = tmp_path / 'data'
    monkeypatch.setattr(config, 'DATA_DIR', str(data_dir))
    monkeypatch.setattr(config, 'DB_PATH', str(data_dir / 'detection.db'))
    monkeypatch.setattr(config, 'ALERT_SCREENSHOT_DIR',
                        str(tmp_path / 'uploads' / 'alerts'))

    # 关键：丢弃模块级连接单例，否则 get_db() 仍返回指向重定向之前那个库的连接
    database.reset_connection()
    yield
    database.reset_connection()