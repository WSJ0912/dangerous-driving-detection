#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""智慧驭行驾驶行为检测系统 - 应用入口

职责：
- Flask 应用工厂 ``create_app()``
- ``/api/*`` 下的 4xx/5xx 统一返回 JSON（不要回落成 HTML 或 index.html）
- 非 ``/api`` 路径保持原有静态页面回落（未命中的路径返回 index.html）
- 启动不因模型/数据库不可用而失败：服务照常起来，``/api/health`` 如实返回 degraded
"""
import logging
import os
import sys

# 确保项目根目录在 Python 路径中（直接 `python backend/app.py` 运行时需要）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.routing import PathConverter

from backend import config


logger = logging.getLogger(__name__)

# 需要以 JSON 形式回给接口调用方的错误码
_API_ERROR_CODES = (400, 404, 405, 413, 500)


class _NonApiPathConverter(PathConverter):
    """匹配任意路径，但排除 ``/api`` 命名空间

    静态回落路由如果连 ``/api/*`` 一起匹配，会产生两个错误语义：
    ``GET /api/detect``（存在但方法只允许 POST）会命中回落路由而返回 404/HTML，
    而不是 405。这里在**路由层**就把 ``/api`` 排除掉，于是：
    - 未知 /api 路径        -> 404，由 errorhandler 转成 JSON
    - 存在但方法不对的 /api -> 405，由 errorhandler 转成 JSON
    """
    regex = r'(?!api(?:/|$))[^/].*?'


def _is_api_path(path):
    """判断请求路径是否属于 /api 命名空间"""
    return path == '/api' or path.startswith('/api/')


def _register_api_error_handlers(app):
    """让 ``/api/*`` 的错误返回 JSON，其它路径保持 Flask 默认行为

    没有这层处理时，``GET /api/nope`` 会命中静态页面回落路由并把 index.html
    返回给接口调用方，``GET /api/detect``（方法不对）会返回 HTML 错误页。
    """

    def make_handler(status_code):
        def handler(error):
            if _is_api_path(request.path):
                if status_code >= 500:
                    logger.error(
                        'API 未处理异常 %s: %s', request.path,
                        getattr(error, 'original_exception', error))
                description = getattr(error, 'description', None) or 'Request failed'
                return jsonify({'error': description, 'status': status_code}), status_code
            # 非 /api 路径：交回 Flask 默认处理，保留原有语义
            return error

        handler.__name__ = 'api_error_handler_%d' % status_code
        return handler

    for code in _API_ERROR_CODES:
        app.register_error_handler(code, make_handler(code))


def create_app():
    """Flask 应用工厂"""
    app = Flask(
        __name__,
        static_folder=config.STATIC_FOLDER,
        template_folder=config.TEMPLATE_FOLDER,
    )
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
    CORS(app)

    from backend.routes import api as api_module
    app.register_blueprint(api_module.api_bp)
    _register_api_error_handlers(app)

    # 启动时把运行时设置（conf 默认值 / 告警阈值 / 冷却帧数）应用到内存状态。
    # 数据库不可用时退回默认设置，不阻塞服务启动。
    try:
        api_module.reset_runtime_state()
    except Exception as exc:
        logger.warning('运行时设置初始化失败，使用默认值: %s', exc)

    # 静态页面回落：由 _NonApiPathConverter 保证 /api/* 不会命中这里
    app.url_map.converters['nonapi'] = _NonApiPathConverter

    @app.route('/', defaults={'path': ''})
    @app.route('/<nonapi:path>')
    def serve_frontend(path):
        # 兜底（正常情况下 /api/* 已被路由层排除）：未命中的 /api 路径返回 JSON 404
        if _is_api_path('/' + path):
            return jsonify({'error': 'Not Found', 'status': 404}), 404
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, 'index.html')

    return app


def main():
    # 确保数据目录与告警截图目录存在
    os.makedirs(config.ALERT_SCREENSHOT_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # 初始化数据库（失败不阻塞启动）
    try:
        from backend.database import get_db
        get_db()
    except Exception as e:
        print(f"[WARN] 数据库初始化失败: {e}")

    # 预加载模型；失败不阻塞启动，/api/health 会如实返回 degraded
    try:
        from backend.models.detector import get_model
        get_model()
    except ImportError:
        print("[WARN] PyTorch 未安装，模型检测功能不可用")
    except Exception as e:
        print(f"[WARN] 模型加载失败: {e}（服务仍会启动，/api/health 将返回 degraded）")

    app = create_app()
    print(f"服务地址: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)


if __name__ == '__main__':
    main()