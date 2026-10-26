#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出权重为部署格式（ONNX / TorchScript）

说明：**生产后端目前只加载 .pt**（``backend/models/detector.py`` 用 ``YOLO(path)``），
所以本脚本是"为将来部署预留"，不要求接入后端。

用法::

    python training/export.py --weights training/experiments/run/weights/best.pt --format onnx
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    DEFAULT_START_WEIGHTS,
    PROJECT_ROOT,
    banner,
    resolve_device,
    resolve_weights,
    sha256_of,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='把 .pt 导出为 ONNX / TorchScript',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--weights', default='default')
    p.add_argument('--format', default='onnx', choices=('onnx', 'torchscript'))
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--out', default=None, help='导出文件放置目录（默认与权重同级的 export/）')
    p.add_argument('--device', default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    weights = resolve_weights(args.weights)
    banner('导出')
    print('权重   : %s' % weights)
    if not os.path.exists(weights):
        print('[错误] 权重不存在，无法导出。')
        return 2
    print('        %d B  sha256=%s' % (os.path.getsize(weights), sha256_of(weights)[:16]))

    device = resolve_device(args.device)
    print('格式   : %s' % args.format)
    print('imgsz  : %s' % args.imgsz)
    print('device : %s' % device)

    from ultralytics import YOLO

    model = YOLO(weights)
    try:
        exported = model.export(format=args.format, imgsz=args.imgsz, device=device)
    except ImportError as exc:
        print('\n[失败] 导出 %s 需要额外的依赖，当前环境缺失：' % args.format)
        print('        %s' % exc)
        print('\n        例如 ONNX 导出通常需要：pip install onnx onnxslim')
        print('        注意：本脚本已把 YOLO_AUTOINSTALL 设为 false，ultralytics 不会')
        print('              再自动帮你 pip install（那属于隐式改动环境，且需要联网）。')
        print('        本机（离线、无 GPU）不一定要装；这只是为将来部署预留。')
        return 2
    except Exception as exc:
        print('\n[失败] 导出出错：%s' % exc)
        return 2

    banner('导出结果')
    print('返回路径: %s' % exported)
    if not os.path.exists(exported):
        print('[错误] 导出的文件不存在。')
        return 2

    size = os.path.getsize(exported)
    print('大小    : %d B' % size)
    print('sha256  : %s' % sha256_of(exported))
    if size <= 0:
        print('[错误] 导出文件为空。')
        return 2

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        target = os.path.join(args.out, os.path.basename(exported))
        shutil.copy2(exported, target)
        print('已复制到: %s' % target)

    print('\n[提示] 生产后端当前只加载 .pt，本导出件供将来部署使用，无需放进 models/weights/。')
    return 0


if __name__ == '__main__':
    sys.exit(main())