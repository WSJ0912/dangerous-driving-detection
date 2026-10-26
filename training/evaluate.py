#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型评估 + **类别顺序一致性把关**

这个脚本最重要的职责不是算 mAP，而是回答一个问题：
**这个权重能不能安全地替换生产权重？**

生产后端用 ``backend/config.py::CLASS_NAMES`` 的**索引顺序**把模型输出翻译成
业务语义（``backend/services/detection.py``）。如果训练时 ``data.yaml`` 的
``names`` 顺序和后端不一致，模型照样能训、mAP 也可能很漂亮，但上线后每一类
都会张冠李戴 —— 而且不会有任何报错。``--check-backend`` 就是拦这个的。

用法::

    python training/evaluate.py --weights training/experiments/run/weights/best.pt \\
        --data training/data.yaml --check-backend --json training/experiments/run/metrics.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_DATA_YAML,
    DEFAULT_START_WEIGHTS,
    banner,
    check_class_consistency,
    normalize_names,
    resolve_device,
    resolve_weights,
    sha256_of,
)

# 业务上真正要紧的类别（疲劳/分心），单独分组展示
DANGER_CLASSES = ('close_eye', 'Yawn', 'smoke', 'drink', 'phone')
AUX_CLASSES = ('face', 'open_eye', 'noYawn')


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='评估权重并校验类别顺序是否可安全上线',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--weights', default='default', help="'default'=生产权重，或 .pt 路径")
    p.add_argument('--data', default=DEFAULT_DATA_YAML)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default=None)
    p.add_argument('--check-backend', action='store_true',
                   help='类别顺序与后端不一致时以退出码 3 失败')
    p.add_argument('--json', default=None, help='把结果写成 JSON（留档用）')
    p.add_argument('--skip-val', action='store_true', help='只做类别一致性检查，不跑验证集')
    return p.parse_args(argv)


def check_backend_or_die(model_names, strict):
    ok, detail = check_class_consistency(model_names)
    banner('类别顺序一致性（能否安全上线）')
    print('后端期望: %s' % CLASS_NAMES)
    print('权重声明: %s' % normalize_names(model_names))
    if ok:
        print('\n[OK] %s' % detail)
        return True
    print('\n[严重] %s' % detail)
    print('\n这意味着：若直接替换生产权重，检测框画得出来，但每个类别的中文名都会是错的。')
    if strict:
        print('已以退出码 3 失败（--check-backend）。')
        return False
    print('（未加 --check-backend，仅警告，继续评估。）')
    return True


def per_class_metrics(result, names):
    """从 ultralytics 的 val 结果里取逐类指标，尽量兼容版本差异"""
    box = getattr(result, 'box', None)
    rows = []
    if box is None:
        return rows

    idxs = None
    for attr in ('ap_class_index',):
        idxs = getattr(box, attr, None)
        if idxs is not None:
            break
    if idxs is None:
        idxs = list(range(len(names)))

    for i in idxs:
        i = int(i)
        name = names[i] if i < len(names) else 'class_%d' % i
        p = r = ap50 = ap = float('nan')
        try:
            p, r, ap50, ap = box.class_result(i)
        except Exception:
            for attr, slot in (('p', 0), ('r', 1), ('ap50', 2), ('ap', 3)):
                arr = getattr(box, attr, None)
                try:
                    val = float(arr[i])
                except Exception:
                    continue
                if slot == 0:
                    p = val
                elif slot == 1:
                    r = val
                elif slot == 2:
                    ap50 = val
                else:
                    ap = val
        rows.append({'index': i, 'class': name,
                     'precision': float(p), 'recall': float(r),
                     'mAP50': float(ap50), 'mAP50_95': float(ap)})
    return rows


def main(argv=None):
    args = parse_args(argv)

    weights = resolve_weights(args.weights)
    banner('评估目标')
    print('权重      : %s' % weights)
    if os.path.exists(weights):
        print('            %d B' % os.path.getsize(weights))
        print('            sha256=%s' % sha256_of(weights))
    else:
        print('[警告] 权重路径不存在，ultralytics 可能会尝试联网获取。')

    from ultralytics import YOLO

    model = YOLO(weights)
    consistent = check_backend_or_die(getattr(model, 'names', None), args.check_backend)
    if not consistent:
        return 3

    payload = {
        'weights': weights,
        'weights_sha256': sha256_of(weights) if os.path.exists(weights) else None,
        'weights_bytes': os.path.getsize(weights) if os.path.exists(weights) else None,
        'backend_class_names': CLASS_NAMES,
        'model_class_names': normalize_names(getattr(model, 'names', None)),
        'class_consistent': bool(consistent),
        'data_yaml': os.path.abspath(args.data),
    }

    if args.skip_val:
        print('\n--skip-val：跳过验证集评估。')
    else:
        if not os.path.exists(args.data):
            print('\n[错误] 数据 yaml 不存在: %s' % args.data)
            return 2
        device = resolve_device(args.device)
        banner('验证集评估')
        print('data=%s imgsz=%s batch=%s device=%s' % (args.data, args.imgsz, args.batch, device))
        result = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch,
                           device=device, plots=False, verbose=False)

        names = normalize_names(getattr(result, 'names', None)) or CLASS_NAMES
        rows = per_class_metrics(result, names)
        payload['per_class'] = rows

        box = getattr(result, 'box', None)
        for attr in ('map50', 'map', 'mp', 'mr'):
            try:
                payload['overall_' + attr] = float(getattr(box, attr))
            except Exception:
                pass

        if rows:
            print('\n逐类指标（Box）：')
            print('  %-4s %-12s %10s %10s %10s %12s' % ('idx', 'class', 'P', 'R', 'mAP50', 'mAP50-95'))
            for row in rows:
                print('  %-4d %-12s %10.4f %10.4f %10.4f %12.4f' % (
                    row['index'], row['class'], row['precision'], row['recall'],
                    row['mAP50'], row['mAP50_95']))

            def group(keys):
                sel = [r for r in rows if r['class'] in keys]
                if not sel:
                    return None
                return (sum(r['mAP50'] for r in sel) / len(sel),
                        sum(r['mAP50_95'] for r in sel) / len(sel))

            print('\n分组宏平均：')
            for title, keys in (('危险类（真正关心的）', DANGER_CLASSES),
                                ('辅助类', AUX_CLASSES)):
                agg = group(keys)
                if agg:
                    print('  %-18s mAP50=%.4f  mAP50-95=%.4f  （%d 类有指标）'
                          % (title, agg[0], agg[1], len([r for r in rows if r['class'] in keys])))
            print('\n[注意] mAP 高不等于业务可用：本项目真正依赖的是 close_eye / Yawn 等疲劳类，')
            print('       而这几类样本通常最少。请重点看上面"危险类"的数值，并核对样本量。')
        else:
            print('\n[警告] 未能从结果中取出逐类指标（ultralytics 版本差异）。')

    if args.json:
        out_dir = os.path.dirname(os.path.abspath(args.json))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print('\n已写出 JSON: %s' % args.json)

    banner('结论')
    if payload.get('class_consistent'):
        print('类别顺序与后端一致 —— 该权重可以安全替换生产权重（替换前请先备份第三方权重）。')
    return 0


if __name__ == '__main__':
    sys.exit(main())