#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练入口：在自采数据集上微调 YOLO11s，产出**属于你自己的**权重

用法示例
--------
冒烟（本机 CPU，只为验证链路能跑通，指标无意义）::

    python training/train.py --smoke

正式训练（租的 GPU 上，先按 training/README.md 准备好数据集）::

    python training/train.py --epochs 100 --imgsz 640 --batch 16 --device 0

中断续训::

    python training/train.py --resume training/experiments/<run>/weights/last.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_DATA_YAML,
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_START_WEIGHTS,
    NUM_CLASSES,
    PROJECT_ROOT,
    banner,
    check_class_consistency,
    dataset_counts,
    read_data_yaml,
    resolve_device,
    resolve_weights,
    sha256_of,
)

SMOKE_PRESET = dict(epochs=1, imgsz=320, batch=4, workers=0, plots=False)
# 仓库不含 *.pt（.gitignore 排除），冒烟时若默认权重缺失就退回到可自动下载的模型
SMOKE_FALLBACK_WEIGHTS = 'yolo11n.pt'


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='微调 YOLO11s 驾驶行为检测模型（8 类）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data', default=DEFAULT_DATA_YAML, help='ultralytics 数据 yaml')
    p.add_argument('--model', default='default',
                   help="起始权重：'default'=项目内 models/weights/best.pt；或本地/远程路径")
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default=None, help='0 / 0,1 / cpu；缺省自动选择')
    p.add_argument('--workers', type=int, default=0, help='Windows 上建议 0')
    p.add_argument('--project', default=DEFAULT_EXPERIMENTS_DIR)
    p.add_argument('--name', default='run')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', default=None, help='从 last.pt 续训')
    p.add_argument('--smoke', action='store_true',
                   help='CPU 快速冒烟预设（epochs=1, imgsz=320, batch=4）：只验证链路，不产出可用模型')
    p.add_argument('--allow-class-mismatch', action='store_true',
                   help='类别顺序与后端不一致时仍然继续（危险，仅供实验）')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.smoke:
        for k, v in SMOKE_PRESET.items():
            setattr(args, k, v)
        args.device = 'cpu'

    # --project 必须是绝对路径：传相对路径时 ultralytics 会把它拼到自己的
    # runs_dir 之下（<项目>/runs/detect/<project>/<name>），产物位置不可预期。
    args.project = os.path.abspath(args.project)
    args.data = os.path.abspath(args.data)

    banner('训练前检查')

    if not os.path.exists(args.data):
        print('[错误] 数据 yaml 不存在: %s' % args.data)
        print('       请先运行: python training/prepare_dataset.py --help')
        return 2

    data = read_data_yaml(args.data)
    counts, _ = dataset_counts(args.data)

    print('数据 yaml : %s' % args.data)
    print('数据集根  : %s' % data['_root'])
    print('划分图片数: %s' % (counts or '无法解析'))
    print('后端类别  : %d 类 %s' % (NUM_CLASSES, CLASS_NAMES))

    # ---- 类别顺序一致性（最关键的一道闸门）----
    ok, detail = check_class_consistency(data.get('names'))
    if ok:
        print('[OK] %s' % detail)
    else:
        print('\n[严重警告] data.yaml 的类别与后端不一致！')
        print(detail)
        print('\n后果：模型能正常训练、指标也可能很好看，但上线后每个类别的中文名都会张冠李戴，')
        print('      而且不会有任何报错。请把 data.yaml 的 names 改成与 backend/config.py 的')
        print('      CLASS_NAMES 完全相同的顺序，或修正你的标注类别映射。')
        if not args.allow_class_mismatch:
            print('\n已中止训练。确实要强行继续请加 --allow-class-mismatch。')
            return 2
        print('\n[!] 你显式加了 --allow-class-mismatch，继续训练（结果大概率不可用）。')

    for split in ('train', 'val'):
        if counts.get(split, 0) == 0:
            print('\n[错误] %s 划分里没有图片（%s）。' % (
                split, data['_splits'].get(split)))
            print('       请先用 training/prepare_dataset.py 准备并校验数据集。')
            return 2

    # ---- 冒烟模式声明 ----
    if args.smoke:
        banner('冒烟模式（--smoke）', char='!')
        print('本次运行仅用于验证【训练链路能跑通】：epochs=1、imgsz=320、batch=4、device=cpu。')
        print('产出的权重不可用于生产，指标数值没有任何意义。正式训练请在 GPU 上跑完整参数。')

    # 全新克隆里没有 models/weights/best.pt（权重不入库）。冒烟模式的目的是验证链路，
    # 不该因此失败 —— 退回到可自动下载的模型；正式训练仍保持明确报错，不静默换权重。
    if (args.smoke and not args.resume
            and str(args.model).strip().lower() in ('default', 'best')
            and not os.path.exists(DEFAULT_START_WEIGHTS)):
        print('\n[提示] 仓库内没有 models/weights/best.pt（权重不入库），')
        print('       冒烟模式改用可自动下载的 %s（首次运行需联网）。'
              % SMOKE_FALLBACK_WEIGHTS)
        args.model = SMOKE_FALLBACK_WEIGHTS

    weights = args.resume or resolve_weights(args.model)
    device = resolve_device(args.device)

    banner('开始训练')
    print('起始权重 : %s' % weights)
    if os.path.exists(weights):
        print('           %d B, sha256=%s' % (os.path.getsize(weights), sha256_of(weights)[:16]))
    print('设备     : %s%s' % (device, '' if device != 'cpu' else '（无 GPU，速度会很慢）'))
    print('epochs=%s imgsz=%s batch=%s workers=%s seed=%s' % (
        args.epochs, args.imgsz, args.batch, args.workers, args.seed))
    print('输出目录 : %s/%s' % (args.project, args.name))

    from ultralytics import YOLO

    model = YOLO(weights)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
        cache=False,
        amp=(device != 'cpu'),
    )
    if args.resume:
        train_kwargs['resume'] = True

    try:
        results = model.train(**train_kwargs)
    except Exception as exc:  # 离线环境绘图/字体问题最常见
        if train_kwargs.get('plots'):
            print('\n[提示] 训练/绘图失败（%s），回退为 plots=False 重试...' % exc)
            train_kwargs['plots'] = False
            results = model.train(**train_kwargs)
        else:
            raise

    save_dir = str(getattr(results, 'save_dir', os.path.join(args.project, args.name)))
    banner('训练完成')
    print('输出目录 : %s' % save_dir)

    wdir = os.path.join(save_dir, 'weights')
    for fname in ('best.pt', 'last.pt'):
        fpath = os.path.join(wdir, fname)
        if os.path.exists(fpath):
            print('  %-10s %d B  sha256=%s' % (fname, os.path.getsize(fpath), sha256_of(fpath)[:16]))

    csv_path = os.path.join(save_dir, 'results.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            last = None
            for line in f:
                if line.strip():
                    last = line.strip()
        if last:
            print('\nresults.csv 首行: %s' % header)
            print('results.csv 末行: %s' % last)

    best = os.path.join(wdir, 'best.pt')
    if os.path.exists(best):
        from ultralytics import YOLO as _YOLO
        names = getattr(_YOLO(best), 'names', None)
        cok, cdetail = check_class_consistency(names)
        print('\n新权重类别一致性: %s' % ('OK' if cok else '不一致'))
        if not cok:
            print(cdetail)

    banner('下一步')
    print('1) 评估（含类别顺序把关）：')
    print('     python training/evaluate.py --weights %s --data %s --check-backend' % (best, args.data))
    print('2) 训练报告留档：')
    print('     python training/evaluate.py --weights %s --data %s --check-backend --json %s'
          % (best, args.data, os.path.join(save_dir, 'metrics.json')))
    print('3) 确认无误后替换生产权重（先备份原权重！）：')
    print('     备份 models/weights/best.pt -> models/weights/best.prev.bak.pt')
    print('     再把你训练出的 best.pt 复制到 models/weights/best.pt')
    print('4) 启动服务，用真实图像跑一次端到端确认。')
    return 0


if __name__ == '__main__':
    sys.exit(main())