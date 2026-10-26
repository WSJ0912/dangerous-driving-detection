#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据集准备与校验

把"零数据集"变成"可以开始训练"，并在训练前把标注问题挡下来。

常用用法
--------
1) 生成冒烟用的**假数据**（只为验证训练链路能跑通，不可用于训练可用模型）::

       python training/prepare_dataset.py --synthetic --write-yaml

2) 转换并校验自己的 YOLO 格式数据集::

       python training/prepare_dataset.py --source D:\\my_dataset --format yolo \\
           --out data/yolo_format --val-ratio 0.2 --write-yaml

3) 只校验、不复制（先看标注有没有问题）::

       python training/prepare_dataset.py --source D:\\my_dataset --format yolo --dry-run

退出码：0 正常；2 数据集/参数问题；3 标注校验发现错误。
"""
import argparse
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_DATASET_DIR,
    DEFAULT_DATA_YAML,
    IMAGE_EXTS,
    NUM_CLASSES,
    PROJECT_ROOT,
    banner,
    write_data_yaml,
    imwrite_safe,
)

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------- 校验

def validate_label_file(path, clip=False):
    """校验单个 YOLO 标签文件

    Returns:
        (instances: list[(cls, cx, cy, w, h)], errors: list[str], clipped: int)
    """
    instances, errors, clipped = [], [], 0
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    except Exception as exc:
        return [], ['无法读取标签文件: %s' % exc], 0

    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append('%s:%d 字段数应为 5，实际 %d: %r' % (path, lineno, len(parts), line[:60]))
            continue
        try:
            cls_f = float(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append('%s:%d 存在非数字字段: %r' % (path, lineno, line[:60]))
            continue

        if cls_f != int(cls_f):
            errors.append('%s:%d 类别号不是整数: %s' % (path, lineno, parts[0]))
            continue
        cls = int(cls_f)
        if not (0 <= cls < NUM_CLASSES):
            errors.append('%s:%d 类别号越界 %d（应为 0~%d）' % (path, lineno, cls, NUM_CLASSES - 1))
            continue

        if w <= 0 or h <= 0:
            errors.append('%s:%d 宽或高不是正数: w=%s h=%s' % (path, lineno, w, h))
            continue

        out_of_range = not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1)
        if out_of_range:
            if not clip:
                errors.append('%s:%d 归一化坐标越界: cx=%.4f cy=%.4f w=%.4f h=%.4f'
                              % (path, lineno, cx, cy, w, h))
                continue
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            clipped += 1

        instances.append((cls, cx, cy, w, h))

    return instances, errors, clipped


def collect_pairs(source, fmt):
    """收集 (image_path, label_path) 对，并统计孤立文件"""
    image_map, label_map = {}, {}

    for root, _dirs, files in os.walk(source):
        for name in files:
            full = os.path.join(root, name)
            stem, ext = os.path.splitext(name)
            if ext.lower() in IMAGE_EXTS:
                image_map[stem] = full
            elif ext.lower() == '.txt':
                label_map[stem] = full

    pairs, orphan_images, orphan_labels = [], [], []
    for stem, img in sorted(image_map.items()):
        if stem in label_map:
            pairs.append((img, label_map[stem]))
        else:
            orphan_images.append(img)
    for stem, lab in sorted(label_map.items()):
        if stem not in image_map:
            orphan_labels.append(lab)

    return pairs, orphan_images, orphan_labels


def copy_pair(image_path, label_path, out_split_dir):
    img_dir = os.path.join(out_split_dir, 'images')
    lab_dir = os.path.join(out_split_dir, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    name = os.path.basename(image_path)
    shutil.copy2(image_path, os.path.join(img_dir, name))
    shutil.copy2(label_path, os.path.join(lab_dir, os.path.splitext(name)[0] + '.txt'))


# ---------------------------------------------------------------- 合成数据

def make_synthetic(out_dir, n_train=16, n_val=4, size=320, seed=0, write_yaml=False):
    import numpy as np

    rnd = random.Random(seed)
    np.random.seed(seed)

    banner('生成合成数据集（假数据）', char='!')
    print('用途：仅验证训练链路能跑通。')
    print('警告：这是随机彩色矩形，不是真实驾驶图像，训练出的模型没有任何实用价值。')

    for split, count in (('train', n_train), ('val', n_val)):
        for i in range(count):
            img = np.full((size, size, 3), rnd.randint(40, 90), np.uint8)
            lines = []
            for _ in range(rnd.randint(2, 4)):
                w = rnd.randint(40, 120)
                h = rnd.randint(40, 120)
                x1 = rnd.randint(0, size - w - 1)
                y1 = rnd.randint(0, size - h - 1)
                color = tuple(rnd.randint(120, 255) for _ in range(3))
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, -1)
                # 前 8 个目标依次覆盖 8 个类别，保证每类都有实例
                cls = (len(lines) + i) % NUM_CLASSES
                lines.append('%d %.6f %.6f %.6f %.6f' % (
                    cls, (x1 + w / 2) / size, (y1 + h / 2) / size, w / size, h / size))
            name = '%s_%03d' % (split, i)
            img_path = os.path.join(out_dir, split, 'images', name + '.jpg')
            lab_path = os.path.join(out_dir, split, 'labels', name + '.txt')
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            os.makedirs(os.path.dirname(lab_path), exist_ok=True)
            imwrite_safe(img_path, img)
            with open(lab_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')

    # 回读磁盘核对：cv2.imwrite 曾在中文路径下静默失败（标签写了、图片没写），
    # 结果直到训练阶段才以"划分里没有图片"暴露出来。这里直接把它拦在生成阶段。
    problems = []
    for split, want in (('train', n_train), ('val', n_val)):
        d = os.path.join(out_dir, split, 'images')
        got = len([f for f in os.listdir(d)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.isdir(d) else 0
        if got != want:
            problems.append('%s 期望 %d 张、实际 %d 张' % (split, want, got))
    if problems:
        print('\n[错误] 合成数据写盘不完整：%s' % '；'.join(problems))
        print('       最常见原因是输出路径含非 ASCII 字符（例如中文目录）。')
        print('       已改用 imwrite_safe 仍失败时，请把 --out 指到纯英文路径再试。')
        return 3
    print('已生成: %s  (train=%d, val=%d) —— 已回读磁盘核对' % (out_dir, n_train, n_val))
    if write_yaml:
        path = write_data_yaml(DEFAULT_DATA_YAML, out_dir)
        print('已生成数据 yaml: %s' % path)
    return 0


# ---------------------------------------------------------------- 主流程

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='准备并校验 YOLO 格式的驾驶行为数据集（8 类）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--source', default=None, help='原始数据集目录')
    p.add_argument('--format', default='yolo', choices=('yolo', 'raw'),
                   help='yolo=源目录已按 YOLO 组织；raw=图片与同名 txt 混放')
    p.add_argument('--out', default=DEFAULT_DATASET_DIR, help='输出目录（data/yolo_format）')
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--synthetic', action='store_true', help='生成冒烟用的合成数据')
    p.add_argument('--write-yaml', action='store_true', help='生成/刷新 training/data.yaml')
    p.add_argument('--dry-run', action='store_true', help='只校验，不复制')
    p.add_argument('--clip', action='store_true', help='自动裁剪越界的归一化坐标而不是报错')
    p.add_argument('--allow-background', action='store_true',
                   help='允许没有标注的图片（纯背景图）')
    p.add_argument('--max-list', type=int, default=10, help='问题清单最多打印多少条')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.synthetic:
        return make_synthetic(args.out, seed=args.seed, write_yaml=args.write_yaml)

    if not args.source:
        print('[错误] 需要 --source，或使用 --synthetic 生成冒烟数据。')
        return 2
    if not os.path.isdir(args.source):
        print('[错误] --source 不是目录: %s' % args.source)
        return 2
    if cv2 is None:
        print('[警告] 未安装 opencv，将跳过"坏图"检测。')

    banner('数据集校验')
    print('源目录   : %s' % args.source)
    print('格式     : %s' % args.format)
    print('后端类别 : %d 类 %s' % (NUM_CLASSES, CLASS_NAMES))

    pairs, orphan_images, orphan_labels = collect_pairs(args.source, args.format)
    print('\n发现图片 %d 张，其中成对标注 %d 组' % (len(pairs) + len(orphan_images), len(pairs)))

    all_errors, class_counts, bad_images = [], {c: 0 for c in range(NUM_CLASSES)}, []
    clipped_total = 0

    for img_path, lab_path in pairs:
        instances, errors, clipped = validate_label_file(lab_path, clip=args.clip)
        all_errors.extend(errors)
        clipped_total += clipped
        for cls, *_ in instances:
            class_counts[cls] += 1
        if cv2 is not None:
            img = cv2.imdecode(__import__('numpy').fromfile(img_path, dtype='uint8'), cv2.IMREAD_COLOR)
            if img is None:
                bad_images.append(img_path)

    # ---- 报告 ----
    print('\n逐类实例数：')
    for i, name in enumerate(CLASS_NAMES):
        n = class_counts[i]
        flag = '   <-- 该类别没有任何实例！' if n == 0 else ''
        print('  %d  %-10s %6d%s' % (i, name, n, flag))

    empty_classes = [CLASS_NAMES[i] for i, n in class_counts.items() if n == 0]
    if empty_classes:
        print('\n[警告] 以下类别样本为 0，模型无法学会它们：%s' % ', '.join(empty_classes))

    problem = False
    if all_errors:
        problem = True
        print('\n[错误] 标注问题 %d 条，前 %d 条：' % (len(all_errors), args.max_list))
        for e in all_errors[:args.max_list]:
            print('   ' + e)
    if bad_images:
        problem = True
        print('\n[错误] 无法解码的图片 %d 张：' % len(bad_images))
        for b in bad_images[:args.max_list]:
            print('   ' + b)
    if orphan_labels:
        problem = True
        print('\n[错误] 有标签但没有对应图片 %d 个：' % len(orphan_labels))
        for o in orphan_labels[:args.max_list]:
            print('   ' + o)
    if orphan_images and not args.allow_background:
        problem = True
        print('\n[错误] 有图片但没有对应标签 %d 张（纯背景图请加 --allow-background）：' % len(orphan_images))
        for o in orphan_images[:args.max_list]:
            print('   ' + o)
    if clipped_total:
        print('\n[提示] 已自动裁剪越界坐标 %d 处（--clip）' % clipped_total)

    if problem:
        print('\n校验未通过，已中止（退出码 3）。修好标注后重跑。')
        return 3

    print('\n[OK] 标注校验通过。')

    if args.dry_run:
        print('--dry-run：未复制任何文件。')
        return 0

    # ---- 复制/切分 ----
    src_train = os.path.join(args.source, 'train')
    src_val = os.path.join(args.source, 'val')
    preserve_split = os.path.isdir(src_train) and os.path.isdir(src_val)

    if preserve_split:
        print('\n源目录已有 train/val 划分，保留原划分。')
        for split, sub in (('train', args.source), ('val', os.path.join(args.source, 'val'))):
            if split == 'train':
                pool, oi, ol = collect_pairs(os.path.join(args.source, 'train'), args.format)
            else:
                pool, oi, ol = collect_pairs(os.path.join(args.source, 'val'), args.format)
            for img_p, lab_p in pool:
                copy_pair(img_p, lab_p, os.path.join(args.out, split))
            print('  %-5s 复制 %d 组' % (split, len(pool)))
    else:
        rnd = random.Random(args.seed)
        shuffled = list(pairs)
        rnd.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * args.val_ratio))) if shuffled else 0
        split_map = {'val': shuffled[:n_val], 'train': shuffled[n_val:]}
        print('\n按 val-ratio=%.2f 切分（seed=%d）。' % (args.val_ratio, args.seed))
        for split, pool in split_map.items():
            for img_p, lab_p in pool:
                copy_pair(img_p, lab_p, os.path.join(args.out, split))
            print('  %-5s 复制 %d 组' % (split, len(pool)))

    banner('完成')
    print('数据集输出: %s' % args.out)
    if args.write_yaml:
        path = write_data_yaml(DEFAULT_DATA_YAML, args.out)
        print('数据 yaml : %s' % path)
    print('\n下一步: python training/train.py --smoke        # 先验证链路')
    print('        python training/train.py --epochs 100 --imgsz 640 --batch 16 --device 0')
    return 0


if __name__ == '__main__':
    sys.exit(main())