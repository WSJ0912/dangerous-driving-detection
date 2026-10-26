#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 Roboflow Universe 获取公开检测数据集，重映射类别后合并为本项目可训练的数据集

背景
----
本项目仓库内没有任何数据集。本脚本把两个**公开的、带检测框的**驾驶行为数据集
取下来，按下面的映射把它们的类别翻译成本项目后端的 8 类，合并到
``data/yolo_format/{train,val}`` 并生成 ``training/data.yaml``。

类别映射（**这是本脚本的核心，也是最需要人工复核的地方**）
--------------------------------------------------------
| 本项目类别    | 来源类别                          | 来源数据集                    |
|--------------|----------------------------------|------------------------------|
| face         | face                             | driver-drowsiness            |
| open_eye     | open_eye                         | driver-drowsiness            |
| close_eye    | closed_eye                       | driver-drowsiness            |
| Yawn         | open_mouth                       | driver-drowsiness（**近似**） |
| noYawn       | closed_mouth                     | driver-drowsiness（**近似**） |
| phone        | Phone                            | reckless-driver-detection    |
| smoke        | Cigarette                        | reckless-driver-detection    |
| drink        | Drinking                         | reckless-driver-detection    |

.. warning::
   ``open_mouth -> Yawn`` 与 ``closed_mouth -> noYawn`` 是**近似映射**：
   张嘴不等于打哈欠。这会让 Yawn/noYawn 的指标偏乐观或偏悲观，报告中必须如实说明，
   不能把训练出的 mAP 直接当作"打哈欠识别精度"。

密钥
----
**不会打印、不会写入日志。** 按以下顺序读取：

1. 环境变量 ``ROBOFLOW_API_KEY``
2. 文件 ``%USERPROFILE%\\.roboflow_key``（内容为单行密钥）

用法
----
::

    python training/fetch_dataset.py --list          # 只列出数据集信息（版本/类别/许可）
    python training/fetch_dataset.py --download      # 下载 -> 重映射 -> 合并 -> 写 data.yaml
"""
import argparse
import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_DATA_YAML,
    DEFAULT_DATASET_DIR,
    NUM_CLASSES,
    banner,
    write_data_yaml,
)

try:
    import requests
except ImportError:
    print("需要 requests：python -m pip install requests")
    raise

API = "https://api.roboflow.com"

# 数据来源与类别映射。version=None 表示取最新版本。
SOURCES = [
    {
        "key": "driver-drowsiness",
        "workspace": "computer-vision-tznlp",
        "project": "driver-drowsiness-q3msq-ddgj8",
        "version": None,
        "citation": "Driver Drowsiness Dataset, Roboflow Universe",
        "map": {
            "face": "face",
            "open_eye": "open_eye",
            "closed_eye": "close_eye",
            "open_mouth": "Yawn",       # 近似
            "closed_mouth": "noYawn",   # 近似
        },
    },
    {
        "key": "reckless-driver-detection",
        "workspace": "ztai",
        "project": "reckless-driver-detection-iz7ob-apeue",
        "version": None,
        "citation": "Reckless driver detection Dataset, Roboflow Universe",
        "map": {
            "Phone": "phone",
            "Cigarette": "smoke",
            "Drinking": "drink",
        },
    },
]


def read_api_key():
    """读取密钥但绝不打印。"""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key and key.strip():
        return key.strip()
    path = os.path.join(os.path.expanduser("~"), ".roboflow_key")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key
    raise SystemExit(
        "未找到 Roboflow 密钥。请任选一种方式：\n"
        "  A) 设置环境变量 ROBOFLOW_API_KEY\n"
        "  B) 把密钥单行写入 %s\n"
        "（密钥不会被本脚本打印或写进任何文件）" % path)


def api_get(path, key, **params):
    params["api_key"] = key
    r = requests.get(API + path, params=params, timeout=60)
    if r.status_code == 401:
        raise SystemExit("Roboflow 拒绝了这个密钥（401）。请确认密钥有效。")
    r.raise_for_status()
    return r.json()


def project_info(src, key):
    """取项目元信息（含版本列表与类别）"""
    data = api_get("/%s/%s" % (src["workspace"], src["project"]), key)
    return data


def pick_version(info, want=None):
    versions = info.get("versions") or info.get("project", {}).get("versions") or []
    if not versions:
        return None, []
    if want is not None:
        for v in versions:
            if int(v.get("id", v.get("version", -1))) == int(want):
                return int(want), versions
    newest = max(versions, key=lambda v: int(v.get("id", v.get("version", 0))))
    return int(newest.get("id", newest.get("version"))), versions


def do_list(key):
    banner("数据集信息（不下载）")
    for src in SOURCES:
        print("\n--- %s ---" % src["key"])
        try:
            info = project_info(src, key)
        except SystemExit:
            raise
        except Exception as exc:
            print("  查询失败: %s" % exc)
            continue
        proj = info.get("project", info)
        print("  名称    : %s" % proj.get("name"))
        print("  类型    : %s" % proj.get("type"))
        print("  许可    : %s" % proj.get("license"))
        print("  图片数  : %s" % proj.get("images"))
        classes = proj.get("classes") or {}
        if isinstance(classes, dict):
            print("  类别    : %s" % ", ".join("%s(%s)" % (k, v) for k, v in classes.items()))
        else:
            print("  类别    : %s" % classes)
        vid, versions = pick_version(info, src["version"])
        print("  版本数  : %d ；将使用版本 %s" % (len(versions), vid))
        print("  映射到本项目:")
        for s, d in src["map"].items():
            print("      %-14s -> %s" % (s, d))
        unmapped = []
        if isinstance(classes, dict):
            for cname in classes.keys():
                if cname not in src["map"]:
                    unmapped.append(cname)
        if unmapped:
            print("  [注意] 未映射的类别（将被忽略）: %s" % ", ".join(unmapped))
    print("\n提示：正式下载前请核对上面每个数据集的『许可』字段，确认允许你的使用方式。")


def download_version(src, version, key, dest_dir):
    """下载某一版本的 YOLO 格式数据集，返回解压目录"""
    url_path = "/%s/%s/%s/yolov8" % (src["workspace"], src["project"], version)
    info = api_get(url_path, key)
    link = info.get("link")
    if not link:
        raise RuntimeError("接口未返回下载链接: %s" % json.dumps(info)[:200])

    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "%s-v%s.zip" % (src["key"], version))
    print("  下载中（链接由 Roboflow 返回，体积可能较大）...")
    with requests.get(link, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    size = os.path.getsize(zip_path)
    print("  已下载: %s (%.1f MB)" % (zip_path, size / 1048576.0))

    extract_dir = os.path.join(dest_dir, "%s-v%s" % (src["key"], version))
    if os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    print("  已解压: %s" % extract_dir)
    return extract_dir


def find_yaml(root):
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f in ("data.yaml", "dataset.yaml"):
                return os.path.join(dirpath, f)
    return None


def load_names(yaml_path):
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return list(names or [])


def remap_split(src_root, src_names, class_map, out_split):
    """把某个 split 的图片与标签复制到目标 split，并重映射类别索引"""
    img_dir = os.path.join(out_split, "images")
    lab_dir = os.path.join(out_split, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    # 源类别名 -> 本项目索引
    idx_map = {}
    for i, name in enumerate(src_names):
        target = class_map.get(name)
        if target is not None:
            idx_map[i] = CLASS_NAMES.index(target)

    copied = skipped = relabeled = 0
    for dirpath, _dirs, files in os.walk(src_root):
        if os.path.basename(dirpath) != "images":
            continue
        lab_root = os.path.join(os.path.dirname(dirpath), "labels")
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue
            lab = os.path.join(lab_root, stem + ".txt")
            if not os.path.exists(lab):
                skipped += 1
                continue
            kept = []
            with open(lab, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    try:
                        c = int(float(parts[0]))
                    except ValueError:
                        continue
                    if c not in idx_map:
                        continue
                    kept.append("%d %s %s %s %s" % (idx_map[c], parts[1], parts[2], parts[3], parts[4]))
            if not kept:
                skipped += 1
                continue
            relabeled += len(kept)
            shutil.copy2(os.path.join(dirpath, name), os.path.join(img_dir, name))
            with open(os.path.join(lab_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(kept) + "\n")
            copied += 1
    return copied, skipped, relabeled


def do_download(key, work_dir):
    banner("下载并合并数据集")
    print("目标类别（后端权威定义）: %s" % CLASS_NAMES)

    totals = {}
    for src in SOURCES:
        print("\n=== %s ===" % src["key"])
        info = project_info(src, key)
        proj = info.get("project", info)
        print("  许可: %s | 图片: %s" % (proj.get("license"), proj.get("images")))

        version, _ = pick_version(info, src["version"])
        if version is None:
            print("  [跳过] 未能确定版本号")
            continue
        root = download_version(src, version, key, work_dir)

        yaml_path = find_yaml(root)
        if not yaml_path:
            print("  [跳过] 解压目录里找不到 data.yaml")
            continue
        src_names = load_names(yaml_path)
        print("  源类别: %s" % src_names)

        missing = [s for s in src["map"] if s not in src_names]
        if missing:
            print("  [警告] 以下期望的源类别不存在，对应本项目类别将没有样本: %s" % missing)

        ds_root = os.path.dirname(yaml_path)
        for split, out_split in (("train", "train"), ("valid", "val"), ("val", "val")):
            src_split = os.path.join(ds_root, split)
            if not os.path.isdir(src_split):
                continue
            c, s, r = remap_split(src_split, src_names, src["map"],
                                  os.path.join(DEFAULT_DATASET_DIR, out_split))
            print("  %-14s -> %-4s 复制 %d 张，跳过 %d 张，重映射标签 %d 条" % (split, out_split, c, s, r))
            totals[out_split] = totals.get(out_split, 0) + c
            break  # 一个数据集只用一个验证划分名

    banner("合并结果")
    for split in ("train", "val"):
        d = os.path.join(DEFAULT_DATASET_DIR, split, "images")
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print("  %-6s %d 张" % (split, n))

    yaml_path = write_data_yaml(DEFAULT_DATA_YAML, DEFAULT_DATASET_DIR)
    print("  数据 yaml: %s" % yaml_path)

    banner("下一步：先校验再训练")
    print("1) python training/prepare_dataset.py --source %s --format yolo --dry-run" % DEFAULT_DATASET_DIR)
    print("2) python training/train.py --smoke              # 先确认管道通")
    print("3) python training/train.py --epochs 100 --imgsz 640 --batch 16 --device 0")
    print("\n提醒：open_mouth->Yawn / closed_mouth->noYawn 是近似映射，报告中必须如实说明。")


def main(argv=None):
    p = argparse.ArgumentParser(description="从 Roboflow 拉取公开数据集并重映射为本项目 8 类")
    p.add_argument("--list", action="store_true", help="只列出数据集信息")
    p.add_argument("--download", action="store_true", help="下载并合并")
    p.add_argument("--work-dir", default=os.path.join(DEFAULT_DATASET_DIR, "_raw_downloads"),
                   help="下载与解压的临时目录")
    args = p.parse_args(argv)

    if not args.list and not args.download:
        p.print_help()
        return 0

    key = read_api_key()
    print("已读取 Roboflow 密钥（%d 字符，内容不显示）" % len(key))

    if args.list:
        do_list(key)
    if args.download:
        do_download(key, args.work_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())