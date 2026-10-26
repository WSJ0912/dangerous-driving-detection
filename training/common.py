# -*- coding: utf-8 -*-
"""训练链路公共模块

唯一权威的类别定义来自 ``backend/config.py::CLASS_NAMES`` —— 训练侧的
``data.yaml``、评估侧的一致性校验、以及生产后端的类别翻译**必须共用同一份**。
任何一处复制字面量都会在将来漂移，而漂移的后果（类别张冠李戴）不会报错。
"""
import hashlib
import os
import sys

import yaml

# 关闭 ultralytics 的隐式依赖自动安装（默认会在导出等功能缺依赖时直接 pip install）。
# 必须在 import ultralytics 之前设置，因此放在公共模块里统一生效。
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend import config as backend_config  # noqa: E402

# 类别定义：直接取自后端，不复制字面量
CLASS_NAMES = list(backend_config.CLASS_NAMES)
NUM_CLASSES = len(CLASS_NAMES)

TRAINING_DIR = os.path.join(PROJECT_ROOT, 'training')
DEFAULT_DATA_YAML = os.path.join(TRAINING_DIR, 'data.yaml')
DEFAULT_EXPERIMENTS_DIR = os.path.join(TRAINING_DIR, 'experiments')
DEFAULT_START_WEIGHTS = os.path.join(PROJECT_ROOT, 'models', 'weights', 'best.pt')
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'yolo_format')

# 说明：生产权重是第三方产物（详见 models/README.md）。
# 想产出"自己的"权重，就以它为起点做微调，或改用自采数据集从 yolo11s 预训练权重起步。


def banner(text, char='='):
    line = char * 72
    print('\n' + line)
    print(text)
    print(line)


def class_names():
    """返回权威类别列表（副本，调用方改不动内部状态）"""
    return list(CLASS_NAMES)


def normalize_names(model_names):
    """把 {'0': 'Yawn', ...} / {0: 'Yawn', ...} / ['Yawn', ...] 统一成有序 list"""
    if model_names is None:
        return []
    if isinstance(model_names, dict):
        try:
            keys = sorted(model_names.keys(), key=lambda k: int(k))
        except (TypeError, ValueError):
            keys = list(model_names.keys())
        return [model_names[k] for k in keys]
    return list(model_names)


def check_class_consistency(model_names):
    """校验模型/数据集声明的类别与后端 ``CLASS_NAMES`` **按顺序**一致

    Returns:
        (ok: bool, detail: str)
    """
    names = normalize_names(model_names)
    if not names:
        return False, '未取得类别名（model_names 为空）'

    if names == CLASS_NAMES:
        return True, '一致：%d 类，顺序与 backend.config.CLASS_NAMES 完全相同' % len(names)

    lines = ['类别不一致 —— 会导致检测结果张冠李戴，且不会报错。',
             '  索引  权重/数据集里的名字        后端期望的名字']
    for i in range(max(len(names), len(CLASS_NAMES))):
        got = names[i] if i < len(names) else '<缺失>'
        want = CLASS_NAMES[i] if i < len(CLASS_NAMES) else '<多余>'
        mark = '  ' if got == want else ' <-- 不一致'
        lines.append('  %-5d %-24s %s%s' % (i, got, want, mark))
    return False, '\n'.join(lines)


def resolve_weights(spec):
    """把 --model 的取值解析成 ultralytics 能接受的权重路径

    - None / '' / 'default'  -> 项目内 models/weights/best.pt（本地文件，离线可用）
    - 存在的本地路径          -> 原样返回
    - 其他（如 'yolo11s.pt'） -> 原样返回（ultralytics 会尝试联网下载）
    """
    if not spec or str(spec).strip().lower() in ('default', 'best'):
        if not os.path.exists(DEFAULT_START_WEIGHTS):
            raise FileNotFoundError(
                '默认起始权重不存在: %s\n'
                '请先放置 best.pt，或用 --model 指定其它权重/配置。' % DEFAULT_START_WEIGHTS)
        return DEFAULT_START_WEIGHTS

    spec = str(spec)
    looks_like_path = (os.sep in spec or '/' in spec or spec.endswith(('.pt', '.yaml')))
    if looks_like_path:
        candidate = spec if os.path.isabs(spec) else os.path.join(PROJECT_ROOT, spec)
        if os.path.exists(candidate):
            return candidate
        # .yaml 结构文件可能由 ultralytics 自己解析，交给它；.pt 不存在则报错
        if spec.endswith('.pt'):
            raise FileNotFoundError('指定的权重不存在: %s' % candidate)
    return spec


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def read_data_yaml(path):
    """读取 ultralytics 数据 yaml，并解析出各 split 的绝对目录"""
    if not os.path.exists(path):
        raise FileNotFoundError('数据 yaml 不存在: %s' % path)
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    root = data.get('path') or os.path.dirname(os.path.abspath(path))
    root = os.path.abspath(root)

    splits = {}
    for key in ('train', 'val', 'test'):
        value = data.get(key)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            splits[key] = [os.path.normpath(p if os.path.isabs(p) else os.path.join(root, p))
                           for p in value]
        else:
            splits[key] = [os.path.normpath(value if os.path.isabs(value)
                                            else os.path.join(root, value))]
    data['_root'] = root
    data['_splits'] = splits
    return data


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def count_images(dirs):
    total = 0
    for d in dirs or []:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.lower().endswith(IMAGE_EXTS):
                total += 1
    return total


def dataset_counts(data_yaml_path):
    """返回 {'train': n, 'val': n, ...}"""
    data = read_data_yaml(data_yaml_path)
    return {k: count_images(v) for k, v in data['_splits'].items()}, data


def write_data_yaml(path, dataset_root, class_names_list=None, splits=('train', 'val')):
    """生成 ultralytics 数据 yaml，并回读校验类别顺序

    ``path`` 显式写成绝对路径：本机 ultralytics 的 ``datasets_dir`` 里有第三方
    残留路径，依赖默认值会解析到别处。
    """
    names = list(class_names_list) if class_names_list else class_names()
    dataset_root = os.path.abspath(dataset_root)

    payload = {
        'path': dataset_root,
        'nc': len(names),
        'names': {i: n for i, n in enumerate(names)},
    }
    for split in splits:
        payload[split] = '%s/images' % split

    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    # 直接写入，不用 tmp + os.replace：本项目所在盘对当前用户没有 Delete 权限，
    # 而 os.replace 覆盖已存在文件时要求删除目标，会让第二次运行直接失败。
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # 回读校验：确保落盘的类别顺序就是我们要的
    with open(path, 'r', encoding='utf-8') as f:
        back = yaml.safe_load(f)
    ok, detail = check_class_consistency(back.get('names'))
    if not ok:
        raise RuntimeError('写入后的 data.yaml 类别校验失败:\n%s' % detail)
    if int(back.get('nc', -1)) != len(names):
        raise RuntimeError('写入后的 nc 与 names 数量不等: %s vs %s' % (back.get('nc'), len(names)))

    return path


def resolve_device(requested=None):
    """无 GPU 时自动回落 cpu"""
    if requested:
        want = str(requested)
        if want.lower() == 'cpu':
            return 'cpu'
        try:
            import torch
            if not torch.cuda.is_available():
                print('[提示] 请求了 device=%s，但本机 CUDA 不可用，已回落 cpu' % want)
                return 'cpu'
        except Exception:
            return 'cpu'
        return want
    try:
        import torch
        return '0' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def imwrite_safe(path, img):
    """把图像写到可能含非 ASCII 字符的路径上（并保证真的写成功）

    为什么不用 ``cv2.imwrite``：在 Windows 上遇到中文路径时它**返回 False 而不抛异常**，
    于是"图没写出来"这件事会一路静默到训练阶段才以"划分里没有图片"的面目出现。
    本项目路径本身就含中文，这个坑已经实际踩到过一次。

    改用 ``cv2.imencode`` + ``ndarray.tofile``（按字节写，绕开 Windows 的 ANSI 路径接口），
    写完再回读校验文件存在且非空，失败就抛错而不是返回假成功。
    """
    import cv2
    ext = os.path.splitext(path)[1] or '.jpg'
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError('图像编码失败（扩展名 %s）: %s' % (ext, path))
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    buf.tofile(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise IOError('图像写入失败（路径可能含非 ASCII 字符）: %s' % path)
    return True
