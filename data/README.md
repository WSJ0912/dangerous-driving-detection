# 数据集目录结构

## 目录说明

```
data/
├── raw/                          # 原始图片
│   ├── train/                    # 训练集原始图片
│   ├── val/                      # 验证集原始图片
│   └── test/                     # 测试集原始图片
│
├── yolo_format/                  # YOLO 标注格式
│   ├── train/
│   │   ├── images/               # 训练集图片（.jpg）
│   │   └── labels/               # 训练集标注（.txt，YOLO 格式）
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
└── README.md                     # 本文件
```

## YOLO 标注格式

每张图片对应一个同名的 `.txt` 文件，每行一个目标:

```
<class_id> <x_center> <y_center> <width> <height>
```

所有值归一化到 [0, 1]:
- `x_center`, `width` 除以图片宽度
- `y_center`, `height` 除以图片高度

## 类别映射

| class_id | 类别名称 |
|----------|----------|
| 0 | Yawn (打哈欠) |
| 1 | close_eye (闭眼) |
| 2 | noYawn (不打哈欠) |
| 3 | open_eye (睁眼) |
| 4 | face (面部) |
| 5 | smoke (吸烟) |
| 6 | drink (饮水) |
| 7 | phone (打电话) |

## 使用方式

将数据集放入对应目录后，在 YOLO 训练配置 `data.yaml` 中指定路径:

```yaml
path: ../data/yolo_format
train: train/images
val: val/images
test: test/images

names:
  0: Yawn
  1: close_eye
  2: noYawn
  3: open_eye
  4: face
  5: smoke
  6: drink
  7: phone
```
