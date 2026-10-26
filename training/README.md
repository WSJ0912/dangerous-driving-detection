# 训练链路

本目录让你能**训练出属于自己的权重**，从而把"自主训练"从说法变成事实。

> **为什么需要它**：当前生产权重 `models/weights/best.pt` 来自团队早前的「智迁驭行」项目
> （与本仓库分发的部署目录字节级相同、checkpoint 内嵌团队训练所用的集群路径、标注 AGPL-3.0），
> 详见 `models/README.md`。本仓库此前既没有数据集也没有训练代码 —— 这是最大的短板。
> 补齐训练链路后，你只要跑一次自己的训练，产出的权重与指标就都出自本仓库。

---

## 0. 先读这一段：本机 vs GPU

| 环境 | 能做什么 | 说明 |
|---|---|---|
| **本机**（AMD Ryzen 5 5600，**无 NVIDIA GPU**，torch 2.10.0+cpu） | 只能 `--smoke` 验证链路 | 实测：16 张 320×320 合成图、1 epoch、imgsz=320 ≈ **2 秒**完成 |
| **租的 GPU**（如 RTX 3090/4090） | 正式训练 | 见 §5 显存与时长建议 |

**本机跑正式训练不可行**（无 GPU，100 epoch / imgsz=640 会跑到天亮级别）。

---

## 1. 快速开始

### 1.1 只验证链路能跑通（本机，CPU，约 1 分钟）

```powershell
# 生成一份"假数据" + training/data.yaml
python training/prepare_dataset.py --synthetic --out "$env:TEMP\synthetic_ds" --write-yaml

# 冒烟训练
python training/train.py --smoke

# 评估 + 类别顺序把关
python training/evaluate.py --weights training/experiments/run/weights/best.pt --check-backend
```

> ⚠️ `--synthetic` 生成的是**随机彩色矩形，不是真实驾驶图像**。
> 它只用来证明"数据集 → 训练 → 权重 → 评估"这条管道通，**不能用来训练可用模型**。
> 冒烟模式下的 mAP 会是 0，这是正常的，**指标数值没有任何意义**。

### 1.2 正式训练（GPU 上）

```powershell
# 1) 准备并校验自己的数据集（见 §2、§3）
python training/prepare_dataset.py --source D:\my_dataset --format yolo --val-ratio 0.2 --write-yaml

# 2) 训练
python training/train.py --epochs 100 --imgsz 640 --batch 16 --device 0 --name my_yolo11s

# 3) 评估并留档（json 可作答辩/软著材料）
python training/evaluate.py --weights training/experiments/my_yolo11s/weights/best.pt `
    --check-backend --json training/experiments/my_yolo11s/metrics.json

# 4) 替换生产权重前先备份原权重，再替换（见 §6）
```

---

## 2. 从零到能训：完整步骤

```
① 获取/采集图像  →  ② 标注（8 类）  →  ③ 按目录约定摆放
        →  ④ prepare_dataset.py 校验 + 切分  →  ⑤ train.py 训练
        →  ⑥ evaluate.py 评估（含类别顺序把关）  →  ⑦ 替换 models/weights/best.pt
```

### ③ 目录约定

脚本支持两种输入：

**A. 已经是 YOLO 格式**（`images/` 与 `labels/` 成对）

```
my_dataset/
├── images/   xxx.jpg ...
└── labels/   xxx.txt ...     # 与图片同名，每行: cls cx cy w h（归一化 0~1）
```

**B. 已经分好 train/val**

```
my_dataset/
├── train/{images,labels}/
└── val/{images,labels}/
```
（脚本会保留你原有的划分，不再重新切分）

转换后的统一输出到 `data/yolo_format/{train,val}/{images,labels}/`（已在 `.gitignore` 中）。

### ④ 校验

```powershell
python training/prepare_dataset.py --source D:\my_dataset --format yolo --write-yaml
```

校验器会逐条检查并统计：

- 每行必须 5 个字段；类别号必须是 `0~7` 的整数；`cx,cy,w,h` 必须在 `[0,1]` 且 `w,h>0`
- **有图片没标签**（纯背景图请加 `--allow-background`）、**有标签没图片**、**无法解码的坏图**
- 输出**逐类实例数**，某类为 0 时会点名警告（类别不均衡会直接毁掉稀有类的指标）
- 发现问题以**退出码 3** 失败；越界坐标可用 `--clip` 自动裁剪
- 只想看看不想落盘：加 `--dry-run`

> 这一步的价值：标注错误如果留到训练阶段，通常表现为"训练正常但指标诡异"，
> 很难定位。挡在训练前最省时间。

---

## 3. 8 类的标注口径（最容易出错的地方）

类别顺序**就是类别索引**，错一位结果就张冠李戴。唯一权威定义在
`backend/config.py::CLASS_NAMES`：

| 索引 | 名称 | 标什么 |
|---|---|---|
| 0 | `Yawn` | 张嘴打哈欠的**面部/嘴部**区域 |
| 1 | `close_eye` | 眼睛**闭合**状态的眼部区域 |
| 2 | `noYawn` | 嘴部**正常闭合**（没在打哈欠） |
| 3 | `open_eye` | 眼睛**睁开**状态的眼部区域 |
| 4 | `face` | 驾驶员**整张脸**的外接框 |
| 5 | `smoke` | 吸烟（手/烟/嘴部区域） |
| 6 | `drink` | 饮水（手/瓶/嘴部区域） |
| 7 | `phone` | 手持打电话（手/手机/耳部区域） |

**易混淆点：**

- `close_eye` 与 `open_eye` 是**互斥对**：同一只眼睛只能标其中一个。
- `Yawn` 与 `noYawn` 是**互斥对**：同一个嘴部只能标其中一个。
- `face` 是其它面部类别的**父框**。后端会用"包含度"检查眼部/嘴部框是否落在脸框内
  （`backend/utils/geometry.py`），所以**同一张脸要同时标 `face` 和眼部/嘴部框**，
  否则后端的过滤逻辑拿不到父框。
- 疲劳判定的业务含义由**后端行为状态机**给出（连续帧计数 + 冷却窗口），
  不是模型直接给出的。模型只负责"这一帧眼睛是睁还是闭"。

---

## 4. 数据集从哪来

项目内**没有任何数据集**。你有三条路：

| 路线 | 说明 |
|---|---|
| **自采 + 自标** | 用手机/行车记录仪拍驾驶员视角视频，抽帧后用 labelImg / X-AnyLabeling / Roboflow 标注。**最干净，版权你自己的。** 8 类各 500~1500 个实例是个合理起点 |
| **公开数据集 + 类别映射** | Roboflow Universe 上有大量 driver behavior / drowsiness / distracted driver 数据集；StateFarm Distracted Driver、DMD 驾驶员分心数据集也是常见选择。**注意：它们的类别定义通常不是这 8 类**，需要写映射表（或只取与你 8 类重合的部分），映射错位正是下面 `--check-backend` 要拦的问题 |
| **混合** | 公开数据打底 + 自采数据补稀有类（`close_eye` / `Yawn` 通常最缺） |

### 4.1 一条已经落地的路子：Roboflow Universe 公开检测数据集

`training/fetch_dataset.py` 已内置两个**公开的、带检测框**的数据集，合起来正好覆盖本项目 8 类：

| 数据集 | 规模 | 提供的类别 |
|---|---|---|
| [Driver Drowsiness](https://universe.roboflow.com/computer-vision-tznlp/driver-drowsiness-q3msq-ddgj8) | 674 张 | `face`、`open_eye`、`closed_eye`、`open_mouth`、`closed_mouth` |
| [Reckless driver detection](https://universe.roboflow.com/ztai/reckless-driver-detection-iz7ob-apeue) | 1k 张 | `Phone`、`Cigarette`、`Drinking`（等） |

类别映射（脚本里写死并可审计）：

| 本项目类别 | 来源类别 | 说明 |
|---|---|---|
| `face` | `face` | 直接对应 |
| `open_eye` | `open_eye` | 直接对应 |
| `close_eye` | `closed_eye` | 直接对应 |
| `Yawn` | `open_mouth` | **近似**：张嘴 ≠ 打哈欠 |
| `noYawn` | `closed_mouth` | **近似** |
| `phone` | `Phone` | 直接对应 |
| `smoke` | `Cigarette` | 直接对应 |
| `drink` | `Drinking` | 直接对应 |

> ⚠️ **两个必须如实说明的点**：
> 1. `open_mouth -> Yawn` 是近似映射。训练出的 mAP **不能**直接称作"打哈欠识别精度"。
> 2. 两个数据集来自不同来源，标注风格与拍摄条件不同，存在**域偏移**；且合计仅约 1.7k 张，
>    规模远小于商用权重的训练集。**训出来的模型很可能不如现在这份现成权重。**
>    它的价值在于"这是你自己的训练产物、指标是你自己的真实数字"，而不是"效果更好"。
>
> 使用前请先用 `--list` 查看各数据集在 Roboflow 上标注的**许可**字段，
> 确认允许你的使用方式；论文/答辩中应引用其 BibTeX。

密钥读取方式（**不会被打印或写入任何文件**）：环境变量 `ROBOFLOW_API_KEY`，或
`%USERPROFILE%\.roboflow_key` 文件（单行密钥）。

```powershell
python training/fetch_dataset.py --list       # 先看版本、类别与许可
python training/fetch_dataset.py --download   # 下载 -> 重映射 -> 合并 -> 写 data.yaml
python training/prepare_dataset.py --source data/yolo_format --format yolo --dry-run
```

---

## 5. GPU 训练建议

`train.py` 的关键参数：

| 参数 | 建议 | 说明 |
|---|---|---|
| `--epochs` | 100~300 | 数据少时 100 够用；`patience` 默认会在无提升时早停 |
| `--imgsz` | **640** | **必须与后端 `config.DEFAULT_INPUT_SIZE` 一致**（现为 640）；训练/推理尺寸不一致会掉精度 |
| `--batch` | 16 | imgsz=640 时约需 12~16 GB 显存；显存不够就降到 8 或 4 |
| `--device` | `0` | 单卡；多卡写 `0,1` |
| `--workers` | 8（Linux）/ 0（Windows） | Windows 上多进程 dataloader 容易出问题 |

- **粗略时长**：3000~5000 张图、100 epoch、单张 24GB 卡大约 **2~4 小时**量级（视数据量与增强而定）。
- **起始权重**：默认用项目内 `models/weights/best.pt`（早前项目的八类权重）做迁移学习，
  收敛最快。也可以 `--model yolo11s.pt` 从官方预训练起步（**会联网下载**）。
- **中断续训**：
  ```powershell
  python training/train.py --resume training/experiments/<run>/weights/last.pt --epochs 300
  ```
- 训练产物全部落在 `training/experiments/<run>/`（已在 `.gitignore` 中，不会进仓库）。

---

## 6. 训练完的验收：**先过类别顺序这一关**

后端 `backend/services/detection.py` 用 `config.CLASS_NAMES[class_id]` 把模型的
**类别索引**翻译成业务语义。如果训练时 `data.yaml` 的 `names` 顺序与后端不一致：

- 训练照常、mAP 可能很漂亮、**不会报任何错**
- 但上线后每个类别的中文名都是错的（比如把 `phone` 显示成 `face`）

所以：

```powershell
python training/evaluate.py --weights <你的 best.pt> --data training/data.yaml --check-backend
```

- 一致 → 打印 `[OK]`，退出码 0
- 不一致 → 打印逐索引对照表，**退出码 3**

`train.py` 在训练前也会做同样的检查，不一致时**默认直接拒绝训练**（退出码 2）；
确要强行继续得加 `--allow-class-mismatch`。

### 替换生产权重

```powershell
# 1) 备份原权重（很重要，出问题能回退）
Copy-Item models\weights\best.pt models\weights\best.thirdparty.bak.pt

# 2) 用你训练出的权重替换
Copy-Item training\experiments\<run>\weights\best.pt models\weights\best.pt

# 3) 复核
python training/evaluate.py --weights models/weights/best.pt --check-backend

# 4) 启动服务，用真实图像跑一次端到端
python backend/app.py
#   打开 http://127.0.0.1:5000/detect.html
```

> `*.pt` 在 `.gitignore` 中，权重不会进仓库。想把你训练的权重也放到 GitHub，
> 需要手动 `git add -f`，并注意 19 MB 的体积。
> ⚠️ **注意：自训不会免除 AGPL-3.0 义务。** 用 ultralytics 训练出的 checkpoint 同样带
> AGPL 标注；若要闭源/商业使用，仍需确认许可或购买 Ultralytics 商业授权。
> 自训的收益是**来源清晰、指标是你自己的**，而不是「摆脱许可」。

---

## 7. 脚本一览

| 脚本 | 作用 |
|---|---|
| `common.py` | 公共模块：类别定义（取自 `backend/config.py`）、一致性校验、yaml 读写、权重解析 |
| `prepare_dataset.py` | 数据集转换 / 校验 / 切分；`--synthetic` 生成冒烟假数据 |
| `train.py` | 训练入口；`--smoke` 为 CPU 冒烟预设；训练前做类别闸门 |
| `evaluate.py` | 逐类指标 + **类别顺序把关**；`--json` 留档 |
| `export.py` | 导出 ONNX / TorchScript（生产后端目前只加载 `.pt`，此项为将来部署预留） |
| `fetch_dataset.py` | 从 Roboflow Universe 拉取公开**检测**数据集，按映射重标为本项目 8 类并合并 |
| `data.yaml.example` | 数据 yaml 模板 |

## 8. 常见问题

**Q：`train.py` 报"train 划分里没有图片"？**
A：`training/data.yaml` 默认指向 `data/yolo_format/`，你还没放数据。先跑 `prepare_dataset.py`。

**Q：类别数不是 8 的倍数/`nc` 对不上？**
A：`prepare_dataset.py` 只接受 0~7 的类别号。若你的数据集是别的类别体系，需要先写映射表。

**Q：训练很慢/卡在 CPU？**
A：确认 `--device 0` 且 `torch.cuda.is_available()` 为 True。本机没有 NVIDIA GPU，只能冒烟。

**Q：`export.py` 说缺 onnx？**
A：导出是可选功能。脚本已把 `YOLO_AUTOINSTALL` 设为 `false`，**不会**偷偷帮你
`pip install`（那会隐式改动环境且需要联网）。要导出就自己 `pip install onnx onnxslim`。

**Q：为什么 `evaluate.py` 的 mAP 是 0？**
A：如果你在用合成假数据（`--synthetic`），这是正常的 —— 随机矩形学不出东西。
指标只有在真实数据集上才有意义。