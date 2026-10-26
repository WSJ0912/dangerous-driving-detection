# 模型权重说明

## 生产模型

**该权重由本仓库的 `training/` 训练链路产出，是作者本人训练的成果**，不是外部获取的现成模型。

| 项目 | 值 |
|------|-----|
| 文件 | `weights/best.pt` |
| 架构 | YOLO11s（ultralytics 8.4.21 + PyTorch 2.8.0+cu128） |
| 类别数 | 8 |
| 推理输入尺寸 | **640 × 640**（letterbox，见 `backend/config.py` 的 `DEFAULT_INPUT_SIZE=640`） |
| 大小 | 19,170,394 字节 |
| SHA256 | `7AD2B81B5B1DA4F33616D179E154F73C8AA5B8339AF4F2ECD2AB2B6097FC5500` |
| 类别 | `Yawn, close_eye, noYawn, open_eye, face, smoke, drink, phone`（与 `backend/config.py` 的 `CLASS_NAMES` 顺序一致，已由 `evaluate.py --check-backend` 校验） |
| 是否入库 | **否** —— `*.pt` 被 `.gitignore` 排除，仓库内没有该文件 |

## 训练配方（可复现）

| 项 | 值 |
|------|-----|
| 起始权重 | `yolo11s.pt`（ultralytics 官方 **COCO 预训练**，首次运行自动下载） |
| 训练数据 | 8 类驾驶行为检测数据集，**6074 张 / 10,600 个标注框**（train 4572 / val 1502） |
| 轮数 / 尺寸 / 批大小 | **100 epoch / imgsz 640 / batch 16** |
| 设备 | 单卡 GPU（租用 NVIDIA RTX 3080 Ti 12GB），fp16 AMP |
| 复现命令 | `python training/train.py --data <data.yaml> --model yolo11s.pt --epochs 100 --imgsz 640 --batch 16 --device 0` |

## 实测指标（验证集 1502 张 / 2229 个实例）

| 类别 | mAP50 | mAP50-95 | 实例数 |
|------|-------|----------|--------|
| `close_eye` | **0.7734** | 0.4679 | 1438 |
| `phone` | 0.8437 | 0.4652 | 1599 |
| `smoke` | 0.6842 | 0.2851 | 1184 |
| `open_eye` | 0.6153 | 0.3070 | 1300 |
| `Yawn` | 0.4989 | 0.3494 | 213 |
| `noYawn` | 0.3121 | 0.2223 | 162 |
| `drink` | 0.9347 | 0.5940 | 1047 |
| `face` | 0.9387 | 0.5790 | 3657 |
| **总体** | **0.700** | **0.409** | 10,600 |

完整留档见 [`../training/experiments/team_ds_gpu_yolo11s_100ep/`](../training/experiments/team_ds_gpu_yolo11s_100ep/)：
`metrics.json`（含权重 SHA256 与类别一致性布尔值）、`results.csv`（100 轮逐轮指标）、
混淆矩阵与 PR/F1 曲线。

> **已知短板（如实说明）**：`Yawn`（213 个实例）与 `noYawn`（162 个）样本极少，
> 前者 mAP50 仅 0.4989、后者 0.3121 —— 而这两类恰是疲劳判定最依赖的。
> 提升只有补数据一条路：需要更多不同光照、眼镜、侧脸、夜间的驾驶座实拍。

## 权重来源与归属

- **本权重由作者本人在本仓库的训练链路上训练产出**（配方与指标见上）。
- 训练数据是一个 8 类驾驶行为检测数据集（6074 张 / 10,600 框）。
  数据本身不在本仓库内（`data/**` 被 `.gitignore` 排除），需自备同结构数据集。
- checkpoint 内嵌许可：**`license: AGPL-3.0 (https://ultralytics.com/license)`**。

### 历史权重（已备份，不参与对外表述）

`weights/best.prev.bak.pt` 是替换前的旧权重，**不是本仓库训练链路的产物**，
保留它只为出问题能回退：

- 19,161,683 字节，SHA256 `E69680C7…73BA`；
- 同一验证集上总体 mAP50 为 **0.6665**，低于现役权重的 0.700。

## 许可与合规提示

- ultralytics 生态使用 **AGPL-3.0**：分发包含该权重的软件会触发 AGPL-3.0 义务。
  **权重是自训的也一样** —— 用 ultralytics 训练出的 checkpoint 会带 AGPL 标注。
  本项目源代码已按 AGPL-3.0 发布（见 [`LICENSE`](../LICENSE)），与之一致。
- 数据集不在仓库内，因此**他人克隆后无法直接复现训练**，需自备同结构数据集。

## 关于权重文件的分发

`models/weights/best.pt` **不随仓库分发**（`*.pt` 被 `.gitignore` 排除）。
它由作者在本仓库的训练链路上训练产出，上面表里的 SHA256 用于**核对"你手上这份是不是同一份"**。

克隆本仓库后，有三种选择：

1. **自己训练一份（推荐）** —— 按 [`training/README.md`](../training/README.md) 准备好
   同结构的 8 类数据集后跑 `train.py`。产出的权重属于你自己的训练链路，
   类别顺序会被 `train.py` 与 `evaluate.py --check-backend` 强制校验，不会张冠李戴。
2. **向作者索取现成权重** —— 放到 `models/weights/best.pt`，然后用下面的命令核对：

   ```powershell
   # Windows PowerShell
   Get-FileHash -Algorithm SHA256 .\models\weights\best.pt
   (Get-Item .\models\weights\best.pt).Length
   ```

   ```bash
   # Linux / macOS
   sha256sum models/weights/best.pt && stat -c %s models/weights/best.pt
   ```

   大小与 SHA256 与上表一致，才可认为是同一份权重。
3. **用任意同类别顺序的 8 类 YOLO11 权重** —— 但**必须**先跑
   `python training/evaluate.py --weights <你的.pt> --check-backend`，
   顺序不一致时该命令以退出码 3 失败。这一步不能省。

## 更新 / 替换模型

1. 把新的 `.pt` 文件放入 `weights/` 目录（例如 `weights/my_model.pt`）。
2. 修改 `backend/config.py` 的 `MODEL_PATH`（默认 `resource_path("../models/weights/best.pt")`）。
3. **若类别数量或名称/顺序不同，必须同步修改** `backend/config.py` 中的
   `CLASS_NAMES`、`DANGER_LEVELS`（以及需要时的 `ALERT_MESSAGES`）。
4. 重启服务，用 `GET /api/health` 确认 `model_loaded=true`、`num_classes` 与 `classes` 是否符合预期。

### 换数据集后重新训练并替换（推荐）

**现役的 `best.pt` 就是这样产出的。** 换一份数据集时按下面流程重跑：

```powershell
python training/prepare_dataset.py --source <你的数据集> --format yolo --write-yaml
python training/train.py --epochs 100 --imgsz 640 --batch 16 --device 0
python training/evaluate.py --weights training/experiments/<run>/weights/best.pt --check-backend
Copy-Item models\weights\best.pt models\weights\best.prev.bak.pt    # 先备份原权重
Copy-Item training\experiments\<run>\weights\best.pt models\weights\best.pt
```

`--check-backend` 会校验新权重的类别顺序与 `backend/config.py::CLASS_NAMES` 完全一致
（不一致时以退出码 3 失败）。**这一步不能省** —— 类别顺序错位不会报任何错，
只会让每个类别的中文名张冠李戴。

## 注意

- `MODEL_PATH_FALLBACK` 指向 `models/weights/best.prev.bak.pt`（替换前的备份）。
  两者都不入库，干净克隆里都不存在 —— 需自行放置权重后服务才能检测。
- 不要提交**非本仓库训练**的 `.pt` 文件：体积大，且涉及来源与许可问题。
- 本仓库训练链路产出的权重来源清晰，但要入库仍需显式强制添加，并注意单文件约 19 MB；
  **AGPL-3.0 义务同样适用于自训权重**（用 ultralytics 训练出的 checkpoint 会带 AGPL 标注）：

  ```powershell
  git add -f models/weights/best.pt
  ```