# 模型权重说明

## 生产模型

| 项目 | 值 |
|------|-----|
| 文件 | `weights/best.pt` |
| 架构 | YOLO11s（ultralytics） |
| 类别数 | 8 |
| 推理输入尺寸 | **640 × 640**（letterbox，见 `backend/config.py` 的 `DEFAULT_INPUT_SIZE=640`） |
| 大小 | 19,161,683 字节 |
| SHA256 | `E69680C7206A2CC1E18F8EB0C64CE068CEB00018E5CC4EC0F5069A34492573BA` |
| 类别 | `Yawn, close_eye, noYawn, open_eye, face, smoke, drink, phone`（与 `backend/config.py` 的 `CLASS_NAMES` 顺序一致） |
| 是否入库 | **否** —— `*.pt` 被 `.gitignore` 排除，仓库内没有该文件 |

## 权重来源

该权重是一个 **8 类 YOLO11s 权重，来自本团队早前的「智迁驭行」项目**，
**不是本仓库 `training/` 训练链路的产物**。可核实的事实：

1. 与该早前项目随包分发的
   `yolo11-drive/42_demo/runs/yolo11s_pretrained/train3/weights/best.pt`
   **字节级完全相同**（19,161,683 字节，SHA256 `E69680C7…73BA`）。
2. checkpoint 内嵌 `train_args.data` 为
   `/public/home/zhanghanyu/songcm/yolo_project/399/...`（团队训练时使用的集群路径）；
   元数据 `date=2025-02-15T20:50:14`、`version=8.3.18`、`epoch=-1`、`best_fitness=None`。
   本机不存在该路径；随包分发的数据集只有标注、没有图像，故无法复现该次训练。
3. checkpoint 内嵌许可：**`license: AGPL-3.0 (https://ultralytics.com/license)`**。

## 许可与合规提示

- ultralytics 生态使用 **AGPL-3.0**：分发包含该权重的软件会触发 AGPL-3.0 义务。
  **注意：用 ultralytics 自己训练出的权重同样带 AGPL 标注，自训不会让这个义务消失。**
- 本仓库**没有**数据集（`data/` 仅 `.gitkeep`），因此无法复现该权重的训练过程。
- 项目自带**训练链路**（`training/`，见 [`training/README.md`](../training/README.md)）：
  可在自有数据集上重新训练，产出的权重是本仓库训练链路的产物，来源清晰、指标是你自己的真实数字。

## 如何获取 / 放置模型

克隆仓库后 `models/weights/` 内不会有 `.pt` 文件（已被 `.gitignore` 排除），需要自行放置：

```text
models/weights/best.pt
```

放置后核对：

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

## 更新 / 替换模型

1. 把新的 `.pt` 文件放入 `weights/` 目录（例如 `weights/my_model.pt`）。
2. 修改 `backend/config.py` 的 `MODEL_PATH`（默认 `resource_path("../models/weights/best.pt")`）。
3. **若类别数量或名称/顺序不同，必须同步修改** `backend/config.py` 中的
   `CLASS_NAMES`、`DANGER_LEVELS`（以及需要时的 `ALERT_MESSAGES`）。
4. 重启服务，用 `GET /api/health` 确认 `model_loaded=true`、`num_classes` 与 `classes` 是否符合预期。

### 用自己的训练产物替换（推荐）

想让权重完全出自本仓库的训练链路，就跑一遍下面这个流程：

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

- 旧的 `MODEL_PATH_FALLBACK` 指向早前项目的 YOLO 部署目录（`yolo11-drive/.../train3/weights/best.pt`），
  该目录已被 `.gitignore` 排除；在干净克隆的环境里它不存在，请以 `weights/best.pt` 为准。
- 不要提交**非本仓库训练**的 `.pt` 文件：体积大，且涉及来源与许可问题。
- 本仓库训练链路产出的权重来源清晰，但要入库仍需显式强制添加，并注意单文件约 19 MB；
  **AGPL-3.0 义务同样适用于自训权重**（用 ultralytics 训练出的 checkpoint 会带 AGPL 标注）：

  ```powershell
  git add -f models/weights/best.pt
  ```