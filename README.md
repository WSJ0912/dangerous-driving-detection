# 智慧驭行驾驶行为检测系统

基于 **YOLO11s + Flask + 原生 HTML/JavaScript + SQLite** 的驾驶行为检测演示系统：
浏览器采集摄像头画面并抽帧上传，后端推理 8 类驾驶行为目标，结合多帧行为分析产生疲劳/分心/吸烟/打电话等告警，
并把检测记录、告警记录与告警截图落库，供历史查询与趋势统计。

> **仓库说明**：本仓库是**源码工程**，不含模型权重与数据集（均被 `.gitignore` 排除）。
> 克隆后请按「快速开始」第 3 步放置 `models/weights/best.pt`，否则服务能启动但检测不可用 ——
> 该文件的获取方式、校验值与来源说明见 [`models/README.md`](models/README.md)。
> 项目自带完整的[训练链路](training/README.md)：可以准备数据集、训练、评估并把关类别一致性，
> 然后用**你自己训练出的权重**替换它。

---

## 1. 功能特性

| 功能 | 状态 | 说明 |
|------|------|------|
| 摄像头实时检测 | ✅ 可用 | 浏览器 `getUserMedia` 取流，按设置间隔抽帧 → `POST /api/detect`，前端绘制检测框 |
| 8 类目标检测 | ✅ 可用 | Yawn / close_eye / noYawn / open_eye / face / smoke / drink / phone |
| 行为分析与告警 | ✅ 可用 | 多帧累计（闭眼/哈欠/分心帧数阈值 + 冷却帧）产生告警，含告警等级与提示文案 |
| 告警截图 | ✅ 可用 | 告警时保存当帧截图到 `backend/uploads/alerts/`，历史页与侧栏可查看缩略图 |
| 检测历史查询 | ✅ 可用 | 分页、按类别/告警类型筛选、时间范围筛选（`start`/`end`）、统计与 7 日趋势 |
| 数据清理 | ✅ 可用 | `DELETE /api/history` 真实删除检测记录/告警记录（前端有二次确认） |
| 服务端设置 | ✅ 可用 | `GET/PUT /api/settings` 持久化到 SQLite，检测间隔、置信度阈值、行为帧阈值、冷却帧、输入尺寸、最大检测数均可改并即时生效 |
| 健康检查 | ✅ 可用 | `GET /api/health` 返回真实模型加载状态（未加载时为 `degraded`），不硬编码 |
| 用户/权限体系 | ❌ 未实现 | 无登录、无鉴权，**不要部署到公网** |
| 生产级部署 | ❌ 未实现 | 仅 Flask 开发服务器；无 WSGI 容器、无 Docker、无反向代理配置 |
| 模型训练链路 | ✅ 已具备 | `training/`：数据集准备与标注校验、训练、逐类评估与**类别顺序把关**、ONNX 导出（见 [`training/README.md`](training/README.md)） |
| 数据集 | ❌ 仓库内无 | 需自行采集或引入公开数据集；`data/yolo_format/` 目前为空 |

---

## 2. 技术栈

| 层 | 技术 |
|----|------|
| 推理 | ultralytics YOLO11s（8 类），CPU 可跑 |
| 后端 | Python 3.13（实测）/ Flask 3.1.0 / flask-cors 6.0.2 |
| 图像 | numpy 2.1.3 / opencv-python 4.13.0.92 |
| 数据库 | SQLite（`backend/data/detection.db`，运行时自动建表） |
| 前端 | 原生 HTML + CSS + JavaScript（无框架、无 CDN、无构建步骤） |
| 测试 | pytest 8.3.4（`tests/`） |

---

## 3. 目录结构

```
基于深度学习的危险驾驶行为检测系统/
├── backend/                     # Flask 后端
│   ├── app.py                   # 入口（应用工厂 create_app + main）
│   ├── config.py                # 全局配置（路径、阈值、类别、Flask 参数）
│   ├── database.py              # SQLite 连接与读写（detection_records / alert_records / app_settings）
│   ├── models/detector.py       # YOLO 模型加载与推理封装
│   ├── routes/api.py            # 全部 HTTP 接口（Blueprint: /api/*）
│   ├── services/                # behavior.py 行为分析 / detection.py 结果解析 / tracker.py 目标追踪
│   ├── utils/                   # geometry.py 几何过滤 / image.py base64 解码与 letterbox
│   ├── data/detection.db        # SQLite 数据库（运行时生成，已忽略）
│   └── uploads/alerts/          # 告警截图（运行时生成，已忽略）
├── frontend_web/                # 手写静态前端（后端实际服务的目录）
│   ├── index.html               # 首页/总览
│   ├── detect.html              # 实时检测页
│   ├── history.html             # 历史记录页
│   ├── settings.html            # 设置页
│   └── dist/                    # 旧构建产物（孤儿目录，当前不被服务，可忽略）
├── models/
│   ├── README.md                # 模型说明（来源/许可/替换方式）
│   └── weights/best.pt          # 8 类 YOLO11s 权重（**不在仓库中**，需自行放置）
├── data/                        # 数据集目录约定（raw/、yolo_format/ 仅 .gitkeep，**无数据**）
├── training/                    # 训练链路（用来训练"你自己的"权重）
│   ├── README.md                # 训练全流程：数据准备 → 训练 → 评估 → 替换权重
│   ├── common.py                # 公共模块（类别定义直接取自 backend/config.py）
│   ├── prepare_dataset.py       # 数据集转换 / 校验 / 切分（--synthetic 生成冒烟假数据）
│   ├── train.py                 # 训练入口（--smoke 为 CPU 冒烟预设）
│   ├── evaluate.py              # 逐类指标 + 类别顺序把关（--check-backend）
│   ├── export.py                # 导出 ONNX / TorchScript（可选，为将来部署预留）
│   ├── data.yaml.example        # 数据 yaml 模板
│   └── experiments/             # 训练产物（已忽略）
├── scripts/                     # start.bat / start.ps1 / start.sh 启动脚本
├── tests/                       # pytest 测试（数据库隔离、几何、行为逻辑、API）
├── 深度学习/                     # 早期临时脚本残留（111.py），非正式代码
├── yolo11-drive/                # 第三方 YOLO 部署 demo 包（约 123 MB，已忽略，非本项目源码）
├── release/                     # 2026-05-05 旧架构快照（约 1.2 GB，已忽略，不可直接交付）
├── 软著材料_初版/                # 软著申报材料（约 53 MB，含个人/单位信息，已忽略）
├── README.md                    # 本文件
├── 修复记录.md                   # 本次修复的问题 → 修法 → 影响文件
├── requirements.txt             # 运行依赖（实测版本）
└── requirements-dev.txt         # 开发/测试依赖
```

---

## 4. 快速开始

### 4.1 环境要求

- **Python 3.10+**（本项目实测环境：Windows + Anaconda Python **3.13.5**）
- 内存 ≥ 4 GB（加载 YOLO11s 权重约需 1 GB 左右）
- 无需 GPU；`torch` 装 CPU 版即可运行（有 CUDA 会自动用 GPU）
- 浏览器：Chrome / Edge 等支持 `getUserMedia` 的现代浏览器

### 4.2 安装依赖

```bash
# 方式 A：系统 Python（最快）
python -m pip install -r requirements.txt

# 方式 B：项目内虚拟环境（推荐，隔离干净）
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/mac: source .venv/bin/activate
pip install -r requirements.txt
```

如果需要跑测试，再装一个：`pip install -r requirements-dev.txt`。

> CPU-only 环境想省下载体积时，可先单独装 CPU 版 PyTorch：
> `pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu`，再执行上面的安装命令。
>
> 依赖注意事项：不要同时安装 `opencv-python-headless`（它与 `opencv-python` 提供同名 `cv2` 模块，
> 会互相覆盖，导致 `import cv2` 不可用）；也无需安装 `gunicorn`（Windows 不支持，
> 本项目使用 Flask 开发服务器 `app.run`，生产部署请自行引入 WSGI 服务器）。

### 4.3 放置模型权重（**必做**）

`*.pt` 已被 `.gitignore` 排除，**克隆下来的仓库里没有模型文件**。请自行获取 8 类 YOLO11s 权重并放到：

```
models/weights/best.pt
```

放置后核对文件信息（与文档记录一致才可认为是同一份权重）：

| 项目 | 值 |
|------|-----|
| 大小 | `19,161,683` 字节 |
| SHA256 | `E69680C7206A2CC1E18F8EB0C64CE068CEB00018E5CC4EC0F5069A34492573BA` |

Windows PowerShell 校验：

```powershell
Get-FileHash -Algorithm SHA256 .\models\weights\best.pt
```

Linux/macOS：

```bash
sha256sum models/weights/best.pt
```

权重的来源、许可与替换方式见 [`models/README.md`](models/README.md) 与本文「模型来源与许可声明」。

### 4.4 启动

```bash
# 在项目根目录执行
python backend/app.py
```

或用启动脚本（会自动选择解释器、检查依赖与权重）：

| 平台 | 命令 |
|------|------|
| Windows PowerShell | `.\scripts\start.ps1` |
| Windows CMD | `scripts\start.bat` |
| Linux / macOS | `bash scripts/start.sh` |

启动成功的输出类似：

```
服务地址: http://0.0.0.0:5000
 * Running on http://127.0.0.1:5000
```

### 4.5 访问

浏览器打开 **http://127.0.0.1:5000** （端口见 `backend/config.py` 的 `FLASK_PORT`）。

> 摄像头页面在 `http://127.0.0.1` 或 `https://` 下才被浏览器允许访问摄像头；
> 若用局域网 IP（如 `http://192.168.x.x:5000`）访问，浏览器可能因非安全上下文而拒绝授权，
> 此时可在浏览器 `chrome://flags/#unsafely-treat-insecure-origin-as-secure` 中把该源加入白名单。

### 4.6 首次自检

```bash
curl http://127.0.0.1:5000/api/health
```

- `model_loaded=true` → 模型加载成功，可以开始检测；
- `status=degraded` / `model_loaded=false` → 权重缺失或加载失败，请回到 4.3 检查路径与文件。

---

## 5. API 一览

所有接口前缀 `/api`；**未知的 `/api/*` 路径或方法返回 JSON 格式的 404/405**（不会回落到 HTML 页面）。

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/detect` | 检测单帧图像 |
| GET | `/api/health` | 健康检查与模型真实状态 |
| GET | `/api/history` | 历史记录 / 统计 / 趋势（支持 `DELETE` 清理） |
| DELETE | `/api/history` | 删除历史数据 |
| GET | `/api/settings` | 读取服务端设置 |
| PUT | `/api/settings` | 更新服务端设置 |
| GET | `/api/screenshots/<filename>` | 读取告警截图 |

### 5.1 POST /api/detect

请求体（JSON）：

```json
{
  "image": "<base64 字符串，可带 data:image/jpeg;base64, 前缀>",
  "conf": 0.3,
  "iou": 0.5
}
```

- `image`：必填。纯 base64 或 dataURL 均可（后端会自动剥离前缀）。
- `conf` / `iou`：可选，取值 0~1，越界会被夹取到边界；不传则使用服务端设置里的 `conf_threshold`。

响应（JSON）：

```json
{
  "success": true,
  "detections": [
    {"class": "face", "confidence": 0.91, "bbox": [x1, y1, x2, y2],
     "danger_level": "low", "track_id": 3}
  ],
  "alerts": [
    {"type": "eye_close", "status": "active", "message": "驾驶员眼睛闭合，请注意安全！", "frame_count": 3}
  ],
  "behavior_state": {"eye_status": "open", "yawn_status": "noYawn", "distracted": false,
                     "eye_close_frames": 0, "yawn_frames": 0, "distracted_frames": 0},
  "image_width": 1280,
  "image_height": 720,
  "input_size": 640
}
```

`behavior_state` 为当前行为状态：`eye_status`（`open` / `closed`）、`yawn_status`（`Yawn` / `noYawn`）、
`distracted`（是否分心）以及三个累计帧计数（`eye_close_frames` / `yawn_frames` / `distracted_frames`）。

> **坐标约定**：`bbox` 一律是**原图像素坐标**（左上 `x1,y1`，右下 `x2,y2`），
> 已由 letterbox 的缩放比与填充量还原；请用响应里的 `image_width`/`image_height` 换算绘制，
> 不要再按 640/416 之类的固定值缩放。

### 5.2 GET /api/health

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "...\\models\\weights\\best.pt",
  "device": "cpu",
  "num_classes": 8,
  "classes": ["Yawn", "close_eye", "noYawn", "open_eye", "face", "smoke", "drink", "phone"]
}
```

模型未加载时返回 `status="degraded"`、`model_loaded=false`，HTTP 状态码仍为 200。

### 5.3 GET /api/history

查询参数（均可选）：

| 参数 | 说明 |
|------|------|
| `limit` / `offset` | 分页（默认 `limit=50`、`offset=0`，`limit` 上限见后端 `_HISTORY_LIMIT_MAX`） |
| `class` | 按检测类别过滤 |
| `alert_type` | 按告警类型过滤（`eye_close` / `yawn` / `distracted`） |
| `start` / `end` | 时间范围（需**同时提供**；历史页的日期筛选） |
| `stats=1` | 返回汇总统计 |
| `trend=1&days=7` | 返回近 N 天趋势序列 |

告警行附带 `screenshot_url`（形如 `/api/screenshots/xxx.jpg`），可直接用于 `<img src>`。

### 5.4 DELETE /api/history

```
DELETE /api/history?scope=all|detections|alerts
→ {"success": true, "deleted": {"detections": 12, "alerts": 3}}
```

### 5.5 GET / PUT /api/settings

```json
{
  "detect_interval_ms": 300,
  "conf_threshold": 0.3,
  "iou_threshold": 0.5,
  "input_size": 640,
  "max_detections": 10,
  "eye_close_frames": 3,
  "yawn_frames": 2,
  "distracted_frames": 5,
  "cooldown_frames": 30
}
```

取值范围（后端校验，越界或类型错误返回 400）：
`detect_interval_ms` 100–5000、`conf_threshold` / `iou_threshold` 0.01–1.0、
`input_size` 320–1280、`max_detections` 1–100、
`eye_close_frames` / `yawn_frames` / `distracted_frames` 1–100、`cooldown_frames` 0–600。

- `GET` 返回 `{"success": true, "settings": {...}}`；未修改过时返回默认值。
- `PUT` 请求体为设置对象，只接受白名单字段并做类型/范围校验；返回更新后的完整设置。
- 设置持久化在 SQLite 的 `app_settings` 表；`conf_threshold` 会影响后续 `/api/detect` 的默认置信度，
  帧阈值与冷却帧会影响行为分析与告警。

### 5.6 GET /api/screenshots/\<filename\>

返回告警截图图片。**安全约束**：只接受文件名（basename），只允许 `jpg` / `png`，
且必须位于 `backend/uploads/alerts/` 目录内；`../` 之类的路径穿越请求会被拒绝。

---

## 6. 配置说明

### 6.1 `backend/config.py`（代码级配置）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MODEL_PATH` | `models/weights/best.pt` | 权重路径（含 PyInstaller 兼容的 `resource_path`） |
| `DB_PATH` | `backend/data/detection.db` | SQLite 数据库 |
| `ALERT_SCREENSHOT_DIR` | `backend/uploads/alerts/` | 告警截图目录 |
| `DEFAULT_INPUT_SIZE` | `640` | 推理输入尺寸（letterbox 目标尺寸） |
| `DETECTION_CONF_THRESHOLD` | `0.3` | 默认置信度阈值 |
| `DETECTION_IOU_THRESHOLD` | `0.5` | NMS IoU 阈值 |
| `MAX_DETECTIONS` | `10` | 单帧最大检测数 |
| `MIN_BOX_SIZE` / `MIN_ASPECT_RATIO` / `MAX_ASPECT_RATIO` | `20` / `0.2` / `5.0` | 小框、异常长宽比过滤 |
| `FACE_CONTAINMENT_MIN` | `0.5` | 眼部/嘴部框需被脸部框包含的比例（低于该值才丢弃） |
| `TRACK_IOU_THRESHOLD` | `0.3` | 目标追踪匹配 IoU |
| `WARNING_THRESHOLDS` | `eye_close=3, yawn=2, distracted=5` | 行为告警帧阈值 |
| `WARNING_COOLDOWN_FRAMES` | `30` | 同一类告警的冷却帧数 |
| `CLASS_NAMES` / `DANGER_LEVELS` / `ALERT_MESSAGES` | 8 类 | 类别、危险等级、告警文案 |
| `FLASK_HOST` / `FLASK_PORT` / `FLASK_DEBUG` | `0.0.0.0` / `5000` / `False` | 服务监听参数 |
| `MAX_CONTENT_LENGTH` | 16 MB | 单请求体上限 |

> 若你放置的权重类别顺序与本项目不同，必须同步修改 `CLASS_NAMES` / `DANGER_LEVELS`。

### 6.2 运行时设置（`/api/settings`）

检测间隔、置信度阈值、行为帧阈值、冷却帧、输入尺寸、最大检测数属于**运行时设置**，
可以通过设置页或 `PUT /api/settings` 修改，无需改代码、无需重启；默认值取自 `config.DEFAULT_SETTINGS`。

---

## 7. 模型权重来源与许可

`models/weights/best.pt` 的**真实来源**如下，本项目在文档、答辩、软著与论文中**不得**声称该权重出自本项目的训练过程，
也不得声称训练数据集由本项目构建：

1. **该权重不是本项目的训练产物。** 它与第三方 YOLO 部署 demo 包中的
   `yolo11-drive/42_demo/runs/yolo11s_pretrained/train3/weights/best.pt` **字节级完全相同**
   （同为 19,161,683 字节，SHA256 同为 `E69680C7206A2CC1E18F8EB0C64CE068CEB00018E5CC4EC0F5069A34492573BA`），
   即本地权重是从该 demo 包中原样拷贝而来的。
2. **checkpoint 内嵌的训练痕迹指向他人机器。** 权重 pickle 内的 `train_args.data` 为
   `/public/home/zhanghanyu/songcm/yolo_project/399/...`（他人 Linux 集群账号路径）；
   元数据 `date=2025-02-15T20:50:14`、`version=8.3.18`、`epoch=-1`、`best_fitness=None`。
   本机不存在该路径，也不存在对应的数据集 yaml。
3. **checkpoint 内嵌许可为 AGPL-3.0**（`license: AGPL-3.0 (https://ultralytics.com/license)`）。
   ultralytics 生态采用 AGPL-3.0：**分发包含该权重的软件会触发 AGPL-3.0 义务**
   （开源整个衍生作品，或向 Ultralytics 购买商业许可）。软著申报、对外发布或商业化之前，
   必须与指导方/权利方确认权重来源、数据集授权与许可证义务。
4. **仓库内没有训练数据，也没有训练脚本。** `data/raw/**`、`data/yolo_format/**` 只有 `.gitkeep`
   占位；`training/experiments/` 为空；全仓库检索不到任何训练入口（无 `train.py`、无 `model.train` 调用）。
   因此**无法在本机复现该权重的训练过程**。

**可以如实表述的范围**：本项目实现的是"**集成第三方 8 类 YOLO11s 检测权重 + 自研的行为分析、
告警生命周期、截图落库、历史统计与 Web 交互链路**"。
"基于深度学习"指的正是这条"用深度学习模型做视觉检测"的集成链路，而不是"自研训练"。

---

## 8. 数据集与训练现状

**训练链路已经具备**（`training/`，完整说明见 [`training/README.md`](training/README.md)），
**但数据集仍需你自己准备** —— 仓库里没有任何真实图像与标注。

| 项目 | 现状 |
|------|------|
| 训练脚本 | ✅ `training/{prepare_dataset,train,evaluate,export}.py` |
| 数据集 yaml | ✅ 由 `prepare_dataset.py --write-yaml` 生成，模板见 `training/data.yaml.example` |
| `data/raw/**`、`data/yolo_format/**` | ⚠️ 仅 `.gitkeep`，**无任何图片/标注**；需自行采集或引入公开数据集 |
| `training/experiments/` | 目录已建，训练后在此产出权重、曲线与 `results.csv` |
| 生产权重 | ❌ 仍为第三方权重（见第 7 节），**尚无属于本项目的训练产物** |

**训练出你自己的权重后，"基于深度学习"就可以如实写成"自研训练"了。** 三步：

```powershell
# 1) 准备并校验数据集（会检查标注格式、类别号、逐类实例数）
python training/prepare_dataset.py --source <你的数据集> --format yolo --write-yaml

# 2) 训练（正式训练需要在有 NVIDIA GPU 的机器上；本机只能 --smoke 验证管道）
python training/train.py --epochs 100 --imgsz 640 --batch 16 --device 0

# 3) 评估，并把关"类别顺序能否安全上线"（不一致会以退出码 3 失败）
python training/evaluate.py --weights training/experiments/<run>/weights/best.pt --check-backend
```

> ⚠️ **类别顺序是本项目最容易踩的坑**：后端用 `CLASS_NAMES[class_id]` 翻译类别名，
> 若训练时 `data.yaml` 的 `names` 顺序与 `backend/config.py` 不一致，模型照样能训、
> mAP 也可能很好看，但上线后**每一类的名字都会张冠李戴且不报错**。
> `train.py` 与 `evaluate.py --check-backend` 就是拦这个的。

目录约定与 YOLO 标注格式说明另见 [`data/README.md`](data/README.md)。

---

## 9. 已知限制与风险

1. **模型非自研**：见第 7 节；涉及 AGPL-3.0 义务与来源合规，对外表述必须如实。
2. **本机无法正式训练**：训练链路已具备（`training/`），但本机**无 NVIDIA GPU**（`torch.cuda.is_available()` 为 False），只能跑 `--smoke` 验证管道；正式训练需在 GPU 上跑。此外数据集仍需自行准备（见第 8 节）。
3. **无鉴权、无防护**：任何能访问端口的人都可以调用检测接口、下载截图、**清空历史数据**。
   仅供本机/内网演示，切勿暴露到公网。
4. **仅开发服务器**：`app.run()` 启的是 Flask 开发服务器，`FLASK_DEBUG=False`；
   不适合高并发生产场景，也没有容器化/反向代理配置。
5. **版本控制基线缺失**：本目录下的 `.git` 是**一次失败的 `git init` 残留**
   （外层只有模板文件与陈旧的 `config.lock`，元数据被写在 `.git\.git\` 里），
   `git status` 会报 `not a git repository`；其中**没有任何提交、任何远端**，
   因此也不存在可回滚的历史。重建仓库不会丢失任何版本（对象库为空）。
6. **`release/` 不可直接交付**：它是 2026-05-05 的旧架构快照（单体 `app.py` + `dist/` 前端），
   且自带 venv 的基础解释器指向本机不存在的 `E:\python`（实测无法启动）。已整体 `.gitignore`。
7. **`frontend_web/dist/` 是孤儿产物**：当前后端服务的是 `frontend_web/` 根目录下的 4 个手写页面，
   `dist/` 不会被服务也不会被重建，属历史构建残留。
8. **`backend/.venv` 半损坏**：缺 `werkzeug`（`import flask` 失败）、无 `activate` 脚本，
   且混有 3.10/3.13 两套 `.pyc`。**请勿使用它**，按 4.2 重新建环境；启动脚本也会自动跳过不可用的解释器。
9. **软著材料缺口**：`软著材料_初版/` 中的说明书目前**零真实界面截图**（附录图全为占位）、
   源代码文档约 38–42 页（低于 60 页惯例）、系统名称/版本号在目录名、前端、软著与申报表之间
   有 4 处不一致，提交前必须统一口径并脱敏。该目录含个人信息，已被 `.gitignore` 排除。
10. **精度未评估**：没有在本项目数据上做过精度评测；检测效果取决于第三方权重与摄像头画质。
11. **UI 命名**：设置页出现 `v2.0`、软著材料使用 `V1.0`，尚未统一。

---

## 10. 测试

```bash
python -m pytest tests -v
```

测试使用 `tmp_path` 把数据库重定向到临时目录，**不会写入** `backend/data/detection.db`；
API 测试用 test_client + 检测器桩，不加载 torch。

当前共 **73 个用例**（本机实测全绿）。其中只有 `test_smoke_real_model.py` 会真正加载权重，其余用例把检测器桩掉 —— 所以那条冒烟用例是检测链路唯一的回归保护。

| 文件 | 覆盖内容 |
|------|----------|
| `tests/test_database.py` | 数据库读写与隔离（临时库） |
| `tests/test_geometry.py` | `containment_ratio` / `filter_detections` 包含语义（脸部内的小眼睛框应保留） |
| `tests/test_logic.py` | 行为分析与告警生命周期逻辑 |
| `tests/test_api.py` | `/api/health`、`/api/settings`、`/api/history`、`DELETE /api/history` 等接口 |
| `tests/test_smoke_real_model.py` | **真实权重 + 真实图像**走完整推理链路（检测数 > 0、bbox 落在原图内、类别顺序与后端一致）；权重或样例图缺失时自动 skip |

---

## 11. 其他文档

| 文件 | 内容 |
|------|------|
| [`models/README.md`](models/README.md) | 权重来源、SHA256、许可、替换模型的方法 |
| [`data/README.md`](data/README.md) | 数据集目录约定、YOLO 标注格式、8 类映射 |
| [`training/README.md`](training/README.md) | **训练全流程**：数据获取与标注口径、GPU 建议、类别顺序把关、替换权重 |
| [`修复记录.md`](修复记录.md) | 本次修复的问题 → 修法 → 影响文件 |
| `_审计报告/` | 审计与验证报告（本地过程材料，已被 `.gitignore` 排除） |
| `软著材料_初版/` | 软著申报材料（含个人信息，已被 `.gitignore` 排除） |

---

## 12. 常见问题

| 现象 | 原因与处理 |
|------|------------|
| `/api/health` 返回 `degraded` | 权重缺失/路径不对/依赖缺失；确认 `models/weights/best.pt` 存在且大小一致 |
| 检测页一直没有框 | 摄像头未授权（需 `127.0.0.1` 或 HTTPS）、或后端返回 0 检测（画面中无驾驶员）；打开浏览器控制台与页面状态栏看错误 |
| 启动脚本报"未找到 Python" | 系统 PATH 里没有 Python；用 `python -m venv .venv` 建环境或直接 `python backend/app.py` |
| `ModuleNotFoundError: No module named 'flask'` | 依赖没装：`python -m pip install -r requirements.txt` |
| 端口 5000 被占用 | 改 `backend/config.py` 的 `FLASK_PORT`，或结束占用进程 |
| 历史/告警截图 404 | 截图文件已被清理或不在 `backend/uploads/alerts/`；历史表里的 `screenshot_url` 指向旧文件名 |
| 想训练自己的模型 | 见 [`training/README.md`](training/README.md)。最短路径：`prepare_dataset.py --synthetic` 冒烟验证管道 → 换成真实数据 → `train.py --device 0` → `evaluate.py --check-backend` |
| 训练报「train 划分里没有图片」 | `training/data.yaml` 指向 `data/yolo_format/`，而你还没放数据；先跑 `prepare_dataset.py` |