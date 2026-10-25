# 智慧驭行驾驶行为检测系统 - Windows 启动脚本（PowerShell）
#
# 用法：
#   .\scripts\start.ps1        （在项目根或任意目录执行均可）
#
# 行为：
#   1) 优先使用项目内虚拟环境（<root>\.venv 或 <root>\backend\.venv），且要求该解释器能 import flask；
#      否则回退到系统 python / py -3；
#   2) 检查 flask / cv2 / numpy 是否可导入，缺依赖时给出安装命令；
#   3) 检查 models\weights\best.pt 是否存在，并提示大小是否与文档一致；
#   4) 启动 python backend\app.py。
#
# 注意：本文件必须保存为 UTF-8 **带 BOM**，否则 Windows PowerShell 5.1 下中文会乱码。

$ErrorActionPreference = 'Continue'

$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath (Join-Path $Root 'backend')

Write-Host '================================================'
Write-Host '  智慧驭行驾驶行为检测系统'
Write-Host '================================================'
Write-Host ''

# ---- 选择解释器：能 import flask 的优先，项目内 venv 优先于系统环境 ----
function Test-PythonFlask {
    param([string]$Exe, [string[]]$Pre = @())
    if (-not (Test-Path -LiteralPath $Exe)) { return $false }
    $global:LASTEXITCODE = 1
    & $Exe @Pre -c 'import flask' *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-PythonRuns {
    param([string]$Exe, [string[]]$Pre = @())
    if (-not (Test-Path -LiteralPath $Exe)) { return $false }
    $global:LASTEXITCODE = 1
    & $Exe @Pre -c 'pass' *> $null
    return ($LASTEXITCODE -eq 0)
}

$candidates = New-Object System.Collections.ArrayList
[void]$candidates.Add(@{ Exe = (Join-Path $Root '.venv\Scripts\python.exe');         Pre = @() })
[void]$candidates.Add(@{ Exe = (Join-Path $Root 'backend\.venv\Scripts\python.exe'); Pre = @() })

$sysPython = Get-Command python -ErrorAction SilentlyContinue
if ($sysPython -and $sysPython.Source -notlike '*\WindowsApps\*') {
    [void]$candidates.Add(@{ Exe = $sysPython.Source; Pre = @() })
}
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    [void]$candidates.Add(@{ Exe = $pyLauncher.Source; Pre = @('-3') })
}

$chosen = $null
foreach ($c in $candidates) {
    if (Test-PythonFlask -Exe $c.Exe -Pre $c.Pre) { $chosen = $c; break }
}
if (-not $chosen) {
    foreach ($c in $candidates) {
        if (Test-PythonRuns -Exe $c.Exe -Pre $c.Pre) { $chosen = $c; break }
    }
}

if (-not $chosen) {
    Write-Host '[错误] 未找到可用的 Python 解释器。'
    Write-Host '       请安装 Python 3.10+ 并加入 PATH，或创建项目内虚拟环境：'
    Write-Host '         python -m venv .venv'
    Write-Host '         .\.venv\Scripts\python.exe -m pip install -r requirements.txt'
    exit 1
}

$pythonExe  = $chosen.Exe
$pythonPre  = $chosen.Pre
$pythonDesc = if ($pythonPre.Count -gt 0) { "$pythonExe $($pythonPre -join ' ')" } else { $pythonExe }
Write-Host "[信息] 使用解释器: $pythonDesc"

# ---- 检查运行依赖 ----
$global:LASTEXITCODE = 1
& $pythonExe @pythonPre -c 'import flask, cv2, numpy' *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host '[错误] 运行依赖不完整（需要 flask / opencv-python / numpy）。请执行：'
    Write-Host "         & '$pythonExe' $($pythonPre -join ' ') -m pip install -r `"$Root\requirements.txt`""
    exit 1
}

# ---- 检查模型权重 ----
$weights = Join-Path $Root 'models\weights\best.pt'
$expectedSize = 19161683
if (Test-Path -LiteralPath $weights) {
    $size = (Get-Item -LiteralPath $weights).Length
    Write-Host "[信息] 模型权重: models\weights\best.pt ($size 字节)"
    if ($size -ne $expectedSize) {
        Write-Host "[警告] 文件大小与文档记录（$expectedSize 字节）不一致，请核对来源与 SHA256（见 models\README.md）。"
    }
} else {
    Write-Host '[警告] 未找到 models\weights\best.pt —— 服务仍可启动，但 /api/detect 无法检测。'
    Write-Host '        请放置 8 类 YOLO11s 权重（说明见 models\README.md），SHA256 前缀 E69680C7。'
}

Write-Host ''
Write-Host '服务地址: http://127.0.0.1:5000'
Write-Host '按 Ctrl+C 停止服务'
Write-Host ''

# ---- 启动服务 ----
& $pythonExe @pythonPre 'app.py'
exit $LASTEXITCODE