@echo off
chcp 65001 >nul
rem ============================================================
rem  智慧驭行驾驶行为检测系统 - Windows 启动脚本（批处理）
rem  用法：scripts\start.bat（在任意目录双击/执行均可）
rem  行为：优先使用项目内虚拟环境，否则回退系统 python；
rem        检查依赖与模型权重后启动 python backend\app.py
rem ============================================================
setlocal

set "ROOT=%~dp0.."
set "PYEXE="
set "PYARGS="

echo ================================================
echo       智慧驭行驾驶行为检测系统
echo ================================================
echo.

rem ---- 1) 优先项目内虚拟环境（要求能 import flask）----
for %%V in ("%ROOT%\.venv\Scripts\python.exe" "%ROOT%\backend\.venv\Scripts\python.exe") do (
    if not defined PYEXE (
        if exist "%%~fV" (
            "%%~fV" -c "import flask" >nul 2>nul && set "PYEXE=%%~fV"
        )
    )
)

rem ---- 2) 回退系统 python（排除 Microsoft Store 的占位程序）----
if not defined PYEXE (
    python -c "import flask" >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
    py -3 -c "import flask" >nul 2>nul && (set "PYEXE=py" & set "PYARGS=-3")
)
if not defined PYEXE (
    python -V >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
    py -3 -V >nul 2>nul && (set "PYEXE=py" & set "PYARGS=-3")
)

if not defined PYEXE (
    echo [错误] 未找到可用的 Python 解释器。
    echo        请安装 Python 3.10+ 并加入 PATH，或创建项目内虚拟环境：
    echo          python -m venv .venv
    echo          .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo [信息] 使用解释器: %PYEXE% %PYARGS%

rem ---- 3) 检查运行依赖 ----
"%PYEXE%" %PYARGS% -c "import flask, cv2, numpy" >nul 2>nul
if errorlevel 1 (
    echo [错误] 运行依赖不完整（需要 flask / opencv-python / numpy）。请执行：
    echo         "%PYEXE%" %PYARGS% -m pip install -r "%ROOT%\requirements.txt"
    pause
    exit /b 1
)

rem ---- 4) 检查模型权重 ----
if exist "%ROOT%\models\weights\best.pt" (
    for %%F in ("%ROOT%\models\weights\best.pt") do echo [信息] 模型权重: models\weights\best.pt ^(%%~zF 字节^)
) else (
    echo [警告] 未找到 models\weights\best.pt —— 服务仍可启动，但 /api/detect 无法检测。
    echo         请放置 8 类 YOLO11s 权重（说明见 models\README.md），SHA256 前缀 E69680C7。
)

echo.
echo 服务地址: http://127.0.0.1:5000
echo 按 Ctrl+C 停止服务
echo.

rem ---- 5) 启动服务 ----
pushd "%ROOT%\backend"
"%PYEXE%" %PYARGS% app.py
popd

pause
endlocal