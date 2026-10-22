@echo off
:: 设置编码为 UTF-8，防止中文乱码
chcp 65001 >nul
:: 设置窗口标题
title 颜月卿的 AI Agent 启动器

:: 切换到当前 bat 文件所在的目录（保证在任何地方双击都能找到项目）
cd /d "%~dp0"

echo =======================================
echo    正在启动 AI Agent 项目...
echo =======================================
echo.

:: 激活虚拟环境
echo [1/2] 正在激活虚拟环境 (.venv)...
call .venv\Scripts\activate.bat

:: 启动服务器
echo [2/2] 正在启动 FastAPI 服务器...
echo.
uvicorn main:app --reload

:: 如果服务器崩溃或按了 Ctrl+C，暂停窗口，防止黑框直接消失
echo.
echo 服务器已停止。按任意键关闭窗口...
pause >nul