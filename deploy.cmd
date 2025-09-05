@if "%SCM_TRACE_LEVEL%" NEQ "4" @echo off

echo Handling Python deployment.

:: 1. 安装依赖
echo Installing dependencies...
pip install -r requirements.txt

:: 2. 创建启动文件
echo Creating startup command...
echo python -m gunicorn app:app -b 0.0.0.0:%%PORT%% --workers 2 --threads 4 --timeout 60 > startup.txt

echo Finished successfully.