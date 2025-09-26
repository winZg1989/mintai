#!/bin/bash

echo "=== Starting RAG Application ==="
cd /home/site/wwwroot || exit 1

echo "Using Python: $(which python3)"
python3 --version

# 使用Azure提供的端口
PORT=${PORT:-8000}
echo "Using port: $PORT"

# 安装依赖
echo "=== Installing Dependencies ==="
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt --no-cache-dir

# 等待安装完成
sleep 10

# 检查uvicorn是否安装成功
echo "=== 检查已安装的包 ==="
python3 -m pip list | grep uvicorn

# 启动应用
echo "=== Starting uvicorn Server ==="
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port $PORT