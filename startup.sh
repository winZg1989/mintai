#!/bin/bash

echo "=== Starting RAG Application ==="
cd /home/site/wwwroot || exit 1

# 添加用户bin目录到PATH
#export PATH="/root/.local/bin:$PATH"

echo "Using Python: $(which python3)"
python3 --version

# 使用Azure提供的端口
PORT=${PORT:-8000}
echo "Using port: $PORT"

# 安装依赖到用户目录
echo "=== Installing Dependencies ==="
#python3 -m pip install --user --upgrade pip
#python3 -m pip install --user -r requirements.txt --no-cache-dir

# 再次确保PATH正确
#export PATH="/root/.local/bin:$PATH"

# 等待安装完成
#sleep 10

# 启动应用
echo "=== Starting uvicorn Server ==="
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
