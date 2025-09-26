#!/bin/bash

echo "=== Starting RAG Application ==="
cd /home/site/wwwroot || exit 1

# 设置端口
PORT=${PORT:-8000}
echo "Using port: $PORT"

# 检查并使用虚拟环境
if [ -d "antenv" ]; then
    echo "=== 激活虚拟环境 ==="
    source antenv/bin/activate
else
    echo "=== 创建虚拟环境 ==="
    python3 -m venv antenv
    source antenv/bin/activate
    python3 -m pip install --upgrade pip
fi

echo "Using Python: $(which python3)"
python3 --version

# 安装依赖
echo "=== Installing Dependencies ==="
python3 -m pip install -r requirements.txt --no-cache-dir

# 检查安装
echo "=== 验证安装 ==="
python3 -m pip list | grep -E "(uvicorn|fastapi)"

# 启动应用
echo "=== Starting uvicorn Server ==="
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port $PORT