# 使用官方 Python 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器中
COPY . /app

# 安装系统依赖（如果要处理 PDF/Docx）
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露容器端口
EXPOSE 8000

# 读取 .env 里的配置
ENV PYTHONUNBUFFERED=1

# 启动 FastAPI 应用（如果用 Flask，请替换命令）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
