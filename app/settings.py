import os
from pydantic import BaseSettings


class Settings(BaseSettings):
APP_NAME: str = "RAG on Azure"


# 选择 LLM 提供商: deepseek 或 openai
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "deepseek")


# DeepSeek
DEEPSEEK_API_KEY: str | None = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat") # 或 deepseek-reasoner
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


# OpenAI（若使用 OpenAI，推荐性价比模型：gpt-4o-mini）
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


# 路径
DATA_DIR: str = os.getenv("DATA_DIR", "data")
UPLOAD_DIR: str = os.path.join(DATA_DIR, "uploads")
CORPUS_DIR: str = os.path.join(DATA_DIR, "corpus")
INDEX_DIR: str = os.path.join(DATA_DIR, "faiss_index")


# 嵌入模型
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", 384))


# 检索参数
TOP_K: int = int(os.getenv("TOP_K", 5))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 64))


# 其他
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", 20))


settings = Settings()