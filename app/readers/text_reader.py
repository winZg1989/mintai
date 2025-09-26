import os
import logging

logger = logging.getLogger(__name__)

def read_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='gbk') as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取TXT文件失败: {e}")
            return ""
    except Exception as e:
        logger.error(f"读取TXT文件失败: {e}")
        return ""