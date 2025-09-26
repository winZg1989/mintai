import docx2txt
import os
import logging

logger = logging.getLogger(__name__)

def read_docx(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    
    try:
        return docx2txt.process(path) or ""
    except Exception as e:
        logger.error(f"读取DOCX文件失败: {e}")
        return ""
