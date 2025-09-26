import os
import logging
from typing import Optional
from app.readers.pdf_reader import read_pdf
from app.readers.docx_reader import read_docx
from app.readers.text_reader import read_text

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    '.pdf': read_pdf,
    '.docx': read_docx,
    '.doc': read_docx,  # 处理.doc文件
    '.txt': read_text,
    '.text': read_text,
}

def read_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        from app.readers.pdf_reader import read_pdf, read_pdf_with_fitz
        
        # 先尝试更好的文本提取
        text = read_pdf_with_fitz(file_path)
        print(f"PyMuPDF读取文本长度: {len(text)}")
        
        # 如果文本仍然很少，尝试OCR（如果可用）
        if len(text.strip()) < 500:
            print("文本较少，尝试OCR...")
            try:
                from app.readers.pdf_reader import read_pdf_with_ocr
                text = read_pdf_with_ocr(file_path)
            except Exception as e:
                print(f"OCR不可用: {e}")
        
        return text
    
    # 其他文件类型的处理
    reader_func = SUPPORTED_EXTENSIONS.get(ext)
    if reader_func:
        return reader_func(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

def get_supported_extensions() -> list:
    """获取支持的文件扩展名列表"""
    return list(SUPPORTED_EXTENSIONS.keys())