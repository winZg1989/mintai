# pdf_reader.py
from PyPDF2 import PdfReader
import os
import logging

logger = logging.getLogger(__name__)

def read_pdf(file_path: str) -> str:
    """普通PDF文本提取"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        logger.error(f"PDF读取失败: {e}")
        return ""

def read_pdf_with_fitz(file_path: str) -> str:
    """使用PyMuPDF提取文本（更好的文本提取）"""
    try:
        # 尝试导入fitz，如果不可用则回退
        import fitz
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except ImportError:
        logger.warning("PyMuPDF (fitz) 不可用，使用普通PDF读取")
        return read_pdf(file_path)
    except Exception as e:
        logger.error(f"PyMuPDF读取失败: {e}")
        # 回退到普通读取
        return read_pdf(file_path)

# 修改 pdf_reader.py 中的OCR函数，使其在没有poppler时优雅降级
def read_pdf_with_ocr(file_path: str) -> str:
    """使用OCR读取扫描件PDF"""
    try:
        # 检查poppler是否可用
        try:
            from pdf2image import convert_from_path
            # 测试poppler是否工作
            import subprocess
            result = subprocess.run(['which', 'pdftoppm'], capture_output=True)
            if result.returncode != 0:
                raise ImportError("poppler not available")
        except (ImportError, Exception):
            print("OCR不可用：poppler未安装")
            return read_pdf(file_path)
        
        import pytesseract
        
        text = ""
        print(f"开始OCR处理: {os.path.basename(file_path)}")
        
        images = convert_from_path(file_path, dpi=200)
        print(f"转换成功，共 {len(images)} 页")
        
        for i, image in enumerate(images):
            print(f"处理第 {i+1} 页...")
            page_text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            text += f"第 {i+1} 页:\n" + page_text + "\n\n"
            
        print(f"OCR完成，提取文本长度: {len(text)}")
        return text
        
    except Exception as e:
        print(f"OCR处理失败: {e}")
        return read_pdf(file_path)