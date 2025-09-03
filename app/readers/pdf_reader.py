from PyPDF2 import PdfReader


def read_pdf(path: str) -> str:
reader = PdfReader(path)
texts = []
for page in reader.pages:
try:
texts.append(page.extract_text() or "")
except Exception:
continue
return "\n".join(texts)