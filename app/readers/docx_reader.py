import docx2txt


def read_docx(path: str) -> str:
return docx2txt.process(path) or ""