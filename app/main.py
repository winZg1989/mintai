import os, uuid, json
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.settings import settings
from app.utils import ensure_dirs, Stopwatch
from app.readers.pdf_reader import read_pdf
from app.readers.docx_reader import read_docx
from app.chunking import split_into_chunks
from app.embeddings import embed_texts
from app.vectorstore import FAISSStore
from app.llm_client import chat


app = FastAPI(title=settings.APP_NAME)
app.add_middleware(
CORSMiddleware,
allow_origins=["*"], allow_credentials=True,
allow_methods=["*"], allow_headers=["*"],
)


ensure_dirs(settings.UPLOAD_DIR, settings.CORPUS_DIR, settings.INDEX_DIR)
store = FAISSStore(settings.INDEX_DIR, settings.EMBEDDING_DIM)


class QueryBody(BaseModel):
query: str
top_k: int | None = None


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
ext = os.path.splitext(file.filename)[1].lower()
if ext not in [".pdf", ".docx"]:
raise HTTPException(400, "仅支持 .pdf / .docx")
# 保存上传
fid = uuid.uuid4().hex
save_path = os.path.join(settings.UPLOAD_DIR, f"{fid}{ext}")
with open(save_path, "wb") as f:
f.write(await file.read())


# 读取文本
if ext == ".pdf":
text = read_pdf(save_path)
else:
text = read_docx(save_path)


if not text.strip():
raise HTTPException(400, "未提取到文本内容")


# 分块 & 向量化
chunks = split_into_chunks(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
vecs = embed_texts(chunks, settings.EMBEDDING_MODEL_NAME)


# 写入索引
metas = [{"doc_id": fid, "chunk_id": i, "text": c} for i, c in enumerate(chunks)]
store.add(vecs, metas)


return {"doc_id": fid, "chunks": len(chunks)}


@app.post("/query")
async def query(q: QueryBody):
if not q.query.strip():
raise HTTPException(400, "问题不能为空")


# 查询向量
qvec = embed_texts([q.query], settings.EMBEDDING_MODEL_NAME)
hits = store.search(qvec[0], top_k=q.top_k or settings.TOP_K)
ctx = [h["text"] for h in hits]


with Stopwatch() as sw:
answer = await chat(q.query, ctx)
return {"answer": answer, "elapsed": round(sw.elapsed, 3), "hits": hits}


@app.get("/")
async def root():
return {"name": settings.APP_NAME, "provider": settings.LLM_PROVIDER, "model": settings.D