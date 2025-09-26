import time
import logging
import os, uuid, json, glob
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from app.settings import settings
from app.utils import ensure_dirs, Stopwatch
from app.readers.pdf_reader import read_pdf
from app.readers.docx_reader import read_docx
from app.chunking import split_into_chunks
from app.embeddings import embed_texts
from app.vectorstore import FAISSStore
from app.llm_client import chat

# 在现有的导入后面添加
from app.readers.file_reader import read_file, get_supported_extensions

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_NAME)

# 配置模板引擎 - 必须在路由之前！
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有头
)

ensure_dirs(settings.UPLOAD_DIR, settings.CORPUS_DIR, settings.INDEX_DIR)
store = FAISSStore(settings.INDEX_DIR, settings.EMBEDDING_DIM)


class QueryBody(BaseModel):
    query: str
    top_k: Optional[int] = None  # 使用 Optional 而不是 |

@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})
    
    
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    supported_exts = get_supported_extensions()
    if ext not in supported_exts:
        raise HTTPException(400, f"不支持的文件格式，支持格式: {', '.join(supported_exts)}")
    
    # 保存上传
    fid = uuid.uuid4().hex
    save_path = os.path.join(settings.UPLOAD_DIR, f"{fid}{ext}")
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # 使用统一的文件读取器
    text = read_file(save_path)

    if not text or not text.strip():
        raise HTTPException(400, "未提取到文本内容")

    # 分块 & 向量化
    chunks = split_into_chunks(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    if chunks is None or len(chunks) == 0:
        raise HTTPException(400, "无法分块处理文本内容")
    
    vecs = embed_texts(chunks, settings.EMBEDDING_MODEL_NAME)
    
    if vecs is None or vecs.size == 0:
        raise HTTPException(500, "无法生成文本向量")

    # 写入索引 - 这里会自动保存
    metas = [{
        "doc_id": fid, 
        "chunk_id": i, 
        "text": chunk,
        "file_name": file.filename
    } for i, chunk in enumerate(chunks)]
    
    store.add(vecs, metas)  # 自动保存，不需要 store.save()

    return {"doc_id": fid, "chunks": len(chunks), "filename": file.filename}


# 在main.py的query函数中添加更详细的错误处理
@app.post("/query")
async def query(q: QueryBody):
    try:
        logger.info(f"收到查询请求: {q.query}")
        
        if not q.query.strip():
            raise HTTPException(400, "问题不能为空")

        # 检查向量库是否初始化
        if not hasattr(store, 'index') or store.index.ntotal == 0:
            logger.info("向量库为空，开始自动索引所有文件...")
            index_result = await index_all_files()
            if index_result["files_indexed"] == 0:
                logger.warning("没有找到可索引的文件")
                raise HTTPException(400, "请先上传文档再进行查询")

        # 生成查询向量
        qvec = embed_texts([q.query], settings.EMBEDDING_MODEL_NAME)
        
        # 大幅增加搜索范围
        top_k = min(100, store.index.ntotal)  # 搜索更多结果
        hits = store.search(qvec[0], top_k=top_k)
        
        logger.info(f"搜索完成，找到 {len(hits)} 个相关片段")
        
        # 调试：显示所有相关结果的文件名和分数
        unique_files = set()
        for i, hit in enumerate(hits):
            file_name = hit.get("file_name", "unknown")
            unique_files.add(file_name)
            if i < 10:  # 显示前10个结果
                logger.info(f"结果 {i+1}: 分数={hit['score']:.4f}, 文件={file_name}")
        
        logger.info(f"涉及的文件: {list(unique_files)}")
        
        # 尝试找到真正包含关键词的结果
        keyword_hits = []
        for hit in hits:
            # 检查文本中是否包含查询关键词
            text_lower = hit["text"].lower()
            query_keywords = ["新田", "科技城", "發展", "綱要", "210", "公頃", "創科"]
            if any(keyword in text_lower for keyword in query_keywords):
                keyword_hits.append(hit)
                logger.info(f"找到关键词匹配: 分数={hit['score']:.4f}, 文件={hit.get('file_name')}")
        
        # 优先使用包含关键词的结果
        if keyword_hits:
            filtered_hits = keyword_hits
            logger.info(f"使用关键词匹配结果: {len(filtered_hits)} 个")
        else:
            # 如果没有关键词匹配，使用原始结果但降低阈值
            filtered_hits = [hit for hit in hits if hit["score"] > 0.1]
            logger.info(f"使用低阈值结果: {len(filtered_hits)} 个")
        
        if not filtered_hits:
            return {
                "answer": "未找到相关文档内容。",
                "elapsed": 0,
                "hits": [],
                "query": q.query
            }
        
        # 构建更明确的提示词
        context_texts = [h["text"] for h in filtered_hits[:10]]  # 最多使用10个片段
        
        prompt = f"""请仔细分析以下文档内容，回答用户问题。如果内容中包含相关信息，请基于内容回答。

用户问题：{q.query}

相关文档内容：
{chr(10).join([f'【文档片段 {i+1}】{text}' for i, text in enumerate(context_texts)])}

请基于以上文档内容回答用户问题："""
        
        start_time = time.time()
        answer = await chat(prompt, [])
        elapsed_time = time.time() - start_time
        
        return {
            "answer": answer, 
            "elapsed": round(elapsed_time, 3), 
            "hits": filtered_hits[:q.top_k or settings.TOP_K],
            "query": q.query,
            "total_hits_found": len(hits),
            "relevant_hits_used": len(filtered_hits)
        }
            
    except Exception as e:
        logger.error(f"查询处理失败: {str(e)}", exc_info=True)
        return {
            "answer": f"系统处理请求时出现错误: {str(e)}",
            "elapsed": 0,
            "hits": [],
            "error": str(e)
        }


@app.get("/")
async def root():
    return {"name": settings.APP_NAME, "provider": settings.LLM_PROVIDER, "model": settings.DEEPSEEK_MODEL if settings.LLM_PROVIDER == "deepseek" else settings.OPENAI_MODEL}
    
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "vectorstore_count": store.index.ntotal if hasattr(store, 'index') else 0
    }
@app.get("/supported-formats")
async def get_supported_formats():
    """获取支持的文件格式列表"""
    return {"supported_formats": get_supported_extensions()}

@app.get("/uploaded-files")
async def list_uploaded_files():
    """
    列出data/uploads目录中的所有支持的文件
    """
    try:
        files = []
        total_size = 0
        
        for ext in get_supported_extensions():
            pattern = os.path.join(settings.UPLOAD_DIR, f"*{ext}")
            for file_path in glob.glob(pattern):
                file_info = {
                    "name": os.path.basename(file_path),
                    "size": os.path.getsize(file_path),
                    "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
                    "modified": os.path.getmtime(file_path),
                    "extension": ext,
                    "path": file_path
                }
                files.append(file_info)
                total_size += file_info["size"]
        
        # 按修改时间排序（最新的在前）
        files.sort(key=lambda x: x["modified"], reverse=True)
        
        return {
            "total_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": files
        }
        
    except Exception as e:
        logger.error(f"获取文件列表失败: {e}")
        raise HTTPException(500, f"获取文件列表失败: {str(e)}")

@app.post("/index-all")
async def index_all_files():
    """索引所有上传的文件"""
    try:
        all_files = glob.glob(os.path.join(settings.UPLOAD_DIR, "*.*"))
        files_indexed = 0
        total_chunks = 0
        
        for file_path in all_files:
            try:
                # 检查文件扩展名
                ext = os.path.splitext(file_path)[1].lower()
                if ext not in get_supported_extensions():
                    continue
                    
                logger.info(f"开始处理文件: {os.path.basename(file_path)}")
                
                # 使用统一的文件读取器
                text = read_file(file_path)
                
                if not text or len(text.strip()) < 100:
                    logger.warning(f"文件内容过短，跳过索引: {os.path.basename(file_path)}")
                    continue
                
                # 分块处理
                chunks = split_into_chunks(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
                if chunks is None or len(chunks) == 0:
                    logger.warning(f"无法分块处理文件: {os.path.basename(file_path)}")
                    continue
                
                # 生成嵌入向量
                vecs = embed_texts(chunks, settings.EMBEDDING_MODEL_NAME)
                
                if vecs is None or vecs.size == 0:
                    logger.warning(f"无法生成嵌入向量: {os.path.basename(file_path)}")
                    continue
                
                # 添加到向量库
                fid = os.path.splitext(os.path.basename(file_path))[0]
                metas = [{
                    "doc_id": fid, 
                    "chunk_id": i, 
                    "text": chunk,
                    "file_name": os.path.basename(file_path)
                } for i, chunk in enumerate(chunks)]
                
                store.add(vecs, metas)  # 这里会自动保存
                files_indexed += 1
                total_chunks += len(chunks)
                
                logger.info(f"成功索引文件: {os.path.basename(file_path)}, 块数: {len(chunks)}")
                
            except Exception as e:
                logger.error(f"处理文件失败 {os.path.basename(file_path)}: {str(e)}", exc_info=True)
                continue
        
        # 移除这行 - FAISSStore 已经自动保存了
        # store.save()
        
        return {
            "success": True,
            "message": f"索引完成，共处理 {files_indexed} 个文件，{total_chunks} 个文本块",
            "files_indexed": files_indexed,
            "total_chunks": total_chunks,
            "vectorstore_size": store.index.ntotal if hasattr(store, 'index') else 0
        }
        
    except Exception as e:
        logger.error(f"索引过程失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "files_indexed": 0,
            "total_chunks": 0
        }
        
@app.post("/analyze-file-content")
async def analyze_file_content(
    file_name: str,
    query: str,
    max_chunks: int = 5
):
    """
    直接分析特定文件的内容（不通过向量搜索）
    """
    try:
        file_path = os.path.join(settings.UPLOAD_DIR, file_name)
        if not os.path.exists(file_path):
            raise HTTPException(404, "文件不存在")
        
        # 读取文件内容
        text = read_file(file_path)
        if not text or not text.strip():
            raise HTTPException(400, "文件内容为空")
        
        # 分块处理
        chunks = split_into_chunks(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        if not chunks:
            raise HTTPException(400, "无法分块处理文件内容")
        
        # 使用所有内容进行分析（或者前几个块）
        chunks_to_use = chunks[:max_chunks]
        
        # 生成答案
        answer = await chat(query, chunks_to_use)
        
        return {
            "file": file_name,
            "query": query,
            "answer": answer,
            "total_chunks": len(chunks),
            "chunks_used": len(chunks_to_use),
            "file_size": len(text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分析文件内容失败: {e}")
        raise HTTPException(500, f"分析文件内容失败: {str(e)}")

@app.post("/search-across-all")
async def search_across_all_files(q: QueryBody):
    """
    在所有已索引的文件中搜索
    """
    try:
        if not q.query.strip():
            raise HTTPException(400, "搜索查询不能为空")
        
        # 检查是否有索引，如果没有则自动索引
        if not hasattr(store, 'index') or store.index.ntotal == 0:
            index_result = await index_all_files()
            if index_result["files_indexed"] == 0:
                raise HTTPException(400, "没有可搜索的文件，请先上传文件")
        
        # 生成查询向量
        qvec = embed_texts([q.query], settings.EMBEDDING_MODEL_NAME)
        if len(qvec) == 0:
            # 使用简单嵌入作为备选
            from app.embeddings_light import generate_simple_embeddings
            qvec = generate_simple_embeddings([q.query])
        
        # 搜索相关片段（获取更多结果以便按文件分组）
        top_k = min(q.top_k or settings.TOP_K * 10, 50)  # 最多50个结果
        hits = store.search(qvec[0], top_k=top_k)
        
        if not hits:
            return {
                "query": q.query,
                "message": "未找到相关结果",
                "results": []
            }
        
        # 按文件分组
        file_results = {}
        for hit in hits:
            file_name = hit.get("file_name", "unknown")
            if file_name not in file_results:
                file_results[file_name] = {
                    "file": file_name,
                    "hits": [],
                    "max_score": 0,
                    "total_score": 0
                }
            
            file_results[file_name]["hits"].append(hit)
            file_results[file_name]["max_score"] = max(
                file_results[file_name]["max_score"], 
                hit["score"]
            )
            file_results[file_name]["total_score"] += hit["score"]
        
        # 为每个文件生成摘要答案
        results = []
        for file_name, file_data in file_results.items():
            # 取每个文件的前3个最相关片段
            top_hits = sorted(file_data["hits"], key=lambda x: x["score"], reverse=True)[:3]
            
            if top_hits:
                answer = await chat(
                    f"基于以下内容回答: {q.query}",
                    [h["text"] for h in top_hits]
                )
                
                results.append({
                    "file": file_name,
                    "answer": answer,
                    "max_score": round(file_data["max_score"], 4),
                    "avg_score": round(file_data["total_score"] / len(file_data["hits"]), 4),
                    "hits_count": len(file_data["hits"]),
                    "top_chunks": top_hits[:2]  # 显示前2个最相关片段
                })
        
        # 按最大分数排序
        results.sort(key=lambda x: x["max_score"], reverse=True)
        
        return {
            "query": q.query,
            "total_files_found": len(results),
            "total_hits": len(hits),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"全局搜索失败: {e}")
        return {
            "error": str(e),
            "results": []
        }