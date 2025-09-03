import os
import tempfile
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import PyPDF2
import docx2txt
from sentence_transformers import SentenceTransformer
import faiss
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'data', 'uploads')
app.config['VECTOR_STORE'] = os.path.join(os.path.dirname(__file__), 'data', 'vectors')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 创建必要的目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VECTOR_STORE'], exist_ok=True)

# 初始化模型
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    DIMENSION = 384
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    model = None

# 全局变量存储向量和文档
vector_index = None
documents = []
document_metadata = {}

# API配置
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'pdf', 'docx', 'txt'}

def extract_text_from_pdf(file_path):
    """从PDF提取文本"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"PDF提取错误: {e}")
        return ""

def extract_text_from_docx(file_path):
    """从Word文档提取文本"""
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        print(f"DOCX提取错误: {e}")
        return ""

def extract_text_from_txt(file_path):
    """从文本文件提取文本"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"TXT提取错误: {e}")
        return ""

def split_text(text, chunk_size=500, overlap=50):
    """将文本分割成重叠的chunks"""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def initialize_faiss_index():
    """初始化FAISS索引"""
    global vector_index
    if vector_index is None:
        vector_index = faiss.IndexFlatL2(DIMENSION)
    return vector_index

def add_to_vector_store(text_chunks, filename):
    """添加文本到向量存储"""
    global vector_index, documents, document_metadata
    
    if not text_chunks:
        return False
    
    # 生成嵌入向量
    embeddings = model.encode(text_chunks)
    
    # 初始化或更新FAISS索引
    vector_index = initialize_faiss_index()
    vector_index.add(embeddings)
    
    # 存储文档信息
    start_idx = len(documents)
    for i, chunk in enumerate(text_chunks):
        doc_id = f"{filename}_{start_idx + i}"
        documents.append({
            'id': doc_id,
            'text': chunk,
            'source': filename,
            'chunk_index': i
        })
    
    # 更新文档元数据
    if filename not in document_metadata:
        document_metadata[filename] = {
            'upload_time': datetime.now().isoformat(),
            'chunk_count': len(text_chunks),
            'total_chars': sum(len(chunk) for chunk in text_chunks)
        }
    
    return True

def query_vector_store(query, top_k=5):
    """查询向量数据库"""
    global vector_index, documents
    
    if vector_index is None or len(documents) == 0:
        return []
    
    # 将查询转换为向量
    query_embedding = model.encode([query])
    
    # 搜索最相似的文档
    distances, indices = vector_index.search(query_embedding, top_k)
    
    # 获取相关文档内容
    results = []
    for idx in indices[0]:
        if idx < len(documents):
            results.append({
                'text': documents[idx]['text'],
                'source': documents[idx]['source'],
                'score': float(distances[0][idx]) if distances[0][idx] < float('inf') else 0.0
            })
    
    return results

def generate_answer_with_deepseek(query, context):
    """使用DeepSeek API生成答案"""
    if not DEEPSEEK_API_KEY:
        return "错误：未配置DeepSeek API密钥"
    
    # 构建提示
    prompt = f"""基于以下上下文信息，请回答问题。如果上下文没有提供足够信息，请如实告知。

上下文：
{context}

问题：{query}

请提供准确、简洁的回答："""
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个专业的文档分析助手，基于提供的上下文信息回答问题。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"调用DeepSeek API时出错：{str(e)}"

def generate_answer_with_openai(query, context):
    """使用OpenAI API生成答案"""
    if not OPENAI_API_KEY:
        return "错误：未配置OpenAI API密钥"
    
    prompt = f"""基于以下上下文信息回答问题：

{context}

问题：{query}

答案："""
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"调用OpenAI API时出错：{str(e)}"

def generate_answer(query, context):
    """生成答案（优先使用DeepSeek，备用OpenAI）"""
    if DEEPSEEK_API_KEY:
        return generate_answer_with_deepseek(query, context)
    elif OPENAI_API_KEY:
        return generate_answer_with_openai(query, context)
    else:
        return "错误：未配置任何API密钥，请设置DEEPSEEK_API_KEY或OPENAI_API_KEY环境变量"

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """文件上传接口"""
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 提取文本内容
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        else:
            text = extract_text_from_txt(file_path)
        
        if not text.strip():
            os.remove(file_path)
            return jsonify({'error': '文件内容为空或无法提取文本'})
        
        # 分割文本并添加到向量存储
        chunks = split_text(text)
        success = add_to_vector_store(chunks, filename)
        
        if success:
            return jsonify({
                'message': f'文件 {filename} 上传和处理成功',
                'chunks': len(chunks),
                'characters': len(text)
            })
        else:
            return jsonify({'error': '文件处理失败'})
    
    return jsonify({'error': '不支持的文件类型，请上传PDF、DOCX或TXT文件'})

@app.route('/ask', methods=['POST'])
def ask_question():
    """问答接口"""
    data = request.get_json()
    query = data.get('question', '').strip()
    
    if not query:
        return jsonify({'error': '问题不能为空'})
    
    # 获取相关文档片段
    relevant_docs = query_vector_store(query, top_k=5)
    
    if not relevant_docs:
        return jsonify({'error': '没有找到相关文档信息，请先上传文档'})
    
    # 构建上下文
    context = "\n\n".join([f"[来源: {doc['source']}]\n{doc['text']}" for doc in relevant_docs])
    
    # 生成答案
    answer = generate_answer(query, context)
    
    # 返回答案和参考来源
    sources = list(set([doc['source'] for doc in relevant_docs]))
    
    return jsonify({
        'answer': answer,
        'sources': sources,
        'relevant_docs': relevant_docs
    })

@app.route('/documents')
def list_documents():
    """获取已上传文档列表"""
    return jsonify({
        'documents': document_metadata,
        'total_chunks': len(documents)
    })

@app.route('/clear', methods=['POST'])
def clear_documents():
    """清空所有文档"""
    global vector_index, documents, document_metadata
    vector_index = None
    documents = []
    document_metadata = {}
    
    # 清空上传文件
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    return jsonify({'message': '所有文档已清空'})

@app.route('/health')
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'documents_count': len(documents),
        'vector_index_ready': vector_index is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', 'False').lower() == 'true')