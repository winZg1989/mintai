# embeddings.py
import gc
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

_model = None  

def get_model(model_name: str):
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(model_name, device="cpu")
            _model.max_seq_length = 512
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
    return _model

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, 384)  # 返回空的2D数组
    
    try:
        model = get_model(model_name)
        batch_size = 16
        all_vectors = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vectors = model.encode(
                batch, 
                convert_to_numpy=True, 
                show_progress_bar=False, 
                normalize_embeddings=True
            )
            all_vectors.append(vectors)
        
        # 合并所有批次的向量
        if all_vectors:
            # 使用 np.vstack 将列表中的多个数组垂直堆叠成一个2D数组
            result = np.vstack(all_vectors)
        else:
            # 如果没有向量，返回空的2D数组
            result = np.array([], dtype=np.float32).reshape(0, 384)
            
        # 确保数据类型正确
        if result.dtype != np.float32:
            result = result.astype(np.float32)
            
        gc.collect()
        return result
        
    except Exception as e:
        print(f"嵌入生成失败: {e}")
        # 返回备选嵌入
        return generate_fallback_embeddings(texts)

def generate_fallback_embeddings(texts: List[str]) -> np.ndarray:
    """生成简单的备选嵌入"""
    import hashlib
    dim = 384  # 假设维度是384
    
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, dim)
    
    embeddings = []
    for text in texts:
        # 使用哈希生成简单的确定性向量
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # 从哈希值生成向量
        vector = []
        for i in range(dim):
            byte_idx = i % len(hash_bytes)
            vector.append((hash_bytes[byte_idx] / 255.0) * 2 - 1)  # 归一化到 [-1, 1]
        
        embeddings.append(vector)
    
    return np.array(embeddings, dtype=np.float32)