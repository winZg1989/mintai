import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any

class FAISSStore:
    def __init__(self, index_dir: str, dimension: int):
        self.index_dir = index_dir
        self.dimension = dimension  # 统一属性名
        os.makedirs(index_dir, exist_ok=True)
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.meta_path = os.path.join(index_dir, "meta.jsonl")
        self.metadata: List[Dict[str, Any]] = []  # 统一使用 metadata
        self.index = None  # 初始为空，由 _load() 初始化
        self._load()

    def _load(self):
        """加载FAISS索引和元数据"""
        try:
            if os.path.exists(self.index_path):
                print(f"Loading FAISS index from {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                # 加载元数据
                if os.path.exists(self.meta_path):
                    with open(self.meta_path, 'r', encoding='utf-8') as f:
                        self.metadata = [json.loads(line) for line in f]
                print(f"Index loaded with {self.index.ntotal} vectors")
            else:
                print("No existing index found, creating new index")
                self.index = faiss.IndexFlatIP(self.dimension)  # 使用正确的属性名
                self.metadata = []
                self._save()  # 保存空索引
        except Exception as e:
            print(f"Error loading index: {e}. Creating new index.")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            self._save()

    def _save(self):
        """保存索引和元数据"""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            for meta in self.metadata:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """添加向量和元数据"""
        if not hasattr(self, 'index') or self.index is None:
            raise RuntimeError("FAISS index not initialized.")
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadatas)
        self._save()

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        if not hasattr(self, 'index') or self.index is None:
            raise RuntimeError("FAISS index not initialized.")
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        query_vec = query_vec.astype(np.float32)
        D, I = self.index.search(query_vec, top_k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.metadata):  # 检查索引有效性
                hits.append({"score": float(score), **self.metadata[idx]})
        return hits