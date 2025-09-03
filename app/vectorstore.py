import os, json
import faiss
import numpy as np
from typing import List, Dict, Any


class FAISSStore:
def __init__(self, index_dir: str, dim: int):
self.index_dir = index_dir
self.dim = dim
os.makedirs(index_dir, exist_ok=True)
self.index_path = os.path.join(index_dir, "index.faiss")
self.meta_path = os.path.join(index_dir, "meta.jsonl")
self.meta: List[Dict[str, Any]] = []
self.index = faiss.IndexFlatIP(dim) # 余弦相似度需先归一化
self._load()


def _load(self):
if os.path.exists(self.index_path):
self.index = faiss.read_index(self.index_path)
if os.path.exists(self.meta_path):
with open(self.meta_path, "r", encoding="utf-8") as f:
self.meta = [json.loads(line) for line in f]


def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
if embeddings.dtype != np.float32:
embeddings = embeddings.astype("float32")
self.index.add(embeddings)
with open(self.meta_path, "a", encoding="utf-8") as f:
for m in metadatas:
f.write(json.dumps(m, ensure_ascii=False) + "\n")
self.meta.extend(metadatas)
faiss.write_index(self.index, self.index_path)


def search(self, query_vec: np.ndarray, top_k: int = 5):
if query_vec.ndim == 1:
query_vec = query_vec.reshape(1, -1)
D, I = self.index.search(query_vec.astype("float32"), top_k)
hits = []
for score, idx in zip(D[0], I[0]):
if idx < 0 or idx >= len(self.meta):
continue
hits.append({"score": float(score), **self.meta[idx]})
return hits