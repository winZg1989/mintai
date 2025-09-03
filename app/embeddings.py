from sentence_transformers import SentenceTransformer
from typing import List


_model = None


def get_model(model_name: str):
global _model
if _model is None:
_model = SentenceTransformer(model_name, device="cpu")
return _model




def embed_texts(texts: List[str], model_name: str) -> list:
model = get_model(model_name)
vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
return vectors