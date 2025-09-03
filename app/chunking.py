import re
from typing import List


SENT_SPLIT = re.compile(r"(?<=[。！？!?])\s+")


# 基本清洗与分段


def normalize(text: str) -> str:
text = text.replace("\r", "\n")
text = re.sub(r"\n{2,}", "\n\n", text)
return text.strip()




def split_into_chunks(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
text = normalize(text)
# 先按段落再拼接
paras = [p.strip() for p in text.split("\n\n") if p.strip()]
chunks, buf = [], []
cur = 0
for p in paras:
sents = SENT_SPLIT.split(p)
for s in sents:
s = s.strip()
if not s:
continue
if cur + len(s) + 1 <= chunk_size:
buf.append(s); cur += len(s) + 1
else:
if buf:
chunks.append(" ".join(buf))
# 处理重叠
merged = " ".join(buf)
tail = merged[-overlap:] if overlap > 0 and len(merged) > overlap else ""
buf = [tail, s] if tail else [s]
cur = sum(len(x) for x in buf) + len(buf) - 1
if buf:
chunks.append(" ".join(x for x in buf if x))
# 去重
uniq = []
seen = set()
for c in chunks:
k = hash(c)
if k not in seen:
seen.add(k); uniq.append(c)
return uniq