import os
import httpx
import logging  # 添加导入
from typing import List
from app.settings import settings

logger = logging.getLogger(__name__)  # 添加logger

TIMEOUT = 60

async def chat(prompt: str, context_chunks: List[str]) -> str:
    try:
        system = (
            "你是一个企业内部知识库问答助手。请严格依据给定上下文回答，"
            "若无法从上下文得到答案，请明确说明‘无法从知识库找到’，并给出下一步建议。"
        )
        context = "\n\n".join(f"[片段{i+1}] {c}" for i, c in enumerate(context_chunks))
        user = f"以下是知识库片段，请结合回答：\n\n{context}\n\n问题：{prompt}"

        if settings.LLM_PROVIDER == "deepseek":
            url = f"{settings.DEEPSEEK_BASE_URL}/chat/completions"
            headers = {"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"}
            payload = {
                "model": settings.DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
            }
        else: # openai
            url = f"{settings.OPENAI_BASE_URL}/chat/completions"
            headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
            payload = {
                "model": settings.OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
            }

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM API调用失败: {e}")
        return "抱歉，服务暂时不可用，请稍后重试。"        