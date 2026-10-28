"""CommerceFlow FastAPI 入口
启动：uvicorn app.main:app --reload --port 8000
"""
import json
import os
from fastapi import FastAPI
from app.api.chat import router as chat_router
from app.rag.retriever import load_bm25_index

app = FastAPI(
    title="CommerceFlow 电商售后智能服务系统",
    description="确定性 Workflow + Agent 决策的智能售后系统（个人学习项目，数据均为模拟）",
    version="0.3.0",
)


@app.on_event("startup")
def startup():
    """启动时加载BM25内存索引（Milvus向量检索按需懒加载）"""
    path = os.path.join("data", "bm25_chunks.json")
    if os.path.exists(path):
        chunks = json.load(open(path, encoding="utf-8"))
        load_bm25_index(chunks)
        print(f"BM25索引加载完成: {len(chunks)} 片")
    else:
        print("警告: 未找到 data/bm25_chunks.json，请先运行 python -m app.rag.ingest")


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(chat_router, prefix="/api")
