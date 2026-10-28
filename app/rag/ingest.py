"""知识库入库脚本：markdown 切分 -> BGE-M3 编码 -> 写入 Milvus
用法：python -m app.rag.ingest [--docs data/knowledge]
"""
import os
import re
import glob
import argparse
from pymilvus import MilvusClient, DataType
from FlagEmbedding import BGEM3FlagModel
from config.settings import get_settings

settings = get_settings()
CHUNK_SIZE = 400      # 每片约400字符
CHUNK_OVERLAP = 50    # 重叠50字符，避免语义被切断


def split_markdown(text: str) -> list[str]:
    """按标题分段后再按长度滑窗切分"""
    # 先按 ## 标题切大块
    sections = re.split(r"\n(?=##?\s)", text)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= CHUNK_SIZE:
            chunks.append(sec)
        else:
            # 滑窗切分
            start = 0
            while start < len(sec):
                chunks.append(sec[start:start + CHUNK_SIZE])
                start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def doc_id(filename: str, chunk: str) -> str:
    """doc_id 用文档名（去.md后缀），同一文档的多个切片共享doc_id，
    评测集的expected_doc_id直接对得上，badcase归因也能定位到源文档"""
    return os.path.basename(filename).replace(".md", "")


def ensure_collection(client: MilvusClient):
    """建collection：dense向量字段 + 标量字段"""
    if client.has_collection(settings.milvus_collection):
        client.drop_collection(settings.milvus_collection)
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
    schema.add_field("title", DataType.VARCHAR, max_length=256)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1024)  # bge-m3 dense维度1024

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense", index_type="HNSW", metric_type="IP", params={"M": 16, "efConstruction": 200})
    client.create_collection(settings.milvus_collection, schema=schema, index_params=index_params)


def ingest(docs_dir: str):
    files = sorted(glob.glob(os.path.join(docs_dir, "*.md")))
    print(f"发现 {len(files)} 个文档")

    model = BGEM3FlagModel(settings.embedding_model, use_fp16=True)
    client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
    ensure_collection(client)

    all_chunks = []
    for f in files:
        title = os.path.basename(f).replace(".md", "")
        text = open(f, encoding="utf-8").read()
        chunks = split_markdown(text)
        embs = model.encode(chunks)["dense_vecs"]
        data = [{
            "doc_id": doc_id(title, c),
            "title": title,
            "text": c[:4000],
            "dense": embs[i].tolist(),
        } for i, c in enumerate(chunks)]
        client.insert(settings.milvus_collection, data)
        all_chunks.extend([{"doc_id": d["doc_id"], "text": d["text"]} for d in data])
        print(f"  {title}: {len(chunks)} 片")

    client.flush(settings.milvus_collection)

    # BM25 索引落一份json，服务启动时加载（小规模政策库没必要上ES）
    import json
    out = os.path.join("data", "bm25_chunks.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(all_chunks, fp, ensure_ascii=False)
    print(f"入库完成，共 {len(all_chunks)} 片，BM25语料写入 {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/knowledge")
    args = parser.parse_args()
    ingest(args.docs)
