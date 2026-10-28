"""混合检索：BGE-M3 向量召回 + BM25 关键词检索 + RRF 融合 + bge-reranker 精排"""
import jieba
from rank_bm25 import BM25Okapi
from pymilvus import MilvusClient
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from config.settings import get_settings

settings = get_settings()

# 懒加载：模型比较大，第一次调用时才初始化
_bge_model = None
_reranker = None
_milvus = None
_bm25 = None
_bm25_docs = []   # BM25语料（本地小规模政策库直接放内存）
_bm25_meta = []


def _get_bge():
    global _bge_model
    if _bge_model is None:
        _bge_model = BGEM3FlagModel(settings.embedding_model, use_fp16=True)
    return _bge_model


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(settings.reranker_model, use_fp16=True)
    return _reranker


def _get_milvus():
    global _milvus
    if _milvus is None:
        _milvus = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
    return _milvus


def load_bm25_index(chunks: list[dict]):
    """加载BM25内存索引，chunks: [{doc_id, text}]，入库时调用"""
    global _bm25, _bm25_docs, _bm25_meta
    _bm25_meta = chunks
    _bm25_docs = [list(jieba.cut(c["text"])) for c in chunks]
    _bm25 = BM25Okapi(_bm25_docs)


def _dense_search(query: str, top_k: int) -> list[dict]:
    """BGE-M3 向量召回，走Milvus"""
    emb = _get_bge().encode([query])["dense_vecs"][0].tolist()
    client = _get_milvus()
    res = client.search(
        collection_name=settings.milvus_collection,
        data=[emb],
        limit=top_k,
        output_fields=["doc_id", "text"],
    )
    return [{"doc_id": h["entity"]["doc_id"], "text": h["entity"]["text"], "score": h["distance"]} for h in res[0]]


def _sparse_search(query: str, top_k: int) -> list[dict]:
    """BM25 关键词召回"""
    if _bm25 is None:
        return []
    tokens = list(jieba.cut(query))
    scores = _bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"doc_id": _bm25_meta[i]["doc_id"], "text": _bm25_meta[i]["text"], "score": float(scores[i])}
            for i in top_idx if scores[i] > 0]


def rrf_fuse(rank_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """RRF倒数排名融合：score = sum(1/(k+rank))，抹平两路分数量纲差异"""
    fused = {}
    for lst in rank_lists:
        for rank, item in enumerate(lst):
            key = item["doc_id"] + "|" + item["text"][:50]
            if key not in fused:
                fused[key] = {**item, "rrf_score": 0.0}
            fused[key]["rrf_score"] += 1.0 / (k + rank + 1)
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


def hybrid_search(query: str, top_k: int = None, top_n: int = None, use_reranker: bool = True) -> list[dict]:
    """完整混合检索链路：dense+sparse -> RRF -> rerank"""
    top_k = top_k or settings.retrieval_top_k
    top_n = top_n or settings.rerank_top_n

    dense = _dense_search(query, top_k)
    sparse = _sparse_search(query, top_k)
    candidates = rrf_fuse([dense, sparse])[:top_k]

    if not use_reranker or not candidates:
        return [{"doc_id": c["doc_id"], "text": c["text"], "score": c["rrf_score"]} for c in candidates[:top_n]]

    # bge-reranker cross-encoder 精排
    pairs = [[query, c["text"]] for c in candidates]
    scores = _get_reranker().compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    for c, s in zip(candidates, scores):
        c["score"] = float(s)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return [{"doc_id": c["doc_id"], "text": c["text"], "score": c["score"]} for c in candidates[:top_n]]
