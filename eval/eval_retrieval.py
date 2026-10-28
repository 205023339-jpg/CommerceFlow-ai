"""RAG 检索评测：Recall@K、MRR
用法：python -m eval.eval_retrieval [--dataset eval/eval_dataset.json]
只评测 answerable=true 的题（完整300题评测集含60道不可作答负例，见README说明）
"""
import json
import argparse
from app.rag.retriever import hybrid_search


def evaluate(dataset_path: str, ks=(1, 3, 5), use_reranker: bool = True):
    data = json.load(open(dataset_path, encoding="utf-8"))
    answerable = [d for d in data if d.get("answerable", True)]
    print(f"评测集: {len(data)} 题，可作答 {len(answerable)} 题，reranker={'on' if use_reranker else 'off'}")

    recall_hits = {k: 0 for k in ks}
    rr_sum = 0.0

    for item in answerable:
        q, gold = item["question"], item["expected_doc_id"]
        results = hybrid_search(q, top_k=max(ks), top_n=max(ks), use_reranker=use_reranker)
        got_ids = [r["doc_id"] for r in results]

        for k in ks:
            if gold in got_ids[:k]:
                recall_hits[k] += 1

        # MRR：第一个命中gold的排名倒数
        if gold in got_ids:
            rr_sum += 1.0 / (got_ids.index(gold) + 1)

    n = len(answerable)
    metrics = {f"Recall@{k}": round(recall_hits[k] / n, 4) for k in ks}
    metrics["MRR"] = round(rr_sum / n, 4)

    print("=" * 40)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/eval_dataset.json")
    parser.add_argument("--no-reranker", action="store_true", help="消融：关闭精排")
    args = parser.parse_args()
    evaluate(args.dataset, use_reranker=not args.no_reranker)
