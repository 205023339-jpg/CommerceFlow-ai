"""意图分类推理封装：优先ONNX，回退PyTorch；低置信度兜底"其他咨询"
被 app/agent/nodes.py 调用
"""
import os
import json
import numpy as np

from classifier.labels import LABELS, NUM_LABELS

CKPT_DIR = os.path.join("classifier", "checkpoints")
ONNX_PATH = os.path.join(CKPT_DIR, "intent.onnx")
FALLBACK_LABEL = "其他咨询"
DEFAULT_THRESHOLD = 0.75  # 没做校准时用的保守默认值

_tokenizer = None
_session = None        # onnxruntime session
_torch_model = None
_threshold = DEFAULT_THRESHOLD


def _load():
    global _tokenizer, _session, _torch_model, _threshold
    if _tokenizer is not None:
        return

    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR if os.path.exists(CKPT_DIR) else "hfl/chinese-roberta-wwm-ext")

    # 加载校准阈值
    calib_path = os.path.join(CKPT_DIR, "calibration.json")
    if os.path.exists(calib_path):
        _threshold = json.load(open(calib_path, encoding="utf-8")).get("threshold", DEFAULT_THRESHOLD)

    if os.path.exists(ONNX_PATH):
        import onnxruntime as ort
        _session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    else:
        # 回退到PyTorch模型（导出ONNX前的开发阶段用）
        from transformers import AutoModelForSequenceClassification
        import torch
        _torch_model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR, num_labels=NUM_LABELS)
        _torch_model.eval()


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def classify_intent(text: str) -> tuple[str, float]:
    """返回 (意图标签, 置信度)。置信度低于阈值时兜底到"其他咨询"，
    后续由工作流决定是否转人工，避免低置信度误分类带偏整条链路。
    """
    _load()
    enc = _tokenizer(text, max_length=64, truncation=True, return_tensors="np")

    if _session is not None:
        logits = _session.run(None, {k: v.astype(np.int64) for k, v in enc.items()})[0][0]
    else:
        import torch
        with torch.no_grad():
            logits = _torch_model(**{k: torch.from_numpy(v) for k, v in enc.items()}).logits[0].numpy()

    probs = _softmax(logits)
    idx = int(probs.argmax())
    conf = float(probs[idx])

    if conf < _threshold:
        return FALLBACK_LABEL, conf
    return LABELS[idx], conf


if __name__ == "__main__":
    # 简单自测
    for t in ["我的快递到哪了", "这个东西不想要了怎么退款", "你好呀", "发票怎么开"]:
        print(t, "->", classify_intent(t))
