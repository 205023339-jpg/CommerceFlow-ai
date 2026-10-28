"""导出ONNX：把微调后的RoBERTa分类头导出，CPU推理延迟从~80ms降到~25ms
用法：python -m classifier.export_onnx
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from classifier.labels import NUM_LABELS

CKPT_DIR = os.path.join("classifier", "checkpoints")
ONNX_PATH = os.path.join(CKPT_DIR, "intent.onnx")


def export():
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR, num_labels=NUM_LABELS)
    model.eval()

    # 构造dummy输入
    dummy = tokenizer("测试输入", max_length=64, truncation=True, return_tensors="pt")

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        ONNX_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"导出完成: {ONNX_PATH} ({os.path.getsize(ONNX_PATH) / 1e6:.1f} MB)")

    # 简单校验ONNX输出与原模型一致
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {k: v.numpy() for k, v in dummy.items()})[0]
    with torch.no_grad():
        pt_out = model(**dummy).logits.numpy()
    diff = np.abs(onnx_out - pt_out).max()
    print(f"输出最大误差: {diff:.6f}", "OK" if diff < 1e-4 else "WARNING: 误差偏大")


if __name__ == "__main__":
    export()
