"""RoBERTa(chinese-roberta-wwm-ext) 17类意图微调 + 阈值校准
用法：python -m classifier.train --data classifier/data/train.jsonl --epochs 5
数据格式：每行 {"text": "...", "label": "物流查询"}
"""
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import f1_score, classification_report

from classifier.labels import LABELS, LABEL2ID, NUM_LABELS

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
MAX_LEN = 64  # 客服短句够用，省显存


class IntentDataset(Dataset):
    def __init__(self, path: str, tokenizer):
        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.samples.append((item["text"], LABEL2ID[item["label"]]))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(text, max_length=MAX_LEN, truncation=True)
        enc["labels"] = label
        return enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="micro")
    return {"micro_f1": f1, "accuracy": (preds == labels).mean()}


def calibrate_threshold(model, tokenizer, val_path: str) -> dict:
    """阈值校准：在验证集上跑一遍，找使 micro-F1 最高的置信度阈值。
    低于阈值的样本预测时兜底到"其他咨询"，避免低置信度误分类。
    """
    ds = IntentDataset(val_path, tokenizer)
    confs, correct = [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            batch = {k: torch.tensor([v] if not isinstance(v, list) else [v]) for k, v in item.items()}
            out = model(**batch)
            probs = torch.softmax(out.logits, dim=-1)[0]
            conf, pred = probs.max(dim=-1)
            confs.append(conf.item())
            correct.append(int(pred.item() == item["labels"]))

    best_t, best_f1 = 0.0, 0.0
    for t in np.arange(0.5, 0.99, 0.01):
        # 低于阈值的按"其他咨询"处理（id=15）
        preds_correct = [c if conf >= t else (LABEL2ID["其他咨询"] == LABELS.index("其他咨询"))
                         for conf, c in zip(confs, correct)]
        # 简化：直接统计保留样本的准确率与覆盖率的调和平均
        kept = [(conf, c) for conf, c in zip(confs, correct) if conf >= t]
        if not kept:
            continue
        acc = sum(c for _, c in kept) / len(kept)
        cov = len(kept) / len(confs)
        f1 = 2 * acc * cov / (acc + cov) if acc + cov > 0 else 0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    return {"threshold": round(best_t, 2), "calib_score": round(best_f1, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="classifier/data/train.jsonl")
    parser.add_argument("--val", default="classifier/data/val.jsonl")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--out", default="classifier/checkpoints")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    train_ds = IntentDataset(args.data, tokenizer)
    val_ds = IntentDataset(args.val, tokenizer) if __import__("os").path.exists(args.val) else None

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            eval_strategy="epoch" if val_ds else "no",
            save_strategy="epoch",
            logging_steps=20,
            warmup_ratio=0.1,
            fp16=torch.cuda.is_available(),
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    # 阈值校准并写入配置，推理时加载
    if val_ds:
        th = calibrate_threshold(model, tokenizer, args.val)
        with open(f"{args.out}/calibration.json", "w", encoding="utf-8") as f:
            json.dump(th, f, ensure_ascii=False, indent=2)
        print("阈值校准结果:", th)
        print(classification_report(
            [s[1] for s in val_ds.samples],
            [np.argmax(model(**{k: torch.tensor([v]) for k, v in val_ds[i].items()}).logits.detach().numpy()) for i in range(len(val_ds))],
            target_names=LABELS, zero_division=0))


if __name__ == "__main__":
    main()
