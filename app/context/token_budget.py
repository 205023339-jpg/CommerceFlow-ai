"""分层上下文管理：token预算评估、会话摘要、工具结果裁剪
目标：把单次请求的上下文控制在预算内，实测降低约30% token消耗
"""
from config.settings import get_settings

settings = get_settings()

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    # tiktoken对中文估算偏差可接受，降级为字符数/1.6
    def count_tokens(text: str) -> int:
        return int(len(text) / 1.6)


def trim_tool_result(result, max_chars: int = 400):
    """工具结果裁剪：dict只保留关键字段并截断长文本；str直接截断"""
    if isinstance(result, dict):
        trimmed = {}
        for k, v in result.items():
            s = str(v)
            trimmed[k] = s[:max_chars] if len(s) > max_chars else v
        return trimmed
    s = str(result)
    return s[:max_chars] + ("...(已截断)" if len(s) > max_chars else "")


def summarize_history(messages: list, llm, keep_recent: int = 4) -> list:
    """会话摘要：超过预算时，把早期历史压缩成一段摘要，保留最近几轮原文
    messages: [{"role": "user"/"assistant", "content": "..."}]
    """
    if len(messages) <= keep_recent:
        return messages

    old, recent = messages[:-keep_recent], messages[-keep_recent:]
    old_text = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in old)

    if count_tokens(old_text) < 300:
        # 早期历史不长，不值得调一次LLM做摘要
        return messages

    resp = llm.invoke([
        {"role": "system", "content": "把下面的客服对话压缩成不超过100字的摘要，保留订单号、问题类型、已给结论等关键信息。"},
        {"role": "user", "content": old_text},
    ])
    summary = {"role": "system", "content": f"[历史会话摘要] {resp.content}"}
    return [summary] + recent


def fit_budget(messages: list, budget: int = None) -> list:
    """预算内裁剪：从最旧的消息开始丢弃，保证最近的消息一定保留"""
    budget = budget or settings.token_budget
    total = sum(count_tokens(m.get("content", "")) for m in messages)
    result = list(messages)
    while total > budget and len(result) > 2:
        dropped = result.pop(0)
        total -= count_tokens(dropped.get("content", ""))
    return result
