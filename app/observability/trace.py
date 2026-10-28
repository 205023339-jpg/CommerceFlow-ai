"""Langfuse 全链路 Trace 封装：工作流每个节点一个span，方便badcase归因"""
import os
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from config.settings import get_settings

settings = get_settings()

_client = None


def enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def get_client() -> Langfuse:
    global _client
    if _client is None:
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    return _client


def get_callback_handler(session_id: str, user_id: str) -> CallbackHandler | None:
    """LangChain/LangGraph 回调，自动记录LLM调用、token用量、延迟"""
    if not enabled():
        return None
    return CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        session_id=session_id,
        user_id=user_id,
    )


def trace_chat(session_id: str, user_id: str, query: str, intent: str,
               answer: str, tool_errors: list, kb_chunks: list) -> str:
    """一次会话打一条trace，返回trace链接（存到badcase_logs方便回查）"""
    if not enabled():
        return ""
    client = get_client()
    trace = client.trace(
        name="chat",
        session_id=session_id,
        user_id=user_id,
        input={"query": query},
        output={"answer": answer, "intent": intent},
        metadata={"tool_errors": tool_errors, "kb_hit": [c.get("doc_id") for c in kb_chunks]},
        tags=["after-sales"],
    )
    client.flush()
    return f"{settings.langfuse_host}/trace/{trace.id}"


def score_trace(trace_url: str, score: int, comment: str = ""):
    """用户反馈打分回写Langfuse（点赞1/点踩0），用于badcase筛选"""
    if not enabled() or not trace_url:
        return
    trace_id = trace_url.rstrip("/").split("/")[-1]
    get_client().score(trace_id=trace_id, name="user-feedback", value=score, comment=comment)
