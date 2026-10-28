"""会话接口：POST /api/chat 售后问答、POST /api/feedback 用户反馈"""
import json
import os
import uuid
from fastapi import APIRouter
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from app.agent.graph import run_chat
from app.observability.trace import trace_chat, score_trace
from app.context.token_budget import summarize_history, fit_budget
from app.db.database import SessionLocal
from app.db.models import BadcaseLog

router = APIRouter()
settings = get_settings()

# 简单的内存会话存储（学习项目没上Redis）
SESSIONS: dict[str, list] = {}


class ChatRequest(BaseModel):
    session_id: str = Field(default="", description="留空则新建会话")
    user_id: str = Field(default="anonymous")
    query: str


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    answer: str
    need_human: bool
    trace_url: str = ""


class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    answer: str
    rating: int = Field(description="1点赞 0点踩")
    comment: str = ""


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = SESSIONS.setdefault(session_id, [])

    # 分层上下文管理：先摘要压缩历史，再按token预算裁剪
    if history:
        llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key,
                         base_url=settings.openai_base_url, temperature=0)
        history = summarize_history(history, llm)
        history = fit_budget(history)

    state = run_chat(session_id, req.user_id, req.query, history)

    answer = state.get("answer", "")
    kb_chunks = state.get("kb_chunks", [])

    # Langfuse trace
    trace_url = trace_chat(session_id, req.user_id, req.query,
                           state.get("intent", ""), answer,
                           state.get("tool_errors", []), kb_chunks)

    # 检索未命中 -> 记badcase，供数据飞轮清洗
    if state.get("intent") not in ("闲聊",) and not kb_chunks and not state.get("order_info"):
        _log_badcase(session_id, req.query, answer, "retrieval_miss", trace_url)

    history.append({"role": "user", "content": req.query})
    history.append({"role": "assistant", "content": answer})
    SESSIONS[session_id] = history

    return ChatResponse(
        session_id=session_id,
        intent=state.get("intent", ""),
        answer=answer,
        need_human=state.get("need_human", False),
        trace_url=trace_url,
    )


@router.post("/feedback")
def feedback(req: FeedbackRequest):
    """用户点赞/点踩：回写Langfuse分数，点踩的落badcase表"""
    trace_url = trace_chat(req.session_id, "anonymous", req.query, "", req.answer, [], [])
    score_trace(trace_url, req.rating, req.comment)
    if req.rating == 0:
        _log_badcase(req.session_id, req.query, req.answer, "user_dislike", trace_url)
    return {"ok": True}


def _log_badcase(session_id: str, question: str, answer: str, reason: str, trace_url: str):
    db = SessionLocal()
    try:
        db.add(BadcaseLog(session_id=session_id, question=question, answer=answer,
                          reason=reason, trace_url=trace_url))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"badcase写入失败: {e}")
    finally:
        db.close()
