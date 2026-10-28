"""AgentState：LangGraph 全局状态定义"""
from typing import TypedDict, Optional, Any
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    messages: list[BaseMessage]        # 对话历史（含裁剪后的上下文）
    session_id: str
    user_id: str
    query: str                         # 当前用户问题原文
    intent: str                        # 意图分类结果（17类之一）
    intent_confidence: float
    order_no: Optional[str]            # 从问题中抽取的订单号
    order_info: Optional[dict]         # query_order 结果
    policy_hits: Optional[dict]        # 政策核验/检索结果
    kb_chunks: list[dict]              # RAG 检索到的知识片段
    answer: Optional[str]              # 最终回复
    need_human: bool                   # 是否转人工
    retries: int                       # 工具调用重试次数
    tool_errors: list[str]             # 记录工具报错，方便badcase归因
