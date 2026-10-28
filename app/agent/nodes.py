"""LangGraph 各节点实现：意图路由、订单查询、政策核验、生成回复"""
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from app.agent.state import AgentState
from app.agent.tools import query_order, check_refund_policy, get_logistics
from app.rag.retriever import hybrid_search
from app.context.token_budget import trim_tool_result
from classifier.predict import classify_intent

settings = get_settings()

# 需要查订单的意图
ORDER_INTENTS = {"退款申请", "物流查询", "退换货", "取消订单", "修改订单", "售后进度", "价格保护"}
# 需要查政策知识库的意图
POLICY_INTENTS = {"退款申请", "退换货", "发票开具", "价格保护", "商品质量问题", "优惠券问题", "会员积分", "安装维修", "其他咨询"}

MAX_RETRIES = 2

SYSTEM_PROMPT = """你是电商平台的售后客服助手。请严格根据【工具结果】和【知识库片段】回答用户问题。
要求：
1. 只能承诺工具结果中明确允许的操作，不确定时引导转人工客服，禁止编造承诺
2. 回答简洁，口语化，不超过200字
3. 涉及金额、时效等关键信息必须引用工具/知识库原文"""


def _extract_order_no(text: str) -> str | None:
    """从用户问题里抽订单号（数字串或CF开头）"""
    m = re.search(r"(CF\d{14}|\d{14,20})", text)
    return m.group(1) if m else None


def intent_route(state: AgentState) -> dict:
    """意图识别节点：ONNX/RoBERTa 分类 + 订单号抽取"""
    query = state["query"]
    label, conf = classify_intent(query)
    order_no = state.get("order_no") or _extract_order_no(query)
    return {"intent": label, "intent_confidence": conf, "order_no": order_no}


def order_lookup(state: AgentState) -> dict:
    """订单查询节点：物流意图查物流，其余查订单详情"""
    order_no = state.get("order_no")
    if not order_no:
        # 没订单号，标记需要向用户追问
        return {"order_info": None, "tool_errors": state.get("tool_errors", []) + ["missing_order_no"]}

    try:
        if state["intent"] == "物流查询":
            info = get_logistics(order_no)
        else:
            info = query_order(order_no)
        # 工具结果裁剪，控制token
        info = trim_tool_result(info, max_chars=400)
        return {"order_info": info}
    except Exception as e:
        retries = state.get("retries", 0) + 1
        errors = state.get("tool_errors", []) + [str(e)]
        return {"retries": retries, "tool_errors": errors, "order_info": None}


def policy_check(state: AgentState) -> dict:
    """政策核验节点：退款类走规则核验，其余走RAG检索"""
    intent = state["intent"]
    results = {"policy_hits": None, "kb_chunks": []}

    if intent in ("退款申请", "退换货", "取消订单") and state.get("order_no"):
        results["policy_hits"] = check_refund_policy(state["order_no"])

    if intent in POLICY_INTENTS:
        chunks = hybrid_search(state["query"], top_n=settings.rerank_top_n)
        results["kb_chunks"] = [{"doc_id": c["doc_id"], "text": trim_tool_result(c["text"], 300), "score": round(c["score"], 4)} for c in chunks]

    return results


def answer(state: AgentState) -> dict:
    """回复生成节点：LLM 基于确定性工具结果 + 知识片段作答"""
    intent = state["intent"]

    # 闲聊直接回，不占检索资源
    if intent == "闲聊":
        return {"answer": "您好，我是售后智能助手，可以帮您查询订单、处理退换货和解答售后政策问题，请问有什么可以帮您？", "need_human": False}

    # 重试超限或订单号缺失，转人工/追问
    if state.get("retries", 0) > MAX_RETRIES:
        return {"answer": "抱歉，系统查询遇到问题，已为您转接人工客服处理。", "need_human": True}
    if intent in ORDER_INTENTS and not state.get("order_info"):
        if "missing_order_no" in state.get("tool_errors", []):
            return {"answer": "麻烦提供一下您的订单号，我帮您查询～", "need_human": False}
        return {"answer": "抱歉，没有查询到该订单，请确认订单号是否正确，或联系人工客服。", "need_human": True}

    llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key,
                     base_url=settings.openai_base_url, temperature=0.1)

    context_parts = [f"【用户意图】{intent}"]
    if state.get("order_info"):
        context_parts.append(f"【订单信息】{state['order_info']}")
    if state.get("policy_hits"):
        context_parts.append(f"【政策核验结果】{state['policy_hits']}")
    if state.get("kb_chunks"):
        context_parts.append("【知识库片段】\n" + "\n".join(c["text"] for c in state["kb_chunks"]))

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="\n\n".join(context_parts) + f"\n\n【用户问题】{state['query']}"),
    ]
    resp = llm.invoke(messages)
    return {"answer": resp.content, "need_human": False}
