"""LangGraph 售后工作流编排
确定性Workflow + Agent决策：意图路由后按意图走不同分支，条件边约束执行路径
"""
from langgraph.graph import StateGraph, START, END
from app.agent.state import AgentState
from app.agent.nodes import intent_route, order_lookup, policy_check, answer, MAX_RETRIES

ORDER_INTENTS = {"退款申请", "物流查询", "退换货", "取消订单", "修改订单", "售后进度", "价格保护"}


def route_after_intent(state: AgentState) -> str:
    """意图路由条件边：查订单 / 查政策 / 直接回复（闲聊、投诉举报走兜底）"""
    intent = state.get("intent", "其他咨询")
    if intent in ORDER_INTENTS:
        return "order_lookup"
    if intent in ("投诉举报",):
        # 投诉类直接转人工，不让大模型自己发挥
        return "answer"
    return "policy_check"


def route_after_order(state: AgentState) -> str:
    """订单查询后：重试未超限且需要政策核验则继续，否则直接生成回复"""
    if state.get("retries", 0) > MAX_RETRIES:
        return "answer"
    return "policy_check"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("intent_route", intent_route)
    g.add_node("order_lookup", order_lookup)
    g.add_node("policy_check", policy_check)
    g.add_node("answer", answer)

    g.add_edge(START, "intent_route")
    g.add_conditional_edges("intent_route", route_after_intent,
                            {"order_lookup": "order_lookup", "policy_check": "policy_check", "answer": "answer"})
    g.add_conditional_edges("order_lookup", route_after_order,
                            {"policy_check": "policy_check", "answer": "answer"})
    g.add_edge("policy_check", "answer")
    g.add_edge("answer", END)
    return g.compile()


# 全局单例，FastAPI 启动时编译一次
graph = build_graph()


def run_chat(session_id: str, user_id: str, query: str, history: list | None = None) -> dict:
    """执行一次工作流，返回最终state"""
    init_state: AgentState = {
        "messages": history or [],
        "session_id": session_id,
        "user_id": user_id,
        "query": query,
        "retries": 0,
        "tool_errors": [],
        "need_human": False,
    }
    return graph.invoke(init_state)
