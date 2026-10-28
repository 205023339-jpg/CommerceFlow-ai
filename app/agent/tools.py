"""Agent 工具函数：订单/物流/政策核验，全部查MySQL，确定性执行"""
from datetime import datetime
from sqlalchemy.orm import Session
from app.db.models import Order
from app.db.database import SessionLocal

# 不同状态的退款政策规则（简化版，真实项目从知识库读）
REFUND_RULES = {
    "已支付": {"can_refund": True, "desc": "未发货订单可直接全额退款"},
    "已发货": {"can_refund": True, "desc": "已发货需拦截包裹或签收后走7天无理由"},
    "已签收": {"can_refund": True, "days": 7, "desc": "签收7天内可申请无理由退货"},
    "退款中": {"can_refund": False, "desc": "已有退款申请在处理中"},
    "已完成": {"can_refund": False, "desc": "订单已完成，超过售后时效需人工审核"},
}


def query_order(order_no: str) -> dict:
    """按订单号查询订单信息，查不到返回error字段"""
    db: Session = SessionLocal()
    try:
        order = db.query(Order).filter(Order.order_no == order_no).first()
        if not order:
            return {"error": "order_not_found", "order_no": order_no}
        return {
            "order_no": order.order_no,
            "product_name": order.product_name,
            "amount": order.amount,
            "status": order.status,
            "created_at": order.created_at.strftime("%Y-%m-%d %H:%M") if order.created_at else "",
            "received_at": order.received_at.strftime("%Y-%m-%d") if order.received_at else "",
        }
    finally:
        db.close()


def check_refund_policy(order_no: str) -> dict:
    """根据订单状态核验退款政策，返回是否可退及说明"""
    info = query_order(order_no)
    if "error" in info:
        return info
    status = info["status"]
    rule = REFUND_RULES.get(status, {"can_refund": False, "desc": "未知状态，转人工"})

    result = {"order_no": order_no, "status": status, **rule}

    # 已签收订单额外校验7天时效
    if status == "已签收" and info.get("received_at"):
        received = datetime.strptime(info["received_at"], "%Y-%m-%d")
        if (datetime.now() - received).days > 7:
            result["can_refund"] = False
            result["desc"] = "已超出签收后7天无理由退货时效，如有质量问题可联系人工客服"
    return result


def get_logistics(order_no: str) -> dict:
    """查询物流信息"""
    db: Session = SessionLocal()
    try:
        order = db.query(Order).filter(Order.order_no == order_no).first()
        if not order:
            return {"error": "order_not_found", "order_no": order_no}
        if not order.logistics_no:
            return {"order_no": order_no, "status": order.status, "msg": "订单尚未发货，暂无物流信息"}
        return {
            "order_no": order_no,
            "status": order.status,
            "company": order.logistics_company,
            "tracking_no": order.logistics_no,
            "shipped_at": order.shipped_at.strftime("%Y-%m-%d %H:%M") if order.shipped_at else "",
        }
    finally:
        db.close()
