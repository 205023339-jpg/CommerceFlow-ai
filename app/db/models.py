"""数据库表模型"""
from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base


class Order(Base):
    """订单表（模拟数据）"""
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_no: Mapped[str] = mapped_column(String(32), unique=True, index=True, comment="订单号")
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    product_name: Mapped[str] = mapped_column(String(128))
    amount: Mapped[float] = mapped_column(Float, comment="实付金额")
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), comment="已支付/已发货/已签收/退款中/已完成")
    logistics_no: Mapped[str] = mapped_column(String(32), default="", comment="运单号")
    logistics_company: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    shipped_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    received_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)


class Ticket(Base):
    """售后工单"""
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticket_no: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    order_no: Mapped[str] = mapped_column(String(32), index=True)
    user_id: Mapped[str] = mapped_column(String(32))
    intent: Mapped[str] = mapped_column(String(32), comment="意图分类结果")
    content: Mapped[str] = mapped_column(Text, comment="用户诉求")
    status: Mapped[str] = mapped_column(String(16), default="处理中")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class KnowledgeDoc(Base):
    """知识库文档登记表（正文切片存Milvus，这里存元信息）"""
    __tablename__ = "knowledge_docs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(64), unique=True, comment="文档唯一ID")
    title: Mapped[str] = mapped_column(String(128))
    source_file: Mapped[str] = mapped_column(String(256))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


class BadcaseLog(Base):
    """badcase 记录：检索未命中 / 用户点踩，供数据飞轮清洗回写"""
    __tablename__ = "badcase_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, nullable=True)
    reason: Mapped[str] = mapped_column(String(64), comment="retrieval_miss/low_score/user_dislike")
    trace_url: Mapped[str] = mapped_column(String(256), default="", comment="Langfuse trace链接")
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
