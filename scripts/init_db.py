"""初始化数据库：建表 + 灌入种子订单
用法：python scripts/init_db.py
前置：docker compose up -d mysql，且 .env 已配置
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import engine, Base, SessionLocal
from app.db.models import Order, Ticket, KnowledgeDoc, BadcaseLog  # noqa: F401 确保建表


def seed_from_sql(path: str):
    """执行 seed_orders.sql（简单按分号切分，学习项目没上迁移工具）"""
    sql_text = open(path, encoding="utf-8").read()
    from sqlalchemy import text
    statements = []
    for stmt in sql_text.split(";"):
        stmt = stmt.strip()
        # 跳过注释和USE语句，engine已连到目标库
        if not stmt or stmt.startswith("--") or stmt.upper().startswith("USE"):
            continue
        # 去掉行内注释
        lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            statements.append(cleaned)

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    print(f"执行了 {len(statements)} 条SQL")


def main():
    print("建表中...")
    Base.metadata.create_all(engine)

    db = SessionLocal()
    try:
        if db.query(Order).count() > 0:
            print("订单表已有数据，跳过灌入（如需重置请先清空orders表）")
            return
    finally:
        db.close()

    seed_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "seed_orders.sql")
    seed_from_sql(seed_path)
    print("种子数据灌入完成，共20条模拟订单")


if __name__ == "__main__":
    main()
