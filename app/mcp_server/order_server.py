"""简易 MCP Server：把订单查询能力通过 MCP 协议暴露出去
Agent 侧作为 MCP client 接入，实现工具与业务系统解耦。
运行：python -m app.mcp_server.order_server （stdio传输）
"""
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import json

from app.agent.tools import query_order, get_logistics, check_refund_policy

server = Server("commerceflow-order")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_order",
            description="根据订单号查询订单的商品、金额、状态等信息",
            inputSchema={
                "type": "object",
                "properties": {"order_no": {"type": "string", "description": "订单号，如 CF20260512000123"}},
                "required": ["order_no"],
            },
        ),
        Tool(
            name="get_logistics",
            description="根据订单号查询物流公司和运单号",
            inputSchema={
                "type": "object",
                "properties": {"order_no": {"type": "string"}},
                "required": ["order_no"],
            },
        ),
        Tool(
            name="check_refund_policy",
            description="核验指定订单当前是否满足退款/退货条件，返回政策依据",
            inputSchema={
                "type": "object",
                "properties": {"order_no": {"type": "string"}},
                "required": ["order_no"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # 工具直接复用本地函数（同一份MySQL数据源），保证MCP内外行为一致
    handlers = {
        "query_order": query_order,
        "get_logistics": get_logistics,
        "check_refund_policy": check_refund_policy,
    }
    if name not in handlers:
        return [TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]
    result = handlers[name](arguments.get("order_no", ""))
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
