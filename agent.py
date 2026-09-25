"""AI 导购智能体 —— 最小可运行 demo

一个基于 LangGraph 编排的「导购 Agent」：
1. 接收用户偏好（预算、品类、场景）
2. 通过「工具调用」实时查询商品与库存（模拟 MCP 工具）
3. 流式输出推荐结果

体现了真实 AI 客服/导购场景里的三个核心工程点：
- Agent 编排（状态图 + 工具调用循环）
- 工具封装（把业务查询能力封装成模型可调用的工具）
- 流式交互（SSE 风格的逐字推送）

运行方式见 README.md。
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


# ============================================================================
# 第一层：模拟业务数据（真实项目里这里是数据库 / 业务 API）
# ============================================================================

# 商品库：模拟一个线上的商品列表
PRODUCTS = [
    {"id": "P001", "name": "无线降噪耳机", "category": "数码", "price": 599, "stock": 120},
    {"id": "P002", "name": "机械键盘 87 键", "category": "数码", "price": 349, "stock": 45},
    {"id": "P003", "name": "智能手表", "category": "数码", "price": 1299, "stock": 8},
    {"id": "P004", "name": "便携咖啡机", "category": "家电", "price": 259, "stock": 60},
    {"id": "P005", "name": "跑步鞋", "category": "运动", "price": 499, "stock": 200},
    {"id": "P006", "name": "瑜伽垫", "category": "运动", "price": 99, "stock": 300},
]


# ============================================================================
# 第二层：封装成「工具」（对应 MCP 协议里把业务 API 标准化成 tool 的步骤）
# ============================================================================

@tool
def search_products(keyword: str) -> str:
    """根据关键词搜索商品，返回匹配的商品列表（id、名称、分类、价格）。

    Args:
        keyword: 商品关键词，例如「耳机」「咖啡机」。
    """
    matched = [p for p in PRODUCTS if keyword and keyword in p["name"]]
    if not matched:
        return "没有找到匹配的商品，请换个关键词试试。"
    lines = [f"- {p['id']} {p['name']}（{p['category']}，¥{p['price']}）" for p in matched]
    return "\n".join(lines)


@tool
def query_inventory(product_id: str) -> str:
    """根据商品 id 查询实时库存数量。

    Args:
        product_id: 商品 id，例如「P001」。
    """
    for p in PRODUCTS:
        if p["id"] == product_id:
            return f"{p['name']}（{product_id}）当前库存：{p['stock']} 件"
    return "未找到该商品，请确认商品 id 是否正确。"


# 两个工具注册成列表，交给模型
TOOLS = [search_products, query_inventory]


# ============================================================================
# 第三层：用 LangGraph 编排 Agent 状态图
# ============================================================================

class AgentState(TypedDict):
    """Agent 状态：用消息列表承载对话历史（含工具调用消息）。"""
    messages: Annotated[list, add_messages]


def make_llm() -> ChatOpenAI:
    """创建绑定了工具的 LLM。

    优先读标准变量 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL；
    若都没有，则自动回退到常见国内平台的 Key（DeepSeek、智谱）。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL")

    # 回退到 DeepSeek
    if not api_key and os.getenv("DEEPSEEK_API_KEY"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = base_url or "https://api.deepseek.com/v1"
        model = model or "deepseek-chat"
    # 回退到智谱
    if not api_key and os.getenv("ZHIPU_API_KEY"):
        api_key = os.getenv("ZHIPU_API_KEY")
        base_url = base_url or "https://open.bigmodel.cn/api/paas/v4"
        model = model or "glm-4-flash"

    if not model:
        model = "gpt-4o-mini"

    kwargs = {"model": model, "temperature": 0.2}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    llm = ChatOpenAI(**kwargs)
    # 让模型在需要时调用上面封装的工具
    return llm.bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict:
    """Agent 节点：调用 LLM，返回新的消息（可能是工具调用）。"""
    llm = make_llm()
    system = SystemMessage(
        content=(
            "你是一位电商导购智能体。根据用户的需求，先搜索商品，"
            "必要时查询库存，再给出简洁、有针对性的推荐。"
        )
    )
    response = llm.invoke([system, *state["messages"]])
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """工具节点：执行模型请求的工具调用，把结果写回消息流。"""
    from langchain_core.messages import ToolMessage

    last = state["messages"][-1]
    tool_messages: list[ToolMessage] = []
    for call in last.tool_calls:
        name = call["name"]
        args = call["args"]
        # 按工具名分发执行
        result = {t.name: t for t in TOOLS}[name].invoke(args)
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    """路由：如果最后一条消息还有工具调用，继续走工具节点，否则结束。"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


# 构建状态图：agent <-> tools
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

app = graph.compile()


# ============================================================================
# 第四层：流式输出（模拟 SSE 首字加速的交互体验）
# ============================================================================

def run_stream(query: str):
    """流式运行 Agent，逐字输出最终答案。"""
    print("🛒 导购智能体启动，正在处理你的需求...\n")

    final_inputs = {"messages": [HumanMessage(content=query)]}

    from langchain_core.messages import ToolMessage

    for event in app.stream(final_inputs, stream_mode="values"):
        messages = event.get("messages", [])
        for msg in messages:
            # 打印工具调用过程（便于看到 Agent 是怎么决策的）
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    print(f"  ⚙️ 调用工具：{call['name']}({call['args']})")
            if isinstance(msg, ToolMessage):
                print(f"  📦 工具返回：{msg.content[:60]}...")
            # 最终答案逐字流式输出
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                print("\n💬 导购回答：", end="", flush=True)
                for chunk in msg.content:
                    print(chunk, end="", flush=True)
                print("\n")


if __name__ == "__main__":
    # 示例查询：让 Agent 走「搜索商品 → 查库存 → 推荐」的完整链路
    demo_query = "我想买一副预算 500 元以内的耳机，有推荐吗？库存够吗？"
    run_stream(demo_query)
