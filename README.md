# AI 导购智能体（LangGraph 最小可运行 demo）

一个演示「AI 智能体」核心能力的教学级项目：基于 **LangGraph** 编排一个**导购 Agent**，
让模型在对话中自主决定调用 **工具**（搜索商品 / 查询库存），并以**流式**方式输出推荐结果。

## 它演示了什么

这个 demo 不求业务复杂，只求把真实 AI 客服/导购场景里三个最核心的工程点讲清楚：

| 工程点 | 对应代码 | 说明 |
|--------|----------|------|
| **Agent 编排** | `StateGraph`（`agent.py` 第三层） | 用状态图表达「模型思考 → 调工具 → 拿结果 → 再思考」的循环 |
| **工具封装** | `@tool` + `bind_tools`（`agent.py` 第二层） | 把业务查询能力封装成模型可调用的标准化工具（对应 MCP 协议思路） |
| **流式交互** | `run_stream` + `app.stream`（`agent.py` 第四层） | 逐字输出，模拟 SSE 首字加速的体验 |

## 架构

```
用户输入（偏好/预算）
      │
      ▼
┌─────────────────────────┐
│   Agent 节点（LLM）      │  模型决定：要不要调工具？调哪个？
└──────────┬──────────────┘
           │ 需要查数据
           ▼
┌─────────────────────────┐
│   工具节点（业务查询）    │  search_products / query_inventory
└──────────┬──────────────┘
           │ 工具结果写回
           ▼
    回到 Agent 节点 → 生成最终推荐 → 流式输出
```

## 运行

### 1. 安装依赖（建议 Python 3.10+）

```bash
pip install -r requirements.txt
```

### 2. 配置模型

复制 `.env.example` 为 `.env`，填入你的 Key。任选一种：

```bash
# 方式 A：OpenAI 官方
OPENAI_API_KEY=sk-xxxx
OPENAI_MODEL=gpt-4o-mini

# 方式 B：国内兼容接口（推荐，省钱）
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

### 3. 运行

```bash
python agent.py
```

预期会看到 Agent 依次「搜索商品 → 查询库存 → 流式输出推荐」的完整过程。

## 目录结构

```
ai-shopping-agent/
├── agent.py           # 核心：Agent 编排 + 工具封装 + 流式输出
├── requirements.txt   # 依赖
├── .env.example       # 环境变量示例
└── README.md          # 本文件
```

## 可以怎么扩展（体现能力的下一步）

- 把「内存商品列表」换成真实数据库 / 业务 API（`PRODUCTS` 这一层）
- 把工具封装成真正的 **MCP server**（`fastmcp` 或官方 SDK），模型侧用 MCP client 连接
- 补上**对话记忆**（跨会话保存用户偏好，做个性化推荐）
- 补上**链路日志 / 可观测**（记录每次工具调用的耗时与结果，便于排查）
