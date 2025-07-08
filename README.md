# LangGraph Agent Learning Project (with Gemini 2.0 Flash API)

This repository documents a complete learning journey of building AI agents using the [LangGraph](https://github.com/langchain-ai/langgraph) framework in Python, powered by Google's Gemini 2.0 Flash API.

---

## 🔍 Overview

This project explores how to design stateful, graph-based LLM agents using LangGraph — starting from simple echo bots and progressing to advanced branching, memory, and tool invocation scenarios (including streamable HTTP tools and MCP servers).

---

## 🚀 Learning Plan

The journey is structured in **4 Phases**:

### ✅ Phase 1: LangGraph Fundamentals

Learn core concepts using simple examples.

#### 🔹 Step 1.1 – What is LangGraph?

- LangGraph is a framework for building stateful, multi-step, deterministic agents using graphs.
 
#### 🔹 Step 1.2 – Install Dependencies


#### 🔹 Step 1.3 – Set Up Gemini 2.0 Flash


#### 🔹 Step 1.4 – Echo Bot

- One node: echo user input using Gemini
- Input → [echo] → Output

#### 🔹 Step 1.5 – Multi-Node Graph

- Nodes: input handler → Gemini node → postprocessor
- Demonstrates modularity

#### 🔹 Step 1.6 – Conditional Branching

- Route based on input type (question vs command)
- Conditional edges using router logic

#### 🔹 Step 1.7 – Looping / Retry

- Retry a node until success criteria met or max retries
- Looping via conditional edges

👉 See [`phase1_langgraph_basics.md`](./phase1_langgraph_basics.md) for full working code and notes.

---

### 🔄 Phase 2: Memory + Tools + Agents

Introduce LangChain-style tools, memory, and reasoning loops.

#### 🔹 Step 2.1 – Add Memory

- Add conversation memory to the agent state
- Use LangChain `ConversationBufferMemory`

#### 🔹 Step 2.2 – LangChain Tools

- Add tools like calculator, search, and file lookup
- Tool calling via Gemini or manual routing

#### 🔹 Step 2.3 – Router with Tool Use

- Classify user intent → Call tool / generate text
- Conditional + Tool node integration

#### 🔹 Step 2.4 – Reflection + Retry

- Retry bad answers using previous output
- Add inner loop with limited retries

#### 🔹 Step 2.5 – Failure Handling

- Handle tool failure and LLM errors gracefully

---

### ⚙️ Phase 3: Agentic Workflows & LangGraph Patterns 

Create agents that can reason, call tools, and self-correct.

#### 🔹 Step 3.1 – Multi-agent Collaboration

- Nodes as different agents (planner, executor, verifier).
- Communicating via shared state.

#### 🔹 Step 3.2 – ReAct Pattern in LangGraph

- Implement reflection and action.
- Combine LLM + Tool + Retry logic.

#### 🔹 Step 3.3 – Recursive / Looping Agents

- Agent that iterates on output until criteria met.

#### 🔹 Step 3.4 – LangGraph Memory Stores

- Store internal state or conversation logs externally (Redis, local).

---

### ⚙️ Phase 4: Advanced Agent Systems

Design robust systems with streaming, APIs, and external service orchestration.

#### 🔹 Step 4.1 – Streamable Tool Results

- Return streaming results to frontend
- Use `yield` in LangGraph node

#### 🔹 Step 4.2 – HTTP Tool Invocation

- Tool sends HTTP request to external service
- Integrate custom tools via `requests`

#### 🔹 Step 4.3 – Stateless MCP Server

- LangGraph agent talks to MCP microservice via HTTP
- Handles command routing

#### 🔹 Step 4.4 – Resilient Agent

- Add retries, error fallback paths, edge case handling

#### 🔹 Step 4.5 – Async & Parallel Tools

- Run tool branches in parallel
- Await async responses

---

## 🧱 Setup Instructions

1. Clone the repo:

```bash
git clone https://github.com/your-username/langgraph-learning.git
cd langgraph-learning
```

2. Install dependencies (using `uv` or `pip`):

```bash
uv pip install -r requirements.txt
```

3. Create `.env` with your Gemini key:

```env
GOOGLE_API_KEY=your_api_key_here
```

4. Run any step:

```bash
uv run step1_4_echo_bot.py
```

---

## 🧠 Tools Used

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Gemini 2.0 Flash](https://ai.google.dev/)
- Python 3.10+

---

## 📁 Folder Structure

```
/langgraph-learning
│
├── phase1_langgraph_basics.md   # Complete notes and code for Phase 1
├── README.md                    # This file
├── .env                         # API key (not committed)
├── requirements.txt             # Python dependencies
├── step1_4_echo_bot.py          # Echo Bot
├── step1_5_multinode.py         # Multi-node Graph
├── step1_6_branching.py         # Conditional Branching
├── step1_7_looping.py           # Loop / Retry Pattern
└── ... (more steps coming)
```

---

## 📌 License

MIT License. Feel free to use, fork, or contribute.

---

## 🙌 Credits

Built with ❤️ by [Your Name]. Powered by LangChain, LangGraph, and Gemini.

---

## 📞 Feedback / Questions?

File an issue or reach out at: [[your-email@example.com](mailto\:your-email@example.com)]

