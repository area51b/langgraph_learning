# Phase 1: LangGraph Fundamentals with Gemini (Python)

This document summarizes all the working code, concepts, and lessons learned during Phase 1 of building agents using the LangGraph framework with the Gemini 2.0 Flash API.

---

## ✅ Prerequisites

- Python 3.10+
- Install dependencies:
  ```bash
  uv pip install langgraph langchain langchain-google-genai python-dotenv
  ```
- `.env` file with:
  ```env
  GOOGLE_API_KEY=your_api_key_here
  ```

---

## 🧪 Step 1.4 – Echo Bot

### 🎯 Goal

Build the simplest LangGraph agent that takes user input and echoes it back using Gemini.

### 📊 Graph Representation

```
[input] → echo → [output]
```

### 🔧 We’ll Build This

- A single-node LangGraph.
- Pass user input to Gemini and return the reply.

### 💻 Code

```python
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

class EchoState(TypedDict):
    input: str
    output: str

def echo_node(state: EchoState) -> EchoState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    response = llm.invoke(state["input"])
    state["output"] = response.content
    return state

builder = StateGraph(EchoState)
builder.add_node("echo", echo_node)
builder.set_entry_point("echo")
builder.set_finish_point("echo")
graph = builder.compile()

if __name__ == "__main__":
    while True:
        text = input("\U0001F9D1 You: ")
        if text.lower() in {"exit", "quit"}:
            break
        state = {"input": text, "output": ""}
        result = graph.invoke(state)
        print("\U0001F916 Gemini:", result["output"])
```

### 🧪 Sample Inputs

```
🧑 You: Hello
🤖 Gemini: Hello! How can I assist?
```

### 🧠 What You Learned

- Basic LangGraph structure
- LLM invocation
- State input/output flow

---

## 🔁 Step 1.5 – Multi-Node Graph

### 🎯 Goal

Split the echo bot into multiple modular nodes.

### 📊 Graph Representation

```
[input_handler] → [gemini_node] → [postprocessor]
```

### 🔧 We’ll Build This

- A preprocessing node
- A Gemini response node
- A postprocessing node to finalize output

### 💻 Code

```python
class MultiNodeState(TypedDict):
    input: str
    raw_response: str
    final_output: str

def input_handler(state: MultiNodeState) -> MultiNodeState:
    return state

def gemini_node(state: MultiNodeState) -> MultiNodeState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    response = llm.invoke(state["input"])
    state["raw_response"] = response.content
    return state

def postprocessor(state: MultiNodeState) -> MultiNodeState:
    state["final_output"] = state["raw_response"].strip()
    return state

builder = StateGraph(MultiNodeState)
builder.add_node("input_handler", input_handler)
builder.add_node("gemini_node", gemini_node)
builder.add_node("postprocessor", postprocessor)

builder.set_entry_point("input_handler")
builder.add_edge("input_handler", "gemini_node")
builder.add_edge("gemini_node", "postprocessor")
builder.set_finish_point("postprocessor")

graph = builder.compile()
```

### 🧠 What You Learned

- Modular node chaining
- Clearer code structure

---

## 🔀 Step 1.6 – Conditional Branching

### 🎯 Goal

Branch logic based on user input: answer questions or handle commands.

### 📊 Graph Representation

```
          [router]
             ↓
    ┌────────┴────────┐
[answer_question] [execute_command]
    └────────┬────────┘
           [postprocessor] → [END]
```

### 🔧 We’ll Build This

- A router function that branches based on input
- Nodes for question answering and command handling
- A unified postprocessor

### 💻 Code (Key Parts)

```python
class BranchState(TypedDict):
    input: str
    type: Literal["question", "command"]
    result: str
    final_output: str

def classify_input(state: BranchState) -> str:
    return "answer_question" if state["input"].endswith("?") else "execute_command"

# (answer_question, execute_command, postprocessor omitted for brevity — same as previous)

builder = StateGraph(BranchState)
builder.add_node("router", lambda state: state)
builder.add_node("answer_question", answer_question)
builder.add_node("execute_command", execute_command)
builder.add_node("postprocessor", postprocessor)

builder.set_entry_point("router")
builder.add_conditional_edges("router", classify_input)
builder.add_edge("answer_question", "postprocessor")
builder.add_edge("execute_command", "postprocessor")
builder.set_finish_point("postprocessor")
```

### 🧠 What You Learned

- Conditional routing
- Branch and merge paths

---

## 🔁 Step 1.7 – Looping / Retry

### 🎯 Goal

Retry a Gemini generation node until output contains "OK" or max retries reached.

### 📊 Graph Representation

```
[generate_text] → [router] ───┐
                   ↓         │
              [finish] ←─────┘
```

### 🔧 We’ll Build This

- Generate text node
- Router with loop logic
- Finish node

### 💻 Code

```python
class LoopState(TypedDict):
    input: str
    output: str
    attempt: int

def generate_text(state: LoopState) -> LoopState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    response = llm.invoke(state["input"])
    state["output"] = response.content
    state["attempt"] += 1
    return state

def routing_logic(state: LoopState) -> str:
    return "finish" if "OK" in state["output"] or state["attempt"] >= 3 else "generate_text"

def finish(state: LoopState) -> LoopState:
    return state

builder = StateGraph(LoopState)
builder.add_node("generate_text", generate_text)
builder.add_node("router", lambda state: state)
builder.add_node("finish", finish)

builder.set_entry_point("generate_text")
builder.add_edge("generate_text", "router")
builder.add_conditional_edges("router", routing_logic)
builder.set_finish_point("finish")

graph = builder.compile()
```

### 🧪 Sample Inputs

```
🧑 You: Say something ending with OK
```

### 🧠 What You Learned

- Looping with conditional edges
- Retry logic with counters

---

## ✅ Phase 1 Summary

You now know how to:

- Create single-node and multi-node LangGraph pipelines
- Define and update typed state
- Route execution using `add_edge()` and `add_conditional_edges()`
- Implement retry loops with counter logic

You’re now ready for **Phase 2: Memory & Tools**!

---

