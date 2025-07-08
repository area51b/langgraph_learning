from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv
import os

# Load Gemini API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- State Definition ---
class LoopState(TypedDict):
    input: str
    output: str
    attempt: int

# --- Node: Generate Gemini Output ---
def generate_text(state: LoopState) -> LoopState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    response = llm.invoke(state["input"])
    state["output"] = response.content
    state["attempt"] += 1
    return state

# --- Router: Decide what to do next ---
def routing_logic(state: LoopState) -> str:
    if "OK" in state["output"]:
        return "finish"
    elif state["attempt"] >= 3:
        return "finish"
    else:
        return "generate_text"

# --- Dummy finish node ---
def finish(state: LoopState) -> LoopState:
    return state

# --- Build the LangGraph ---
builder = StateGraph(LoopState)

builder.add_node("generate_text", generate_text)
builder.add_node("finish", finish)

# Add a dummy router node
builder.add_node("router", lambda state: state)
builder.add_edge("generate_text", "router")

# Conditional edge from router
builder.add_conditional_edges("router", routing_logic)

builder.set_entry_point("generate_text")
builder.set_finish_point("finish")

graph = builder.compile()

if __name__ == "__main__":
    while True:
        text = input("\n🧑 You: ")
        if text.lower() in {"exit", "quit"}:
            break

        state: LoopState = {
            "input": text,
            "output": "",
            "attempt": 0
        }

        result = graph.invoke(state)
        print(f"🌀 Attempts: {result['attempt']}")
        print("🤖 Final Output:", result["output"])
