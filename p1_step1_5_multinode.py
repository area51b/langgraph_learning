from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
import os

# Load Gemini API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- State Definition ---
class MultiNodeState(TypedDict):
    input: str
    raw_response: str
    final_output: str

# --- Node 1: Input Handler ---
def input_handler(state: MultiNodeState) -> MultiNodeState:
    # (Here you could clean or validate the input, but we’ll just pass it through)
    print("📦 Preprocessing input...")
    return state

# --- Node 2: Gemini ---
def gemini_node(state: MultiNodeState) -> MultiNodeState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    try:
        response = llm.invoke(state["input"])
        state["raw_response"] = response.content
        return state
    except Exception as e:
        return {"input": state["input"], "raw_response": f"Error: {str(e)}", "final_output": ""}

# --- Node 3: Postprocessor ---
def postprocessor(state: MultiNodeState) -> MultiNodeState:
    print("🛠️ Postprocessing response...")
    state["final_output"] = state["raw_response"].strip()
    return state

# --- Build the Graph ---
builder = StateGraph(MultiNodeState)

# Add nodes
builder.add_node("input_handler", input_handler)
builder.add_node("gemini_node", gemini_node)
builder.add_node("postprocessor", postprocessor)

# Define edges
builder.set_entry_point("input_handler")
builder.add_edge("input_handler", "gemini_node")
builder.add_edge("gemini_node", "postprocessor")
builder.set_finish_point("postprocessor")

# Compile the graph
graph = builder.compile()

# --- CLI Loop to Run Agent ---
if __name__ == "__main__":
    while True:
        user_input = input("\n🧑 You: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        state: MultiNodeState = {
            "input": user_input,
            "raw_response": "",
            "final_output": ""
        }

        result = graph.invoke(state)
        print("🤖 Gemini:", result["final_output"])
