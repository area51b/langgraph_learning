from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Literal
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- State ---
class BranchState(TypedDict):
    input: str
    type: Literal["question", "command"]
    result: str
    final_output: str

# --- Router: Decides path
def classify_input(state: BranchState) -> str:
    text = state["input"].strip()
    if text.endswith("?"):
        return "answer_question"
    else:
        return "execute_command"

# --- Node A: Answer question using Gemini
def answer_question(state: BranchState) -> BranchState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    response = llm.invoke(state["input"])
    state["type"] = "question"
    state["result"] = response.content
    return state

# --- Node B: Handle command
def execute_command(state: BranchState) -> BranchState:
    state["type"] = "command"
    state["result"] = f"🛠️ Command acknowledged: {state['input']}"
    return state

# --- Node C: Postprocessing
def postprocessor(state: BranchState) -> BranchState:
    state["final_output"] = state["result"].strip()
    return state

# --- Build the Graph ---
builder = StateGraph(BranchState)

builder.add_node("router", lambda state: state)  # placeholder node
builder.add_node("answer_question", answer_question)
builder.add_node("execute_command", execute_command)
builder.add_node("postprocessor", postprocessor)

# Start at dummy router, which immediately branches
builder.set_entry_point("router")
builder.add_conditional_edges("router", classify_input)

# Merge branches into postprocessor
builder.add_edge("answer_question", "postprocessor")
builder.add_edge("execute_command", "postprocessor")

builder.set_finish_point("postprocessor")

graph = builder.compile()

# --- Run the Agent ---
if __name__ == "__main__":
    while True:
        text = input("\n🧑 You: ")
        if text.lower() in {"exit", "quit"}:
            break

        state: BranchState = {
            "input": text,
            "type": "command",      # default
            "result": "",
            "final_output": ""
        }

        result = graph.invoke(state)
        print("🤖 Gemini:", result["final_output"])
