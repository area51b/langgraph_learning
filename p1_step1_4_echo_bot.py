from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
import os

# Load Gemini API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define proper state schema
class EchoState(TypedDict):
    input: str
    output: str

# Node function
def echo_node(state: EchoState) -> EchoState:
    user_input = state["input"]
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    try:
        response = llm.invoke(user_input)
        state["output"] = response.content
        return state
    except Exception as e:
        return {"input": user_input, "output": f"❌ Error: {str(e)}"}

# Build the LangGraph
builder = StateGraph(EchoState)
builder.add_node("echo", echo_node)
builder.set_entry_point("echo")
builder.set_finish_point("echo")
graph = builder.compile()

# Run
if __name__ == "__main__":
    while True:
        text = input("🧑 You: ")
        if text.lower() in {"exit", "quit"}:
            break
        state = {"input": text}
        result = graph.invoke(state)
        # Print the output
        print("🤖 Gemini:", result["output"])
