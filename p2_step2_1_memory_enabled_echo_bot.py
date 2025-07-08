from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import TypedDict, List
import os

# Load Gemini API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define state schema with conversation history
class ConversationState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    current_input: str
    response: str

# Node function with memory
def conversation_node(state: ConversationState) -> ConversationState:
    current_input = state["current_input"]
    messages = state.get("messages", [])
    
    # Add current input to message history
    messages.append(HumanMessage(content=current_input))
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    
    try:
        # Pass full conversation history to LLM
        response = llm.invoke(messages)
        
        # Add AI response to message history
        messages.append(AIMessage(content=response.content))
        
        # Update state
        return {
            "messages": messages,
            "current_input": current_input,
            "response": response.content
        }
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return {
            "messages": messages,
            "current_input": current_input,
            "response": error_msg
        }

# Build the LangGraph
builder = StateGraph(ConversationState)
builder.add_node("conversation", conversation_node)
builder.set_entry_point("conversation")
builder.set_finish_point("conversation")
graph = builder.compile()

# Run with persistent memory
if __name__ == "__main__":
    print("🧠 Memory-Enabled Conversational Agent")
    print("Type 'history' to see conversation history")
    print("Type 'clear' to clear conversation history")
    print("Type 'exit' or 'quit' to end\n")
    
    # Initialize persistent state
    persistent_state = {
        "messages": [],
        "current_input": "",
        "response": ""
    }
    
    while True:
        user_input = input("🧑 You: ")
        
        if user_input.lower() in {"exit", "quit"}:
            break
        elif user_input.lower() == "clear":
            persistent_state["messages"] = []
            print("🧹 Conversation history cleared!")
            continue
        elif user_input.lower() == "history":
            print("\n📜 Conversation History:")
            for i, msg in enumerate(persistent_state["messages"], 1):
                role = "🧑 Human" if isinstance(msg, HumanMessage) else "🤖 AI"
                print(f"{i}. {role}: {msg.content}")
            print()
            continue
        
        # Update state with new input
        persistent_state["current_input"] = user_input
        
        # Process through graph
        result = graph.invoke(persistent_state)
        
        # Update persistent state with result
        persistent_state = result
        
        # Print response
        print("🤖 Gemini:", result["response"])
        print(f"💾 Memory: {len(result['messages'])} messages stored\n")
