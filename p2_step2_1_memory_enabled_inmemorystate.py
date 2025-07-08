from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import TypedDict, List
import os

# Load Gemini API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define state schema
class ConversationState(TypedDict):
    current_input: str
    response: str
    user_id: str

# Initialize InMemoryStore
store = InMemoryStore()

# Node function with InMemoryStore
def conversation_node(state: ConversationState) -> ConversationState:
    current_input = state["current_input"]
    user_id = state["user_id"]
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    
    try:
        # Get conversation history from store
        namespace = "messages"
        memory_key = f"conversation_{user_id}"
        
        # Get messages using correct API
        try:
            stored_item = store.get(namespace, memory_key)
            # Extract value from Item object
            messages = stored_item.value if stored_item else []
        except Exception:
            # If key doesn't exist, start with empty list
            messages = []
        
        # Add current input to message history
        messages.append(HumanMessage(content=current_input))
        
        # Pass full conversation history to LLM
        response = llm.invoke(messages)
        
        # Add AI response to message history
        messages.append(AIMessage(content=response.content))
        
        # Store updated conversation history
        store.put(namespace, memory_key, messages)
        
        # Update state
        return {
            "current_input": current_input,
            "response": response.content,
            "user_id": user_id
        }
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return {
            "current_input": current_input,
            "response": error_msg,
            "user_id": user_id
        }

# Build the LangGraph with store
builder = StateGraph(ConversationState)
builder.add_node("conversation", conversation_node)
builder.set_entry_point("conversation")
builder.set_finish_point("conversation")

# Compile graph with store
graph = builder.compile(store=store)

# Utility functions for memory management
def show_conversation_history(user_id: str):
    """Display conversation history for a user"""
    namespace = "messages"
    memory_key = f"conversation_{user_id}"
    
    try:
        stored_item = store.get(namespace, memory_key)
        messages = stored_item.value if stored_item else []
        if not messages:
            print("📭 No conversation history found")
            return
    except Exception:
        print("📭 No conversation history found")
        return
    
    print("\n📜 Conversation History:")
    for i, msg in enumerate(messages, 1):
        role = "🧑 Human" if isinstance(msg, HumanMessage) else "🤖 AI"
        print(f"{i}. {role}: {msg.content}")
    print()

def clear_conversation_history(user_id: str):
    """Clear conversation history for a user"""
    namespace = "messages"
    memory_key = f"conversation_{user_id}"
    store.put(namespace, memory_key, [])
    print("🧹 Conversation history cleared!")

def get_message_count(user_id: str) -> int:
    """Get count of messages for a user"""
    namespace = "messages"
    memory_key = f"conversation_{user_id}"
    
    try:
        stored_item = store.get(namespace, memory_key)
        messages = stored_item.value if stored_item else []
        return len(messages) if messages else 0
    except Exception:
        return 0

# Run with InMemoryStore
if __name__ == "__main__":
    print("🧠 LangGraph InMemoryStore Conversational Agent")
    print("Commands:")
    print("  'history' - Show conversation history")
    print("  'clear' - Clear conversation history")
    print("  'stats' - Show store statistics")
    print("  'exit' or 'quit' - End conversation")
    print()
    
    # Set user ID (in real app, this would be dynamic)
    user_id = "user_001"
    
    while True:
        user_input = input("🧑 You: ")
        
        if user_input.lower() in {"exit", "quit"}:
            break
        elif user_input.lower() == "clear":
            clear_conversation_history(user_id)
            continue
        elif user_input.lower() == "history":
            show_conversation_history(user_id)
            continue
        elif user_input.lower() == "stats":
            message_count = get_message_count(user_id)
            print(f"💾 Memory Store Statistics:")
            print(f"  Messages stored: {message_count}")
            print(f"  User ID: {user_id}")
            print()
            continue
        
        # Create state for this interaction
        state = {
            "current_input": user_input,
            "response": "",
            "user_id": user_id
        }
        
        # Process through graph
        result = graph.invoke(state)
        
        # Print response
        print("🤖 Gemini:", result["response"])
        
        # Show memory info
        message_count = get_message_count(user_id)
        print(f"💾 Memory: {message_count} messages stored\n")
