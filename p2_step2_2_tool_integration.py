from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import TypedDict, List, Literal
import os
import math
import datetime
import random

# Load Gemini API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define Tools
@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations. 
    Supports basic operations (+, -, *, /, **), sqrt, sin, cos, tan, log, etc.
    Example: "2 + 3 * 4" or "sqrt(16)" or "sin(3.14159/2)"
    """
    try:
        # Safe evaluation with math functions
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

@tool
def get_current_time() -> str:
    """Get the current date and time"""
    now = datetime.datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def web_search_simulator(query: str) -> str:
    """
    Simulate a web search (for demonstration purposes).
    In a real implementation, this would call actual search APIs.
    """
    # Simulated search results
    search_results = {
        "weather": "Today's weather: Sunny, 25°C with light breeze",
        "news": "Latest news: Technology sector shows growth in Q4",
        "python": "Python is a high-level programming language known for simplicity",
        "ai": "AI developments continue to advance with new LLM models",
        "stock": "Stock market: Mixed performance with tech stocks leading",
    }
    
    # Find relevant result
    query_lower = query.lower()
    for key, result in search_results.items():
        if key in query_lower:
            return f"Search result for '{query}': {result}"
    
    return f"Search result for '{query}': No specific information found in simulation"

# Define state schema
class ToolAgentState(TypedDict):
    messages: List[HumanMessage | AIMessage | ToolMessage]
    current_input: str
    response: str
    user_id: str
    tool_calls_made: List[str]

# Initialize store and tools
store = InMemoryStore()
tools = [calculator, get_current_time, web_search_simulator]

# Create LLM with tools
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.1
)
llm_with_tools = llm.bind_tools(tools)

# Tool execution node
def tool_execution_node(state: ToolAgentState) -> ToolAgentState:
    """Execute any tool calls made by the LLM"""
    messages = state["messages"]
    tool_calls_made = []
    
    # Get the last message (should be from LLM with tool calls)
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            # Find and execute the tool
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Execute the appropriate tool
            if tool_name == "calculator":
                result = calculator.invoke(tool_args)
            elif tool_name == "get_current_time":
                result = get_current_time.invoke(tool_args)
            elif tool_name == "web_search_simulator":
                result = web_search_simulator.invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"
            
            # Add tool result to messages
            tool_message = ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
            messages.append(tool_message)
            tool_calls_made.append(f"{tool_name}({tool_args})")
    
    return {
        "messages": messages,
        "current_input": state["current_input"],
        "response": state["response"],
        "user_id": state["user_id"],
        "tool_calls_made": tool_calls_made
    }

# Main conversation node
def conversation_node(state: ToolAgentState) -> ToolAgentState:
    """Main conversation node with tool calling capability"""
    current_input = state["current_input"]
    user_id = state["user_id"]
    
    try:
        # Get conversation history from store
        namespace = "messages"
        memory_key = f"conversation_{user_id}"
        
        stored_item = store.get(namespace, memory_key)
        messages = stored_item.value if stored_item else []
        
        # Add current input to message history
        messages.append(HumanMessage(content=current_input))
        
        # Get LLM response (may include tool calls)
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Store updated conversation history
        store.put(namespace, memory_key, messages)
        
        return {
            "messages": messages,
            "current_input": current_input,
            "response": response.content if response.content else "Processing tools...",
            "user_id": user_id,
            "tool_calls_made": []
        }
    except Exception as e:
        return {
            "messages": messages if 'messages' in locals() else [],
            "current_input": current_input,
            "response": f"❌ Error: {str(e)}",
            "user_id": user_id,
            "tool_calls_made": []
        }

# Final response node
def final_response_node(state: ToolAgentState) -> ToolAgentState:
    """Generate final response after tool execution"""
    messages = state["messages"]
    user_id = state["user_id"]
    
    try:
        # If tools were called, get final response from LLM
        if state["tool_calls_made"]:
            final_response = llm.invoke(messages)
            messages.append(final_response)
            
            # Store updated conversation
            namespace = "messages"
            memory_key = f"conversation_{user_id}"
            store.put(namespace, memory_key, messages)
            
            return {
                "messages": messages,
                "current_input": state["current_input"],
                "response": final_response.content,
                "user_id": user_id,
                "tool_calls_made": state["tool_calls_made"]
            }
        else:
            # No tools called, return original response
            return state
    except Exception as e:
        return {
            "messages": messages,
            "current_input": state["current_input"],
            "response": f"❌ Error in final response: {str(e)}",
            "user_id": user_id,
            "tool_calls_made": state["tool_calls_made"]
        }

# Conditional edge function
def should_use_tools(state: ToolAgentState) -> Literal["tools", "final"]:
    """Determine if we need to execute tools"""
    messages = state["messages"]
    if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
        return "tools"
    return "final"

# Build the graph
builder = StateGraph(ToolAgentState)

# Add nodes
builder.add_node("conversation", conversation_node)
builder.add_node("tools", tool_execution_node)  
builder.add_node("final", final_response_node)

# Add edges
builder.set_entry_point("conversation")
builder.add_conditional_edges(
    "conversation",
    should_use_tools,
    {"tools": "tools", "final": "final"}
)
builder.add_edge("tools", "final")
builder.set_finish_point("final")

# Compile graph
graph = builder.compile(store=store)

# Utility functions
def show_conversation_history(user_id: str):
    """Display conversation history"""
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
        if isinstance(msg, HumanMessage):
            print(f"{i}. 🧑 Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"{i}. 🤖 AI: {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"{i}. 🔧 Tool: {msg.content}")
    print()

def clear_conversation_history(user_id: str):
    """Clear conversation history"""
    namespace = "messages"
    memory_key = f"conversation_{user_id}"
    store.put(namespace, memory_key, [])
    print("🧹 Conversation history cleared!")

# Main execution
if __name__ == "__main__":
    print("🔧 Multi-Tool Agent with Memory")
    print("Available tools:")
    print("  🧮 Calculator - Perform mathematical calculations")
    print("  ⏰ Current Time - Get current date and time")
    print("  🔍 Web Search - Simulate web search")
    print("\nCommands:")
    print("  'history' - Show conversation history")
    print("  'clear' - Clear conversation history")
    print("  'exit' or 'quit' - End conversation")
    print("\nExample queries:")
    print("  'Calculate 15 * 8 + sqrt(64)'")
    print("  'What time is it?'")
    print("  'Search for Python programming'")
    print("  'My name is John, calculate 5 + 3, then tell me my name'")
    print()
    
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
        
        # Create state for this interaction
        state = {
            "messages": [],
            "current_input": user_input,
            "response": "",
            "user_id": user_id,
            "tool_calls_made": []
        }
        
        # Process through graph
        result = graph.invoke(state)
        
        # Print response
        print("🤖 Gemini:", result["response"])
        
        # Show tool usage if any
        if result["tool_calls_made"]:
            print(f"🔧 Tools used: {', '.join(result['tool_calls_made'])}")
        
        print()
