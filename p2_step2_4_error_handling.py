from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Literal, Optional
import os
import time
import random
from datetime import datetime

# Load Gemini API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Enhanced state schema with error handling
class RobustAgentState(TypedDict):
    input: str
    output: str
    error_count: int
    max_retries: int
    last_error: Optional[str]
    attempt_history: list[str]
    fallback_used: bool
    current_node: str
    has_success: bool  # Track if we've succeeded

# Simulate different types of errors for demonstration
class SimulatedError(Exception):
    pass

def primary_processing_node(state: RobustAgentState) -> RobustAgentState:
    """Primary node that might fail - simulates API calls or complex operations"""
    current_attempt = state["error_count"] + 1
    print(f"🔄 Attempting primary processing (attempt {current_attempt})")
    
    # Simulate random failures for demonstration (30% failure rate)
    if random.random() < 0.3:
        error_msg = f"Simulated API failure at {datetime.now().strftime('%H:%M:%S')}"
        print(f"❌ Primary processing failed: {error_msg}")
        
        # Update error state
        state["error_count"] += 1
        state["last_error"] = error_msg
        state["attempt_history"].append(f"Attempt {current_attempt}: Failed - {error_msg}")
        state["current_node"] = "primary_processing"
        state["has_success"] = False
        
        return state
    
    # Success case
    user_input = state["input"]
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    
    try:
        response = llm.invoke(f"Process this request professionally: {user_input}")
        state["output"] = f"✅ Primary Success: {response.content}"
        state["attempt_history"].append(f"Attempt {current_attempt}: Success")
        state["current_node"] = "primary_processing"
        state["has_success"] = True  # Mark as successful
        print("✅ Primary processing succeeded")
        return state
        
    except Exception as e:
        # Real API error
        error_msg = f"Real API error: {str(e)}"
        print(f"❌ Real API error: {error_msg}")
        
        state["error_count"] += 1
        state["last_error"] = error_msg
        state["attempt_history"].append(f"Attempt {current_attempt}: API Error - {error_msg}")
        state["current_node"] = "primary_processing"
        state["has_success"] = False
        
        return state

def route_decision(state: RobustAgentState) -> Literal["retry", "fallback", "final_output"]:
    """Routing function that contains all the decision logic"""
    print(f"🤔 Deciding next action... (errors: {state['error_count']}, max: {state['max_retries']}, success: {state['has_success']})")
    
    # If we have success, go to final output
    if state["has_success"]:
        print("✅ Success achieved, proceeding to final output")
        return "final_output"
    # If we have errors and haven't hit retry limit, retry
    elif state["error_count"] > 0 and state["error_count"] < state["max_retries"]:
        print(f"🔄 Will retry (attempt {state['error_count'] + 1}/{state['max_retries']})")
        return "retry"
    # If retry limit exceeded, use fallback
    elif state["error_count"] >= state["max_retries"]:
        print("🛡️ Max retries reached, using fallback")
        return "fallback"
    else:
        # First attempt (error_count = 0, no success yet) - shouldn't happen
        print("⚠️ Unexpected state, proceeding to final output")
        return "final_output"

def retry_node(state: RobustAgentState) -> RobustAgentState:
    """Retry the primary operation with exponential backoff"""
    print(f"🔄 Retry node: Preparing for attempt {state['error_count'] + 1}")
    
    # Exponential backoff (optional)
    delay = min(2 ** (state["error_count"] - 1), 8)  # Max 8 seconds, starts at 1 second
    print(f"⏱️ Waiting {delay} seconds before retry...")
    time.sleep(delay)
    
    # Don't increment error count here - let primary_processing handle it
    state["current_node"] = "retry"
    return state

def fallback_node(state: RobustAgentState) -> RobustAgentState:
    """Fallback mechanism when primary processing fails"""
    print("🛡️ Executing fallback processing")
    
    user_input = state["input"]
    
    # Simple fallback: basic string processing without LLM
    try:
        # Simulate a simpler, more reliable fallback
        fallback_response = f"Fallback response for: '{user_input}' (processed locally without AI)"
        
        state["output"] = f"⚠️ Fallback Used: {fallback_response}"
        state["fallback_used"] = True
        state["attempt_history"].append("Fallback processing succeeded")
        state["current_node"] = "fallback"
        
        print("✅ Fallback processing succeeded")
        return state
        
    except Exception as e:
        # Even fallback failed - this is rare but possible
        error_msg = f"Fallback failed: {str(e)}"
        print(f"❌ Fallback failed: {error_msg}")
        
        state["output"] = f"❌ Complete Failure: {error_msg}"
        state["attempt_history"].append(f"Fallback failed: {error_msg}")
        state["current_node"] = "fallback"
        
        return state

def final_output_node(state: RobustAgentState) -> RobustAgentState:
    """Final output node with summary of attempts"""
    print("📋 Generating final output with attempt summary")
    
    # Add attempt summary to output
    summary = f"\n\n📊 Execution Summary:\n"
    summary += f"• Total attempts: {len(state['attempt_history'])}\n"
    summary += f"• Errors encountered: {state['error_count']}\n"
    summary += f"• Fallback used: {'Yes' if state['fallback_used'] else 'No'}\n"
    summary += f"• Attempt history: {' → '.join(state['attempt_history'])}"
    
    if state["output"]:
        state["output"] += summary
    else:
        state["output"] = f"No output generated{summary}"
    
    state["current_node"] = "final_output"
    return state

# Build the robust graph with error handling
def build_robust_graph() -> StateGraph:
    """Build the graph with retry and fallback logic"""
    
    # Create the graph
    builder = StateGraph(RobustAgentState)
    
    # Add nodes
    builder.add_node("primary_processing", primary_processing_node)
    builder.add_node("retry", retry_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("final_output", final_output_node)
    
    # Define edges
    builder.set_entry_point("primary_processing")
    
    # From primary_processing, use conditional routing directly
    builder.add_conditional_edges(
        "primary_processing",
        route_decision,
        {
            "retry": "retry",
            "fallback": "fallback", 
            "final_output": "final_output"
        }
    )
    
    # From retry, go back to primary_processing (loopback)
    builder.add_edge("retry", "primary_processing")
    
    # From fallback, go to final_output
    builder.add_edge("fallback", "final_output")
    
    # End at final_output
    builder.add_edge("final_output", END)
    
    return builder.compile()

# Demo function
def demo_error_handling():
    """Demonstrate different error handling scenarios"""
    
    graph = build_robust_graph()
    
    test_cases = [
        {
            "input": "Hello, how are you?",
            "max_retries": 3,
            "description": "Basic greeting with 3 max retries"
        },
        {
            "input": "Explain quantum physics",
            "max_retries": 2,
            "description": "Complex request with 2 max retries"
        },
        {
            "input": "What's the weather like?",
            "max_retries": 1,
            "description": "Weather request with 1 max retry"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Test Case {i}: {test_case['description']}")
        print(f"{'='*60}")
        
        # Initialize state
        initial_state = {
            "input": test_case["input"],
            "output": "",
            "error_count": 0,
            "max_retries": test_case["max_retries"],
            "last_error": None,
            "attempt_history": [],
            "fallback_used": False,
            "current_node": "",
            "has_success": False
        }
        
        # Run the graph
        try:
            result = graph.invoke(initial_state)
            print(f"\n🎯 Final Result:")
            print(f"Input: {result['input']}")
            print(f"Output: {result['output']}")
            
        except Exception as e:
            print(f"❌ Graph execution failed: {str(e)}")
        
        print(f"\n⏱️ Waiting 2 seconds before next test...")
        time.sleep(2)

# Interactive mode
def interactive_mode():
    """Interactive mode for testing error handling"""
    
    graph = build_robust_graph()
    
    print("🤖 Robust Agent with Error Handling")
    print("Features: Retry policies, fallbacks, loopbacks")
    print("Commands: 'exit', 'quit' to stop, 'demo' for automated tests")
    print("-" * 50)
    
    while True:
        try:
            text = input("\n🧑 You: ")
            
            if text.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            
            if text.lower() == "demo":
                demo_error_handling()
                continue
            
            # Get retry settings
            try:
                max_retries = int(input("🔧 Max retries (default 3): ") or "3")
            except ValueError:
                max_retries = 3
            
            # Initialize state
            initial_state = {
                "input": text,
                "output": "",
                "error_count": 0,
                "max_retries": max_retries,
                "last_error": None,
                "attempt_history": [],
                "fallback_used": False,
                "current_node": "",
                "has_success": False
            }
            
            print(f"\n🚀 Processing with max {max_retries} retries...")
            
            # Run the graph
            result = graph.invoke(initial_state)
            
            print(f"\n🤖 Agent Response:")
            print(result["output"])
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Interactive mode")
    print("2. Demo mode (automated tests)")
    
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "2":
        demo_error_handling()
    else:
        interactive_mode()
