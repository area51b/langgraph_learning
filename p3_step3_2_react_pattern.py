# Phase 3.2: ReAct Pattern in LangGraph
# Concept: ReAct (Reasoning + Acting) is a powerful pattern where the agent:

# 1. Reasons about what to do next
# 2. Acts by calling tools or taking actions
# 3. Observes the results
# 4. Reflects and decides whether to continue or finish

# This creates a loop: Reason → Act → Observe → Reflect → Reason...
# Key Learning Points:

# 1. Implement reasoning steps before actions
# 2. Tool calling with reflection
# 3. Looping until task completion
# 4. Error handling with retry logic


from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any
import os
import json
import math
import requests

# Load Gemini API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define ReAct State Schema
class ReActState(TypedDict):
    input: str
    thought: str
    action: str
    action_input: str
    observation: str
    final_answer: str
    step_count: int
    max_steps: int
    finished: bool
    tools_used: List[str]  # Track which tools were used

# Tool definitions
def calculator_tool(expression: str) -> str:
    """Calculate mathematical expressions safely"""
    try:
        # Simple eval for basic math - in production, use a proper math parser
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

def web_search_tool(query: str) -> str:
    """Mock web search tool - replace with real API in production"""
    mock_results = {
        "weather": "Current weather in New York: 72°F, partly cloudy",
        "python": "Python is a high-level programming language known for its simplicity",
        "langgraph": "LangGraph is a framework for building stateful, multi-agent applications"
    }
    
    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return value
    return f"Search results for '{query}': No specific results found"

# Available tools registry
TOOLS = {
    "calculator": calculator_tool,
    "web_search": web_search_tool
}

def reasoning_node(state: ReActState) -> ReActState:
    """Agent thinks about what to do next"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    
    # Build context from previous steps
    context = f"""
You are a helpful assistant that uses the ReAct pattern (Reasoning + Acting).
You have access to these tools: {list(TOOLS.keys())}

Task: {state['input']}
Current step: {state['step_count']}

Previous steps:
"""
    
    if state.get('thought'):
        context += f"Thought: {state['thought']}\n"
    if state.get('action'):
        context += f"Action: {state['action']}\n"
    if state.get('observation'):
        context += f"Observation: {state['observation']}\n"
    
    context += """
Now think step by step about what to do next. You should:
1. Reason about the current situation
2. Decide if you need to use a tool or if you can provide a final answer
3. If using a tool, specify which tool and what input to give it

Respond in this format:
Thought: [Your reasoning about what to do next]
Action: [tool_name OR "final_answer"]
Action Input: [input for the tool OR your final answer]
"""
    
    try:
        response = llm.invoke(context)
        content = response.content
        
        # Parse the response
        thought = ""
        action = ""
        action_input = ""
        
        for line in content.split('\n'):
            if line.startswith('Thought:'):
                thought = line.replace('Thought:', '').strip()
            elif line.startswith('Action:'):
                action = line.replace('Action:', '').strip()
            elif line.startswith('Action Input:'):
                action_input = line.replace('Action Input:', '').strip()
        
        state['thought'] = thought
        state['action'] = action
        state['action_input'] = action_input
        state['step_count'] += 1
        
        return state
        
    except Exception as e:
        state['thought'] = f"Error in reasoning: {str(e)}"
        state['action'] = "final_answer"
        state['action_input'] = "I encountered an error while processing your request."
        return state

def action_node(state: ReActState) -> ReActState:
    """Execute the planned action"""
    action = state['action']
    action_input = state['action_input']
    
    if action == "final_answer":
        state['final_answer'] = action_input
        state['finished'] = True
        return state
    
    # Execute tool
    if action in TOOLS:
        try:
            result = TOOLS[action](action_input)
            state['observation'] = result
            # Track tool usage
            if action not in state['tools_used']:
                state['tools_used'].append(action)
        except Exception as e:
            state['observation'] = f"Tool error: {str(e)}"
    else:
        state['observation'] = f"Unknown tool: {action}"
    
    return state

def should_continue(state: ReActState) -> str:
    """Decide whether to continue or finish"""
    if state['finished']:
        return "end"
    if state['step_count'] >= state['max_steps']:
        # Force finish if max steps reached
        state['final_answer'] = "I've reached the maximum number of steps. Based on my analysis so far, I cannot complete this task fully."
        return "end"
    return "continue"

# Build the ReAct Graph
def create_react_agent():
    builder = StateGraph(ReActState)
    
    # Add nodes
    builder.add_node("reasoning", reasoning_node)
    builder.add_node("action", action_node)
    
    # Add edges
    builder.set_entry_point("reasoning")
    builder.add_edge("reasoning", "action")
    builder.add_conditional_edges(
        "action",
        should_continue,
        {
            "continue": "reasoning",
            "end": END
        }
    )
    
    return builder.compile()

# Test the agent
if __name__ == "__main__":
    agent = create_react_agent()
    
    print("🤖 ReAct Agent Ready!")
    print("Try: 'Calculate 15 * 23 + 45'")
    print("Or: 'What is Python and then calculate 2^8'")
    print("Type 'exit' to quit\n")
    
    while True:
        user_input = input("🧑 You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        
        # Initial state
        initial_state = {
            "input": user_input,
            "thought": "",
            "action": "",
            "action_input": "",
            "observation": "",
            "final_answer": "",
            "step_count": 0,
            "max_steps": 5,
            "finished": False,
            "tools_used": []  # Initialize empty tools list
        }
        
        print("\n🔄 ReAct Process:")
        print("-" * 50)
        
        # Run the agent with step-by-step streaming
        current_state = initial_state
        
        for step in range(initial_state["max_steps"]):
            print(f"\n🔄 Step {step + 1}:")
            
            # Reasoning step
            current_state = reasoning_node(current_state)
            print(f"💭 Thought: {current_state['thought']}")
            print(f"🎯 Planned Action: {current_state['action']}")
            print(f"📥 Action Input: {current_state['action_input']}")
            
            # Action step
            if current_state['action'] == "final_answer":
                print(f"✅ Final Answer: {current_state['action_input']}")
                break
            elif current_state['action'] in TOOLS:
                print(f"🔧 Using Tool: {current_state['action']}")
                current_state = action_node(current_state)
                print(f"📤 Tool Result: {current_state['observation']}")
            else:
                current_state = action_node(current_state)
                print(f"❌ Tool Error: {current_state['observation']}")
            
            # Check if we should continue
            if current_state['finished'] or current_state['step_count'] >= current_state['max_steps']:
                if not current_state['finished']:
                    print(f"⏰ Reached max steps ({current_state['max_steps']})")
                break
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"   Tools Used: {current_state['tools_used'] if current_state['tools_used'] else 'None'}")
        print(f"   Total Steps: {current_state['step_count']}")
        print("-" * 50)
        print()


# Example Test Queries:
# ---------------------
# 🎯 Queries That Trigger CALCULATOR:

# "What is 15% of 250?"
# "Calculate the area of a circle with radius 5"
# "How much is 1200 divided by 8?"
# "What's 25 squared plus 30?"
# "If I save $50 per month, how much will I have in 2 years?"

# 🎯 Queries That Trigger WEB_SEARCH:

# "What's the latest news about Bitcoin?"
# "Tell me about current weather conditions"
# "What are the latest developments in AI?"
# "Search for recent stock market performance"
# "Find current information about climate change"
# "What's happening with space exploration?"

# 🎯 Queries That Trigger MULTIPLE TOOLS:

# "Search for Bitcoin price and calculate 10% of the current value"
# "Find AI industry size and calculate its growth rate"
# "Get weather info and calculate temperature in Celsius from Fahrenheit"
