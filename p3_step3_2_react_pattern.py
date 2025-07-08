from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from typing import TypedDict, List, Literal, Annotated
import operator
import json
import requests
import math
import os

# Load Gemini API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define ReAct state schema
class ReActState(TypedDict):
    query: str
    thoughts: List[str]
    actions: List[str]
    observations: List[str]
    final_answer: str
    iteration: int
    max_iterations: int
    tools_used: List[str]

# Initialize LLM with function calling capability
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.3,
    max_tokens=2048
)

# =================================
# DEFINE CUSTOM TOOLS
# =================================

def web_search_tool(query: str) -> str:
    """
    Enhanced mock web search tool with more diverse results
    """
    # More comprehensive mock results that will trigger on specific keywords
    mock_results = {
        "weather": "Current weather: Sunny, 72°F with clear skies. Perfect for outdoor activities.",
        "stock": "Stock market today: S&P 500 up 1.2%, Nasdaq up 0.8%, Dow Jones up 0.5%.",
        "news": "Breaking news: Technology sector showing strong growth this quarter.",
        "python": "Python is the most popular programming language in 2024, used by 68% of developers.",
        "ai": "AI industry worth $184 billion in 2024, expected to reach $826 billion by 2030.",
        "climate": "Climate change: Global temperatures have risen 1.1°C since pre-industrial times.",
        "bitcoin": "Bitcoin price: $45,000 USD, up 3% in the last 24 hours.",
        "covid": "COVID-19 update: New variant shows decreased severity, vaccination rates at 70%.",
        "election": "2024 election results: Voter turnout was 65%, highest in recent history.",
        "economy": "Economic outlook: GDP growth projected at 2.8% this year with low unemployment.",
        "space": "Space exploration: Mars mission planned for 2026, SpaceX preparing for launch.",
        "quantum": "Quantum computing breakthrough: IBM announces 1000-qubit processor.",
    }
    
    # Try to find relevant information
    query_lower = query.lower()
    for key, value in mock_results.items():
        if key in query_lower:
            return f"🔍 Search results for '{query}': {value}"
    
    # Generic fallback
    return f"🔍 Search results for '{query}': Recent information about {query} shows ongoing developments in this area."

def calculator_tool(expression: str) -> str:
    """
    Safe calculator tool for mathematical expressions
    """
    try:
        # Safe evaluation of mathematical expressions
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("_")
        }
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Calculation result: {expression} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

def search_memory_tool(keyword: str) -> str:
    """
    Search through previous thoughts and observations
    """
    # This would search through conversation history in a real implementation
    return f"Memory search for '{keyword}': This is a mock memory search result."

# =================================
# REACT AGENT IMPLEMENTATION
# =================================

class ReActAgent:
    def __init__(self):
        self.llm = llm
        self.tools = {
            "web_search": web_search_tool,
            "calculator": calculator_tool,
            "search_memory": search_memory_tool
        }
    
    def think(self, state: ReActState) -> ReActState:
        """
        THINKING step: Agent reasons about the problem and decides what to do next
        """
        query = state["query"]
        thoughts = state["thoughts"]
        actions = state["actions"]
        observations = state["observations"]
        iteration = state["iteration"]
        
        # Build context from previous iterations
        context = self._build_context(thoughts, actions, observations)
        
        thinking_prompt = f"""
        You are a ReAct (Reasoning and Acting) agent. You must think step by step and use tools when needed.
        
        Query: {query}
        Current iteration: {iteration}
        
        Previous context:
        {context}
        
        IMPORTANT: You MUST use tools to get information. Don't rely on your training data.
        
        Available tools (use exact format):
        - web_search("your search query") - Get current information from web
        - calculator("mathematical expression") - Calculate numbers 
        - search_memory("keyword") - Search previous conversation
        
        Decision criteria:
        - If query asks for current/recent information → use web_search
        - If query involves numbers/calculations → use calculator  
        - If query references previous conversation → use search_memory
        - Only use "answer" if you have sufficient information from tools
        
        EXAMPLES:
        - For "what's the weather" → web_search("weather today")
        - For "what is 15% of 200" → calculator("0.15 * 200")  
        - For "tell me about bitcoin" → web_search("bitcoin latest news")
        
        Think about:
        1. What type of information is needed?
        2. Which tool can provide this information?
        3. What specific query/expression to use?
        
        Response format (EXACT):
        THOUGHT: [your reasoning about what tool to use and why]
        NEXT_ACTION: [exactly one of: web_search("query"), calculator("expression"), search_memory("keyword"), or "answer"]
        """
        
        try:
            response = self.llm.invoke(thinking_prompt)
            thinking_result = response.content
            
            # Parse the thinking result
            if "THOUGHT:" in thinking_result and "NEXT_ACTION:" in thinking_result:
                thought_part = thinking_result.split("NEXT_ACTION:")[0].replace("THOUGHT:", "").strip()
                action_part = thinking_result.split("NEXT_ACTION:")[1].strip()
                
                state["thoughts"].append(thought_part)
                
                return state, action_part
            else:
                # Fallback if format is not followed
                state["thoughts"].append(thinking_result)
                return state, "answer"
                
        except Exception as e:
            state["thoughts"].append(f"Thinking error: {str(e)}")
            return state, "answer"
    
    def act(self, state: ReActState, action: str) -> ReActState:
        """
        ACTING step: Agent uses tools or provides final answer
        """
        if action == "answer":
            return self._provide_final_answer(state)
        
        # Parse tool call - improved parsing
        if action == "answer":
            return self._provide_final_answer(state)
        
        # Better parsing for tool calls
        if "(" in action and ")" in action:
            try:
                # Extract tool name and argument
                tool_name = action.split("(")[0].strip()
                # Get everything between first ( and last )
                args_part = action[action.find("(")+1:action.rfind(")")].strip()
                # Remove quotes if present
                tool_args = args_part.strip('"\'')
                
                if tool_name in self.tools:
                    result = self.tools[tool_name](tool_args)
                    state["actions"].append(f"{tool_name}({tool_args})")
                    state["observations"].append(result)
                    state["tools_used"].append(tool_name)
                    print(f"🔧 ACTION: {tool_name}({tool_args})")
                    print(f"📊 OBSERVATION: {result}")
                else:
                    observation = f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}"
                    state["observations"].append(observation)
                    print(f"❌ UNKNOWN TOOL: {observation}")
            except Exception as e:
                observation = f"Tool parsing error: {str(e)}"
                state["observations"].append(observation)
                print(f"❌ PARSING ERROR: {observation}")
        else:
            observation = f"Invalid action format: {action}. Use tool_name('argument') or 'answer'"
            state["observations"].append(observation)
            print(f"❌ INVALID ACTION: {observation}")
        
        return state
    
    def _provide_final_answer(self, state: ReActState) -> ReActState:
        """
        Provide the final answer based on all thinking and observations
        """
        query = state["query"]
        thoughts = state["thoughts"]
        observations = state["observations"]
        
        final_prompt = f"""
        Based on your thinking process and observations, provide a final answer.
        
        Original Query: {query}
        
        Your thinking process:
        {self._format_list(thoughts)}
        
        Your observations:
        {self._format_list(observations)}
        
        Provide a comprehensive, well-structured final answer that:
        1. Directly addresses the original query
        2. Incorporates insights from your research/calculations
        3. Is clear and easy to understand
        4. Cites the tools/information used
        
        Final Answer:
        """
        
        try:
            response = self.llm.invoke(final_prompt)
            state["final_answer"] = response.content
            print("📝 FINAL ANSWER GENERATED")
        except Exception as e:
            state["final_answer"] = f"Error generating final answer: {str(e)}"
        
        return state
    
    def _build_context(self, thoughts: List[str], actions: List[str], observations: List[str]) -> str:
        """
        Build context from previous iterations
        """
        context = ""
        for i in range(len(thoughts)):
            context += f"Iteration {i+1}:\n"
            context += f"  Thought: {thoughts[i]}\n"
            if i < len(actions):
                context += f"  Action: {actions[i]}\n"
            if i < len(observations):
                context += f"  Observation: {observations[i]}\n"
            context += "\n"
        return context
    
    def _format_list(self, items: List[str]) -> str:
        """
        Format a list of items for display
        """
        if not items:
            return "None"
        return "\n".join([f"- {item}" for item in items])

# =================================
# LANGGRAPH IMPLEMENTATION
# =================================

agent = ReActAgent()

def thinking_node(state: ReActState) -> ReActState:
    """
    Node for the thinking process
    """
    print(f"🤔 THINKING (Iteration {state['iteration']})")
    updated_state, next_action = agent.think(state)
    updated_state["next_action"] = next_action  # Store for decision making
    return updated_state

def acting_node(state: ReActState) -> ReActState:
    """
    Node for the acting process
    """
    print(f"🎯 ACTING")
    next_action = state.get("next_action", "answer")
    return agent.act(state, next_action)

def increment_iteration(state: ReActState) -> ReActState:
    """
    Increment iteration counter
    """
    state["iteration"] += 1
    return state

def should_continue(state: ReActState) -> Literal["continue", "finish"]:
    """
    Decide whether to continue the ReAct loop or finish
    """
    # Check if we have a final answer
    if state.get("final_answer"):
        return "finish"
    
    # Check iteration limit
    if state["iteration"] >= state["max_iterations"]:
        print(f"🔄 Max iterations ({state['max_iterations']}) reached")
        return "finish"
    
    # Check if the last action was "answer"
    if state.get("next_action") == "answer":
        return "finish"
    
    return "continue"

def build_react_graph():
    """
    Build the ReAct pattern graph
    """
    builder = StateGraph(ReActState)
    
    # Add nodes
    builder.add_node("think", thinking_node)
    builder.add_node("act", acting_node)
    builder.add_node("increment", increment_iteration)
    
    # Define flow
    builder.add_edge(START, "think")
    builder.add_edge("think", "act")
    builder.add_edge("act", "increment")
    
    # Conditional branching
    builder.add_conditional_edges(
        "increment",
        should_continue,
        {
            "continue": "think",
            "finish": END
        }
    )
    
    return builder.compile()

# =================================
# MAIN EXECUTION
# =================================

def main():
    """
    Main function to run the ReAct agent
    """
    graph = build_react_graph()
    
    print("🚀 ReAct Agent Ready!")
    print("This agent uses the Reasoning + Acting pattern with tools.")
    print("Available tools: web_search, calculator, search_memory\n")
    
    while True:
        query = input("🧑 Enter your query (or 'exit' to quit): ")
        
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        
        # Initialize state
        initial_state = {
            "query": query,
            "thoughts": [],
            "actions": [],
            "observations": [],
            "final_answer": "",
            "iteration": 1,
            "max_iterations": 5,
            "tools_used": []
        }
        
        print(f"\n🔍 Processing: {query}")
        print("=" * 60)
        
        try:
            # Run the ReAct agent
            result = graph.invoke(initial_state)
            
            # Display results
            print("\n" + "=" * 60)
            print("📊 REACT PROCESS SUMMARY:")
            print(f"Total iterations: {result['iteration'] - 1}")
            print(f"Tools used: {', '.join(set(result['tools_used'])) if result['tools_used'] else 'None'}")
            
            print("\n🧠 THINKING PROCESS:")
            for i, thought in enumerate(result["thoughts"], 1):
                print(f"{i}. {thought}")
            
            print("\n🎯 ACTIONS TAKEN:")
            for i, action in enumerate(result["actions"], 1):
                print(f"{i}. {action}")
            
            print("\n📝 FINAL ANSWER:")
            print(result["final_answer"])
            print("\n" + "=" * 60)
            
        except Exception as e:
            print(f"❌ System Error: {str(e)}")

if __name__ == "__main__":
    main()



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
