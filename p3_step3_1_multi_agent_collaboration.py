from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, List, Literal
import os

# Load Gemini API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define comprehensive state schema for multi-agent collaboration
class ResearchState(TypedDict):
    user_query: str
    research_plan: str
    research_results: str
    verification_status: str
    final_answer: str
    iterations: int
    max_iterations: int

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)

# AGENT 1: PLANNER - Creates research strategy
def planner_agent(state: ResearchState) -> ResearchState:
    """
    The Planner Agent analyzes the user query and creates a structured research plan.
    """
    user_query = state["user_query"]
    
    planner_prompt = f"""
    You are a Research Planner Agent. Your job is to create a detailed research plan.
    
    User Query: {user_query}
    
    Create a structured research plan with:
    1. Key topics to investigate
    2. Specific questions to answer
    3. Research approach/methodology
    4. Success criteria for verification
    
    Be concise but comprehensive. Format as a numbered list.
    """
    
    try:
        response = llm.invoke(planner_prompt)
        state["research_plan"] = response.content
        print("📋 PLANNER: Research plan created")
        return state
    except Exception as e:
        state["research_plan"] = f"❌ Planning Error: {str(e)}"
        return state

# AGENT 2: EXECUTOR - Conducts the research
def executor_agent(state: ResearchState) -> ResearchState:
    """
    The Executor Agent follows the research plan and generates results.
    """
    user_query = state["user_query"]
    research_plan = state["research_plan"]
    
    executor_prompt = f"""
    You are a Research Executor Agent. Follow the research plan to provide comprehensive answers.
    
    Original Query: {user_query}
    Research Plan: {research_plan}
    
    Execute the research plan step by step:
    - Address each point in the plan
    - Provide detailed, factual information
    - Use your knowledge base to answer thoroughly
    - Structure your response clearly
    
    Focus on being accurate and comprehensive.
    """
    
    try:
        response = llm.invoke(executor_prompt)
        state["research_results"] = response.content
        print("🔍 EXECUTOR: Research completed")
        return state
    except Exception as e:
        state["research_results"] = f"❌ Execution Error: {str(e)}"
        return state

# AGENT 3: VERIFIER - Checks quality and completeness
def verifier_agent(state: ResearchState) -> ResearchState:
    """
    The Verifier Agent evaluates the research results and determines if they're satisfactory.
    """
    user_query = state["user_query"]
    research_plan = state["research_plan"]
    research_results = state["research_results"]
    
    verifier_prompt = f"""
    You are a Research Verifier Agent. Evaluate the research results for quality and completeness.
    
    Original Query: {user_query}
    Research Plan: {research_plan}
    Research Results: {research_results}
    
    Evaluate:
    1. Does the research answer the original query?
    2. Are all points from the plan addressed?
    3. Is the information accurate and well-structured?
    4. Are there any gaps or missing information?
    
    Respond with either:
    - "APPROVED: [brief reason]" if the research is satisfactory
    - "NEEDS_IMPROVEMENT: [specific issues to address]" if it needs more work
    """
    
    try:
        response = llm.invoke(verifier_prompt)
        state["verification_status"] = response.content
        print("✅ VERIFIER: Results evaluated")
        return state
    except Exception as e:
        state["verification_status"] = f"❌ Verification Error: {str(e)}"
        return state

# FINALIZER - Prepares the final answer
def finalizer_agent(state: ResearchState) -> ResearchState:
    """
    The Finalizer Agent prepares the final polished answer for the user.
    """
    user_query = state["user_query"]
    research_results = state["research_results"]
    verification_status = state["verification_status"]
    
    finalizer_prompt = f"""
    You are a Finalizer Agent. Create a polished, user-friendly final answer.
    
    Original Query: {user_query}
    Research Results: {research_results}
    Verification: {verification_status}
    
    Create a final answer that:
    - Directly addresses the user's query
    - Is well-structured and easy to read
    - Includes key insights from the research
    - Is concise but comprehensive
    
    Format as a professional response.
    """
    
    try:
        response = llm.invoke(finalizer_prompt)
        state["final_answer"] = response.content
        print("📝 FINALIZER: Final answer prepared")
        return state
    except Exception as e:
        state["final_answer"] = f"❌ Finalizer Error: {str(e)}"
        return state

# CONDITIONAL LOGIC - Determines next step based on verification
def should_continue(state: ResearchState) -> Literal["continue", "finalize"]:
    """
    Decision function: Continue improving or finalize based on verification.
    """
    verification = state["verification_status"]
    iterations = state["iterations"]
    max_iterations = state["max_iterations"]
    
    # Check if we've hit max iterations
    if iterations >= max_iterations:
        print(f"🔄 Max iterations ({max_iterations}) reached - finalizing")
        return "finalize"
    
    # Check verification status
    if verification.startswith("APPROVED"):
        print("✅ Research approved - finalizing")
        return "finalize"
    elif verification.startswith("NEEDS_IMPROVEMENT"):
        print("🔄 Research needs improvement - continuing")
        return "continue"
    else:
        print("⚠️ Unclear verification - finalizing")
        return "finalize"

# ITERATION COUNTER - Tracks improvement cycles
def increment_iteration(state: ResearchState) -> ResearchState:
    """
    Increment iteration counter for improvement cycles.
    """
    state["iterations"] += 1
    print(f"🔄 Iteration {state['iterations']}/{state['max_iterations']}")
    return state

# BUILD THE MULTI-AGENT GRAPH
def build_research_graph():
    """
    Build the multi-agent research graph with conditional flows.
    """
    # Create the StateGraph
    builder = StateGraph(ResearchState)
    
    # Add all agent nodes
    builder.add_node("planner", planner_agent)
    builder.add_node("executor", executor_agent)
    builder.add_node("verifier", verifier_agent)
    builder.add_node("increment", increment_iteration)
    builder.add_node("finalizer", finalizer_agent)
    
    # Define the flow
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "verifier")
    builder.add_edge("verifier", "increment")
    
    # Conditional branching after verification
    builder.add_conditional_edges(
        "increment",
        should_continue,
        {
            "continue": "executor",  # Go back to executor for improvement
            "finalize": "finalizer"  # Move to finalizer
        }
    )
    
    builder.add_edge("finalizer", END)
    
    # Compile the graph
    return builder.compile()

# MAIN EXECUTION
def main():
    """
    Main function to run the multi-agent research system.
    """
    # Build the graph
    graph = build_research_graph()
    
    print("🚀 Multi-Agent Research Assistant Ready!")
    print("This system uses specialized agents: Planner → Executor → Verifier → Finalizer")
    print("The system can iterate to improve results automatically.\n")
    
    while True:
        user_query = input("🧑 Enter your research question (or 'exit' to quit): ")
        
        if user_query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        
        # Initialize state
        initial_state = {
            "user_query": user_query,
            "research_plan": "",
            "research_results": "",
            "verification_status": "",
            "final_answer": "",
            "iterations": 0,
            "max_iterations": 2  # Allow up to 2 improvement cycles
        }
        
        print(f"\n🔍 Processing: {user_query}")
        print("=" * 50)
        
        try:
            # Run the multi-agent system
            result = graph.invoke(initial_state)
            
            # Display results
            print("\n" + "=" * 50)
            print("📋 RESEARCH PLAN:")
            print(result["research_plan"])
            print("\n" + "-" * 30)
            print("✅ VERIFICATION:")
            print(result["verification_status"])
            print("\n" + "-" * 30)
            print("📝 FINAL ANSWER:")
            print(result["final_answer"])
            print("\n" + "=" * 50)
            
        except Exception as e:
            print(f"❌ System Error: {str(e)}")

if __name__ == "__main__":
    main()
