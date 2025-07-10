from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, List
import os

# Load Gemini API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define state schema for recursive essay writing
class EssayState(TypedDict):
    topic: str
    current_essay: str
    critique: str
    iteration_count: int
    max_iterations: int
    improvement_history: List[str]
    is_satisfied: bool

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

def write_initial_essay(state: EssayState) -> EssayState:
    """Generate the first draft of the essay"""
    print(f"📝 Writing initial essay (Iteration {state['iteration_count'] + 1})")
    
    prompt = f"""
    Write a well-structured essay on the topic: "{state['topic']}"
    
    Requirements:
    - 3-4 paragraphs
    - Clear introduction, body, and conclusion
    - Engaging and informative content
    - Proper flow and transitions
    """
    
    try:
        response = llm.invoke(prompt)
        state["current_essay"] = response.content
        state["iteration_count"] = 1
        state["improvement_history"].append(f"Initial essay written")
        print("✅ Initial essay completed")
        return state
    except Exception as e:
        state["current_essay"] = f"❌ Error writing essay: {str(e)}"
        state["is_satisfied"] = True  # Stop on error
        return state

def critique_essay(state: EssayState) -> EssayState:
    """Analyze the current essay and provide detailed critique"""
    print(f"🔍 Critiquing essay (Iteration {state['iteration_count']})")
    
    prompt = f"""
    Analyze this essay and provide constructive criticism:
    
    ESSAY:
    {state['current_essay']}
    
    Please evaluate:
    1. Content quality and depth
    2. Structure and organization
    3. Writing style and clarity
    4. Engagement and flow
    5. Areas for improvement
    
    Provide specific suggestions for improvement. If the essay is already excellent, say "SATISFIED".
    """
    
    try:
        response = llm.invoke(prompt)
        state["critique"] = response.content
        
        # Check if AI is satisfied with the essay
        if "SATISFIED" in response.content.upper():
            state["is_satisfied"] = True
            print("✅ AI is satisfied with the essay quality!")
        else:
            print("📋 Critique completed - improvements needed")
            
        return state
    except Exception as e:
        state["critique"] = f"❌ Error critiquing essay: {str(e)}"
        state["is_satisfied"] = True  # Stop on error
        return state

def improve_essay(state: EssayState) -> EssayState:
    """Improve the essay based on the critique"""
    print(f"🔧 Improving essay (Iteration {state['iteration_count']})")
    
    prompt = f"""
    Improve this essay based on the critique provided:
    
    CURRENT ESSAY:
    {state['current_essay']}
    
    CRITIQUE:
    {state['critique']}
    
    Please rewrite the essay addressing all the points mentioned in the critique.
    Make it better while maintaining the core message and topic.
    """
    
    try:
        response = llm.invoke(prompt)
        state["current_essay"] = response.content
        state["iteration_count"] += 1
        state["improvement_history"].append(f"Iteration {state['iteration_count'] - 1}: Improved based on critique")
        print("✅ Essay improved")
        return state
    except Exception as e:
        state["current_essay"] = f"❌ Error improving essay: {str(e)}"
        state["is_satisfied"] = True  # Stop on error
        return state

def should_continue(state: EssayState) -> str:
    """Decide whether to continue iterating or finish"""
    if state["is_satisfied"]:
        print("🎉 Essay refinement complete - AI is satisfied!")
        return "finish"
    elif state["iteration_count"] >= state["max_iterations"]:
        print(f"⏰ Max iterations ({state['max_iterations']}) reached")
        return "finish"
    else:
        print(f"🔄 Continuing to iteration {state['iteration_count'] + 1}")
        return "continue"

def finalize_essay(state: EssayState) -> EssayState:
    """Final processing and summary"""
    print("\n" + "="*50)
    print("📊 ESSAY IMPROVEMENT SUMMARY")
    print("="*50)
    print(f"Topic: {state['topic']}")
    print(f"Total iterations: {state['iteration_count']}")
    print(f"Satisfied: {'Yes' if state['is_satisfied'] else 'No (max iterations reached)'}")
    
    print("\n📈 Improvement History:")
    for i, improvement in enumerate(state["improvement_history"], 1):
        print(f"{i}. {improvement}")
    
    print("\n📝 FINAL ESSAY:")
    print("-" * 30)
    print(state["current_essay"])
    print("-" * 30)
    
    return state

# Build the recursive LangGraph
def build_recursive_essay_graph():
    builder = StateGraph(EssayState)
    
    # Add nodes
    builder.add_node("write_initial", write_initial_essay)
    builder.add_node("critique", critique_essay)
    builder.add_node("improve", improve_essay)
    builder.add_node("finalize", finalize_essay)
    
    # Define the flow
    builder.set_entry_point("write_initial")
    builder.add_edge("write_initial", "critique")
    
    # Conditional branching based on critique
    builder.add_conditional_edges(
        "critique",
        should_continue,
        {
            "continue": "improve",
            "finish": "finalize"
        }
    )
    
    # After improvement, go back to critique (creating the loop)
    builder.add_edge("improve", "critique")
    builder.add_edge("finalize", END)
    
    return builder.compile()

# Run the recursive essay writer
if __name__ == "__main__":
    graph = build_recursive_essay_graph()
    
    print("🚀 Recursive Self-Improving Essay Writer")
    print("="*40)
    
    while True:
        topic = input("\n📝 Enter essay topic (or 'exit' to quit): ").strip()
        if topic.lower() in {"exit", "quit"}:
            break
            
        max_iterations = input("🔢 Max iterations (default: 3): ").strip()
        try:
            max_iterations = int(max_iterations) if max_iterations else 3
        except ValueError:
            max_iterations = 3
        
        # Initialize state
        initial_state = {
            "topic": topic,
            "current_essay": "",
            "critique": "",
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "improvement_history": [],
            "is_satisfied": False
        }
        
        print(f"\n🎯 Starting recursive essay writing process...")
        print(f"Topic: {topic}")
        print(f"Max iterations: {max_iterations}")
        print("-" * 50)
        
        # Run the graph
        try:
            result = graph.invoke(initial_state)
            print(f"\n✅ Process completed successfully!")
        except Exception as e:
            print(f"\n❌ Error during execution: {str(e)}")
    
    print("\n👋 Thanks for using the Recursive Essay Writer!")
