from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any
import os
import json
import datetime
from pathlib import Path

# Load Gemini API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define state schema with memory
class MemoryState(TypedDict):
    input: str
    output: str
    conversation_history: List[Dict[str, Any]]
    session_id: str
    memory_summary: str

class LocalMemoryStore:
    """Simple file-based memory store for persistence"""
    
    def __init__(self, storage_path: str = "memory_store"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def save_conversation(self, session_id: str, conversation: List[Dict[str, Any]]):
        """Save conversation to local file"""
        file_path = self.storage_path / f"{session_id}.json"
        with open(file_path, 'w') as f:
            json.dump(conversation, f, indent=2)
    
    def load_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """Load conversation from local file"""
        file_path = self.storage_path / f"{session_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return []
    
    def get_memory_summary(self, session_id: str) -> str:
        """Generate a summary of conversation history"""
        conversation = self.load_conversation(session_id)
        if not conversation:
            return "No previous conversation history."
        
        # Simple summary - in practice, you'd use an LLM to summarize
        total_exchanges = len([msg for msg in conversation if msg['role'] == 'user'])
        recent_topics = [msg['content'][:50] + "..." for msg in conversation[-3:] if msg['role'] == 'user']
        
        return f"Session has {total_exchanges} exchanges. Recent topics: {', '.join(recent_topics)}"

# Initialize memory store
memory_store = LocalMemoryStore()

def memory_node(state: MemoryState) -> MemoryState:
    """Node that processes input with memory context"""
    user_input = state["input"]
    session_id = state["session_id"]
    
    # Load existing conversation
    conversation_history = memory_store.load_conversation(session_id)
    
    # Add current input to history
    conversation_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    # Get memory summary for context
    memory_summary = memory_store.get_memory_summary(session_id)
    
    # Create context-aware prompt
    context_prompt = f"""
    Memory Summary: {memory_summary}
    
    Recent conversation history:
    {chr(10).join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:]])}
    
    Current user input: {user_input}
    
    Please respond considering the conversation history and context.
    """
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    
    try:
        response = llm.invoke(context_prompt)
        ai_response = response.content
        
        # Add AI response to history
        conversation_history.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Save updated conversation
        memory_store.save_conversation(session_id, conversation_history)
        
        return {
            "input": user_input,
            "output": ai_response,
            "conversation_history": conversation_history,
            "session_id": session_id,
            "memory_summary": memory_summary
        }
        
    except Exception as e:
        return {
            "input": user_input,
            "output": f"❌ Error: {str(e)}",
            "conversation_history": conversation_history,
            "session_id": session_id,
            "memory_summary": memory_summary
        }

# Build the LangGraph
builder = StateGraph(MemoryState)
builder.add_node("memory_agent", memory_node)
builder.set_entry_point("memory_agent")
builder.set_finish_point("memory_agent")
graph = builder.compile()

# Run with session management
if __name__ == "__main__":
    print("🧠 Memory-Enabled Agent Started!")
    print("Type 'new_session' to start a new session")
    print("Type 'sessions' to list all sessions")
    print("Type 'exit' to quit")
    
    current_session = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"📝 Current session: {current_session}")
    
    while True:
        text = input("🧑 You: ")
        
        if text.lower() in {"exit", "quit"}:
            break
        elif text.lower() == "new_session":
            current_session = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"📝 New session started: {current_session}")
            continue
        elif text.lower() == "sessions":
            sessions = list(memory_store.storage_path.glob("*.json"))
            print(f"📁 Available sessions: {[s.stem for s in sessions]}")
            continue
        
        state = {
            "input": text,
            "output": "",
            "conversation_history": [],
            "session_id": current_session,
            "memory_summary": ""
        }
        
        result = graph.invoke(state)
        
        # Print the output with memory context
        print("🤖 Gemini:", result["output"])
        print(f"💭 Memory: {result['memory_summary']}")
        print(f"📊 History length: {len(result['conversation_history'])}")
        print("-" * 50)
