from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any
import os
import datetime
from collections import defaultdict, deque

# Load Gemini API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define state schema
class InMemoryState(TypedDict):
    input: str
    output: str
    conversation_history: List[Dict[str, Any]]
    session_id: str
    memory_summary: str

class InMemoryStore:
    """Fast in-memory store for development"""
    
    def __init__(self, max_history_per_session: int = 100):
        self.conversations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_per_session))
        self.session_metadata: Dict[str, Dict] = {}
        self.max_history = max_history_per_session
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.conversations[session_id].append(message)
        
        # Update session metadata
        if session_id not in self.session_metadata:
            self.session_metadata[session_id] = {
                "created_at": datetime.datetime.now().isoformat(),
                "message_count": 0
            }
        self.session_metadata[session_id]["message_count"] += 1
        self.session_metadata[session_id]["last_activity"] = datetime.datetime.now().isoformat()
    
    def get_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        return list(self.conversations[session_id])
    
    def get_recent_messages(self, session_id: str, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent messages from a session"""
        conversation = self.get_conversation(session_id)
        return conversation[-count:] if conversation else []
    
    def get_memory_summary(self, session_id: str) -> str:
        """Generate memory summary"""
        conversation = self.get_conversation(session_id)
        if not conversation:
            return "New conversation started."
        
        user_messages = [msg for msg in conversation if msg['role'] == 'user']
        ai_messages = [msg for msg in conversation if msg['role'] == 'assistant']
        
        metadata = self.session_metadata.get(session_id, {})
        
        return f"Session: {len(user_messages)} user messages, {len(ai_messages)} AI responses. Started: {metadata.get('created_at', 'Unknown')}"
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.conversations:
            self.conversations[session_id].clear()
            if session_id in self.session_metadata:
                del self.session_metadata[session_id]
    
    def list_sessions(self) -> List[str]:
        """List all active sessions"""
        return list(self.conversations.keys())
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about all sessions"""
        total_sessions = len(self.conversations)
        total_messages = sum(len(conv) for conv in self.conversations.values())
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "active_sessions": [sid for sid in self.conversations.keys() if len(self.conversations[sid]) > 0]
        }

# Initialize in-memory store
memory_store = InMemoryStore(max_history_per_session=50)

def smart_memory_node(state: InMemoryState) -> InMemoryState:
    """Node with intelligent memory management"""
    user_input = state["input"]
    session_id = state["session_id"]
    
    # Add user input to memory
    memory_store.add_message(session_id, "user", user_input)
    
    # Get recent context (last 5 messages)
    recent_messages = memory_store.get_recent_messages(session_id, 5)
    memory_summary = memory_store.get_memory_summary(session_id)
    
    # Create intelligent context prompt
    context_lines = []
    if recent_messages:
        context_lines.append("Recent conversation context:")
        for msg in recent_messages[:-1]:  # Exclude current message
            context_lines.append(f"{msg['role']}: {msg['content']}")
        context_lines.append("")
    
    context_lines.append(f"Current user input: {user_input}")
    context_lines.append(f"Memory summary: {memory_summary}")
    context_lines.append("")
    context_lines.append("Please respond naturally, considering the conversation context.")
    
    context_prompt = "\n".join(context_lines)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    
    try:
        response = llm.invoke(context_prompt)
        ai_response = response.content
        
        # Add AI response to memory
        memory_store.add_message(session_id, "assistant", ai_response)
        
        # Get updated conversation history
        conversation_history = memory_store.get_conversation(session_id)
        
        return {
            "input": user_input,
            "output": ai_response,
            "conversation_history": conversation_history,
            "session_id": session_id,
            "memory_summary": memory_summary
        }
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        memory_store.add_message(session_id, "assistant", error_msg)
        
        return {
            "input": user_input,
            "output": error_msg,
            "conversation_history": memory_store.get_conversation(session_id),
            "session_id": session_id,
            "memory_summary": memory_summary
        }

# Build the LangGraph
builder = StateGraph(InMemoryState)
builder.add_node("smart_memory", smart_memory_node)
builder.set_entry_point("smart_memory")
builder.set_finish_point("smart_memory")
graph = builder.compile()

# Enhanced CLI interface
def print_commands():
    print("\n🔧 Available Commands:")
    print("  new_session <name>  - Start a new session")
    print("  switch <session>    - Switch to existing session")
    print("  sessions           - List all sessions")
    print("  clear              - Clear current session")
    print("  stats              - Show memory statistics")
    print("  help               - Show this help")
    print("  exit               - Quit the program")
    print("-" * 50)

# Run with enhanced session management
if __name__ == "__main__":
    print("🧠 Smart In-Memory Agent Started!")
    print_commands()
    
    current_session = f"session_{datetime.datetime.now().strftime('%H%M%S')}"
    print(f"📝 Current session: {current_session}")
    
    while True:
        text = input(f"🧑 [{current_session}] You: ")
        
        if text.lower() in {"exit", "quit"}:
            break
        elif text.lower() == "help":
            print_commands()
            continue
        elif text.lower().startswith("new_session"):
            parts = text.split()
            if len(parts) > 1:
                current_session = parts[1]
            else:
                current_session = f"session_{datetime.datetime.now().strftime('%H%M%S')}"
            print(f"📝 New session started: {current_session}")
            continue
        elif text.lower().startswith("switch"):
            parts = text.split()
            if len(parts) > 1 and parts[1] in memory_store.list_sessions():
                current_session = parts[1]
                print(f"📝 Switched to session: {current_session}")
            else:
                print("❌ Session not found. Available sessions:", memory_store.list_sessions())
            continue
        elif text.lower() == "sessions":
            sessions = memory_store.list_sessions()
            print(f"📁 Available sessions: {sessions}")
            for session in sessions:
                summary = memory_store.get_memory_summary(session)
                print(f"  {session}: {summary}")
            continue
        elif text.lower() == "clear":
            memory_store.clear_session(current_session)
            print(f"🗑️ Session {current_session} cleared")
            continue
        elif text.lower() == "stats":
            stats = memory_store.get_session_stats()
            print(f"📊 Memory Statistics:")
            print(f"  Total sessions: {stats['total_sessions']}")
            print(f"  Total messages: {stats['total_messages']}")
            print(f"  Active sessions: {stats['active_sessions']}")
            continue
        
        state = {
            "input": text,
            "output": "",
            "conversation_history": [],
            "session_id": current_session,
            "memory_summary": ""
        }
        
        result = graph.invoke(state)
        
        # Print the output with enhanced info
        print("🤖 Gemini:", result["output"])
        print(f"💭 {result['memory_summary']}")
        print(f"📊 Messages in session: {len(result['conversation_history'])}")
        print("-" * 50)
