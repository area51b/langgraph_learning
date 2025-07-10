from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Optional
import os
import json
import datetime
import redis
from redis.connection import ConnectionPool
import time
import hashlib

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define state schema
class RedisMemoryState(TypedDict):
    input: str
    output: str
    conversation_history: List[Dict[str, Any]]
    session_id: str
    memory_summary: str
    user_id: Optional[str]

class RedisMemoryStore:
    """Production-ready Redis memory store"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 max_connections: int = 10,
                 session_ttl: int = 86400,  # 24 hours
                 max_messages_per_session: int = 1000):
        
        self.session_ttl = session_ttl
        self.max_messages = max_messages_per_session
        
        # Create connection pool for better performance
        self.pool = ConnectionPool.from_url(
            redis_url,
            max_connections=max_connections,
            retry_on_timeout=True
        )
        self.redis_client = redis.Redis(connection_pool=self.pool)
        
        # Test connection
        try:
            self.redis_client.ping()
            print("✅ Redis connection established")
        except redis.ConnectionError:
            print("❌ Redis connection failed - make sure Redis is running")
            raise
    
    def _get_session_key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"session:{session_id}"
    
    def _get_metadata_key(self, session_id: str) -> str:
        """Generate Redis key for session metadata"""
        return f"metadata:{session_id}"
    
    def _get_user_sessions_key(self, user_id: str) -> str:
        """Generate Redis key for user's sessions"""
        return f"user_sessions:{user_id}"
    
    def add_message(self, session_id: str, role: str, content: str, user_id: Optional[str] = None):
        """Add a message to session with atomic operations"""
        pipe = self.redis_client.pipeline()
        
        try:
            # Create message object
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.datetime.now().isoformat(),
                "id": hashlib.md5(f"{session_id}:{role}:{content}:{time.time()}".encode()).hexdigest()[:8]
            }
            
            session_key = self._get_session_key(session_id)
            metadata_key = self._get_metadata_key(session_id)
            
            # Add message to session list
            pipe.lpush(session_key, json.dumps(message))
            
            # Trim list to max messages (keep most recent)
            pipe.ltrim(session_key, 0, self.max_messages - 1)
            
            # Update metadata
            metadata = {
                "last_activity": datetime.datetime.now().isoformat(),
                "message_count": self.redis_client.llen(session_key) + 1,
                "user_id": user_id
            }
            pipe.hset(metadata_key, mapping=metadata)
            
            # Set TTL for session and metadata
            pipe.expire(session_key, self.session_ttl)
            pipe.expire(metadata_key, self.session_ttl)
            
            # Add session to user's session list if user_id provided
            if user_id:
                user_sessions_key = self._get_user_sessions_key(user_id)
                pipe.sadd(user_sessions_key, session_id)
                pipe.expire(user_sessions_key, self.session_ttl)
            
            # Execute all operations atomically
            pipe.execute()
            
        except Exception as e:
            print(f"❌ Error adding message: {e}")
            raise
    
    def get_conversation(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get conversation history (most recent first)"""
        try:
            session_key = self._get_session_key(session_id)
            messages = self.redis_client.lrange(session_key, 0, limit - 1)
            
            # Parse and reverse to get chronological order
            conversation = []
            for msg in reversed(messages):
                try:
                    conversation.append(json.loads(msg.decode('utf-8')))
                except json.JSONDecodeError:
                    continue
            
            return conversation
            
        except Exception as e:
            print(f"❌ Error getting conversation: {e}")
            return []
    
    def get_recent_messages(self, session_id: str, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent messages for context"""
        conversation = self.get_conversation(session_id, count)
        return conversation[-count:] if conversation else []
    
    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """Get session metadata"""
        try:
            metadata_key = self._get_metadata_key(session_id)
            metadata = self.redis_client.hgetall(metadata_key)
            
            # Decode bytes to strings
            return {k.decode('utf-8'): v.decode('utf-8') for k, v in metadata.items()}
            
        except Exception as e:
            print(f"❌ Error getting metadata: {e}")
            return {}
    
    def get_memory_summary(self, session_id: str) -> str:
        """Generate intelligent memory summary"""
        try:
            conversation = self.get_conversation(session_id, 50)  # Last 50 messages
            metadata = self.get_session_metadata(session_id)
            
            if not conversation:
                return "New conversation session."
            
            user_messages = [msg for msg in conversation if msg['role'] == 'user']
            ai_messages = [msg for msg in conversation if msg['role'] == 'assistant']
            
            # Calculate session age
            if metadata.get('last_activity'):
                try:
                    last_activity = datetime.datetime.fromisoformat(metadata['last_activity'])
                    age = datetime.datetime.now() - last_activity
                    age_str = f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m ago"
                except:
                    age_str = "recently"
            else:
                age_str = "recently"
            
            # Recent topics (last 3 user messages)
            recent_topics = [msg['content'][:30] + "..." for msg in user_messages[-3:]]
            
            return f"Session: {len(user_messages)} exchanges, last active {age_str}. Recent: {', '.join(recent_topics)}"
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return "Error generating memory summary."
    
    def clear_session(self, session_id: str):
        """Clear a session and its metadata"""
        try:
            session_key = self._get_session_key(session_id)
            metadata_key = self._get_metadata_key(session_id)
            
            pipe = self.redis_client.pipeline()
            pipe.delete(session_key)
            pipe.delete(metadata_key)
            pipe.execute()
            
        except Exception as e:
            print(f"❌ Error clearing session: {e}")
    
    def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """List all sessions or user's sessions"""
        try:
            if user_id:
                user_sessions_key = self._get_user_sessions_key(user_id)
                sessions = self.redis_client.smembers(user_sessions_key)
                return [s.decode('utf-8') for s in sessions]
            else:
                # Get all session keys
                session_keys = self.redis_client.keys("session:*")
                return [key.decode('utf-8').replace('session:', '') for key in session_keys]
                
        except Exception as e:
            print(f"❌ Error listing sessions: {e}")
            return []
    
    def get_redis_stats(self) -> Dict[str, Any]:
        """Get Redis server statistics"""
        try:
            info = self.redis_client.info()
            return {
                "connected_clients": info.get('connected_clients', 0),
                "used_memory": info.get('used_memory_human', 'Unknown'),
                "total_commands_processed": info.get('total_commands_processed', 0),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0)
            }
        except Exception as e:
            print(f"❌ Error getting Redis stats: {e}")
            return {}
    
    def cleanup_expired_sessions(self):
        """Manual cleanup of expired sessions (Redis handles this automatically)"""
        try:
            # Get all session keys
            session_keys = self.redis_client.keys("session:*")
            metadata_keys = self.redis_client.keys("metadata:*")
            
            expired_count = 0
            for key in session_keys:
                ttl = self.redis_client.ttl(key)
                if ttl == -1:  # No TTL set
                    self.redis_client.expire(key, self.session_ttl)
                elif ttl == -2:  # Key doesn't exist
                    expired_count += 1
            
            return {
                "total_sessions": len(session_keys),
                "expired_sessions": expired_count
            }
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            return {}

# Initialize Redis memory store
try:
    memory_store = RedisMemoryStore(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        session_ttl=86400,  # 24 hours
        max_messages_per_session=500
    )
except Exception as e:
    print(f"❌ Failed to initialize Redis store: {e}")
    print("Please ensure Redis is running and accessible")
    exit(1)

def redis_memory_node(state: RedisMemoryState) -> RedisMemoryState:
    """Node with Redis-backed memory"""
    user_input = state["input"]
    session_id = state["session_id"]
    user_id = state.get("user_id")
    
    # Add user input to Redis
    memory_store.add_message(session_id, "user", user_input, user_id)
    
    # Get recent context
    recent_messages = memory_store.get_recent_messages(session_id, 6)
    memory_summary = memory_store.get_memory_summary(session_id)
    
    # Create context-aware prompt
    context_lines = []
    if recent_messages:
        context_lines.append("Recent conversation context:")
        for msg in recent_messages[:-1]:  # Exclude current message
            context_lines.append(f"{msg['role']}: {msg['content']}")
        context_lines.append("")
    
    context_lines.extend([
        f"Current user input: {user_input}",
        f"Session context: {memory_summary}",
        "",
        "Please respond naturally, considering the conversation history and context."
    ])
    
    context_prompt = "\n".join(context_lines)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    
    try:
        response = llm.invoke(context_prompt)
        ai_response = response.content
        
        # Add AI response to Redis
        memory_store.add_message(session_id, "assistant", ai_response, user_id)
        
        # Get updated conversation
        conversation_history = memory_store.get_conversation(session_id)
        
        return {
            "input": user_input,
            "output": ai_response,
            "conversation_history": conversation_history,
            "session_id": session_id,
            "memory_summary": memory_summary,
            "user_id": user_id
        }
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        memory_store.add_message(session_id, "assistant", error_msg, user_id)
        
        return {
            "input": user_input,
            "output": error_msg,
            "conversation_history": memory_store.get_conversation(session_id),
            "session_id": session_id,
            "memory_summary": memory_summary,
            "user_id": user_id
        }

# Build the LangGraph
builder = StateGraph(RedisMemoryState)
builder.add_node("redis_memory", redis_memory_node)
builder.set_entry_point("redis_memory")
builder.set_finish_point("redis_memory")
graph = builder.compile()

# Enhanced CLI with Redis features
def print_redis_commands():
    print("\n🚀 Redis Memory Agent Commands:")
    print("  user <user_id>       - Set user ID for session tracking")
    print("  new_session <name>   - Start a new session")
    print("  switch <session>     - Switch to existing session")
    print("  sessions             - List all sessions")
    print("  my_sessions          - List current user's sessions")
    print("  clear                - Clear current session")
    print("  metadata             - Show session metadata")
    print("  redis_stats          - Show Redis server statistics")
    print("  cleanup              - Run session cleanup")
    print("  help                 - Show this help")
    print("  exit                 - Quit the program")
    print("-" * 60)

# Main application loop
if __name__ == "__main__":
    print("🚀 Redis Memory Agent Started!")
    print_redis_commands()
    
    current_session = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    current_user = None
    
    print(f"📝 Current session: {current_session}")
    
    while True:
        user_display = f"[{current_user}]" if current_user else ""
        session_display = current_session.replace("session_", "")
        
        text = input(f"🧑 {user_display}[{session_display}] You: ")
        
        if text.lower() in {"exit", "quit"}:
            break
        elif text.lower() == "help":
            print_redis_commands()
            continue
        elif text.lower().startswith("user "):
            parts = text.split()
            if len(parts) > 1:
                current_user = parts[1]
                print(f"👤 User set to: {current_user}")
            else:
                print("❌ Usage: user <user_id>")
            continue
        elif text.lower().startswith("new_session"):
            parts = text.split()
            if len(parts) > 1:
                current_session = f"session_{parts[1]}"
            else:
                current_session = f"session_{datetime.datetime.now().strftime('%H%M%S')}"
            print(f"📝 New session started: {current_session}")
            continue
        elif text.lower().startswith("switch "):
            parts = text.split()
            if len(parts) > 1:
                session_name = parts[1]
                if not session_name.startswith("session_"):
                    session_name = f"session_{session_name}"
                current_session = session_name
                print(f"📝 Switched to session: {current_session}")
            else:
                print("❌ Usage: switch <session>")
            continue
        elif text.lower() == "sessions":
            sessions = memory_store.list_sessions()
            if sessions:
                print(f"📁 All sessions ({len(sessions)}):")
                for session in sessions[:10]:  # Show first 10
                    metadata = memory_store.get_session_metadata(session)
                    user_info = f"[{metadata.get('user_id', 'unknown')}]" if metadata.get('user_id') else ""
                    print(f"  {session} {user_info} - {metadata.get('message_count', 0)} messages")
                if len(sessions) > 10:
                    print(f"  ... and {len(sessions) - 10} more")
            else:
                print("📁 No sessions found")
            continue
        elif text.lower() == "my_sessions":
            if current_user:
                sessions = memory_store.list_sessions(current_user)
                if sessions:
                    print(f"📁 {current_user}'s sessions ({len(sessions)}):")
                    for session in sessions:
                        metadata = memory_store.get_session_metadata(session)
                        print(f"  {session} - {metadata.get('message_count', 0)} messages")
                else:
                    print(f"📁 No sessions found for {current_user}")
            else:
                print("❌ Set user ID first with 'user <user_id>'")
            continue
        elif text.lower() == "clear":
            memory_store.clear_session(current_session)
            print(f"🗑️ Session {current_session} cleared")
            continue
        elif text.lower() == "metadata":
            metadata = memory_store.get_session_metadata(current_session)
            if metadata:
                print(f"📊 Session Metadata for {current_session}:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
            else:
                print(f"📊 No metadata found for {current_session}")
            continue
        elif text.lower() == "redis_stats":
            stats = memory_store.get_redis_stats()
            print(f"📊 Redis Server Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            continue
        elif text.lower() == "cleanup":
            result = memory_store.cleanup_expired_sessions()
            print(f"🧹 Cleanup completed: {result}")
            continue
        
        # Process regular message
        state = {
            "input": text,
            "output": "",
            "conversation_history": [],
            "session_id": current_session,
            "memory_summary": "",
            "user_id": current_user
        }
        
        result = graph.invoke(state)
        
        # Display response with Redis info
        print("🤖 Gemini:", result["output"])
        print(f"💭 {result['memory_summary']}")
        print(f"📊 Messages: {len(result['conversation_history'])}")
        
        # Show metadata occasionally
        if len(result['conversation_history']) % 5 == 0:
            metadata = memory_store.get_session_metadata(current_session)
            if metadata:
                print(f"🔍 Session info: {metadata.get('message_count', 0)} total messages")
        
        print("-" * 60)

# docker run -d -p 6379:6379 redis:latest
# REDIS_URL=redis://localhost:6379
