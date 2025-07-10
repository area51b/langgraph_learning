# Phase 3.4: LangGraph Memory Stores - Reference Guide

## Overview
This phase covers implementing persistent memory stores for LangGraph agents, enabling conversation history and context retention across sessions.

## Learning Objectives
- Understand different memory store approaches
- Implement local file-based persistence
- Create in-memory stores for development
- Manage conversation sessions and context
- Handle memory limitations and cleanup

---

## 3.4.1 Local File-Based Memory Store

### Key Concepts
- **Persistent Storage**: Conversations survive application restarts
- **Session Management**: Each conversation has unique session ID
- **JSON Storage**: Human-readable conversation history
- **Context Awareness**: Agent considers previous messages

### Implementation Highlights

```python
class LocalMemoryStore:
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
```

### State Schema
```python
class MemoryState(TypedDict):
    input: str
    output: str
    conversation_history: List[Dict[str, Any]]
    session_id: str
    memory_summary: str
```

### Message Format
```python
{
    "role": "user" | "assistant",
    "content": "message content",
    "timestamp": "2024-01-01T12:00:00.000000"
}
```

### Usage Commands
- `new_session` - Start new conversation session
- `sessions` - List all available sessions
- `exit` - Quit the application

### Pros & Cons
**Pros:**
- Persistent across restarts
- Human-readable storage
- Simple implementation
- Good for small-scale applications

**Cons:**
- Slower I/O operations
- File system limitations
- No concurrent access handling
- Manual cleanup required

---

## 3.4.2 In-Memory Store for Development

### Key Concepts
- **Fast Operations**: No disk I/O, instant access
- **Memory Management**: Automatic cleanup with `deque(maxlen=...)`
- **Session Switching**: Switch between active sessions
- **Statistics Tracking**: Monitor memory usage

### Implementation Highlights

```python
class InMemoryStore:
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
```

### Enhanced Features
- **Session Metadata**: Track creation time, message count, last activity
- **Recent Messages**: Get last N messages for context
- **Memory Summary**: Generate conversation statistics
- **Session Management**: Clear, switch, list sessions

### Enhanced Commands
- `new_session <name>` - Start named session
- `switch <session>` - Switch to existing session
- `sessions` - List all sessions with summaries
- `clear` - Clear current session
- `stats` - Show memory statistics
- `help` - Show available commands

### Memory Management
```python
# Automatic cleanup with deque
conversations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_per_session))

# Get recent context only
recent_messages = memory_store.get_recent_messages(session_id, 5)
```

### Pros & Cons
**Pros:**
- Very fast operations
- Automatic memory management
- Great for development
- Rich session management

**Cons:**
- No persistence across restarts
- Memory limited by RAM
- Lost on application crash
- Not suitable for production

---

## Context Management Patterns

### Memory Summary Generation
```python
def get_memory_summary(self, session_id: str) -> str:
    """Generate memory summary"""
    conversation = self.get_conversation(session_id)
    if not conversation:
        return "New conversation started."
    
    user_messages = [msg for msg in conversation if msg['role'] == 'user']
    ai_messages = [msg for msg in conversation if msg['role'] == 'assistant']
    
    return f"Session: {len(user_messages)} user messages, {len(ai_messages)} AI responses."
```

### Context-Aware Prompting
```python
# Create context-aware prompt
context_prompt = f"""
Memory Summary: {memory_summary}

Recent conversation history:
{chr(10).join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:]])}

Current user input: {user_input}

Please respond considering the conversation history and context.
"""
```

### Error Handling with Memory
```python
try:
    response = llm.invoke(context_prompt)
    ai_response = response.content
    
    # Add AI response to memory
    memory_store.add_message(session_id, "assistant", ai_response)
    
except Exception as e:
    error_msg = f"❌ Error: {str(e)}"
    memory_store.add_message(session_id, "assistant", error_msg)
```

---

## Best Practices

### 1. Memory Limits
- Set reasonable history limits to prevent memory overflow
- Use `deque(maxlen=N)` for automatic cleanup
- Implement conversation summarization for long histories

### 2. Session Management
- Use meaningful session IDs
- Implement session cleanup strategies
- Track session metadata for monitoring

### 3. Context Window Management
- Limit context to recent messages (5-10 messages)
- Summarize older conversations
- Balance context richness vs. token limits

### 4. Error Handling
- Always handle file I/O errors
- Gracefully handle missing sessions
- Log errors to memory for debugging

### 5. Performance Considerations
- Use in-memory stores for development
- Consider async I/O for file operations
- Batch operations when possible

---

## Choosing the Right Store

### Local File Store - Use When:
- Small to medium scale applications
- Need conversation persistence
- Human-readable storage required
- Single-user applications

### In-Memory Store - Use When:
- Development and testing
- High-performance requirements
- Temporary conversations
- Prototype development

### Redis Store - Use When: (Coming Next)
- Production applications
- Multiple users/sessions
- Distributed systems
- High availability requirements

---

## Common Patterns

### Session ID Generation
```python
# Timestamp-based
session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# UUID-based
import uuid
session_id = str(uuid.uuid4())

# User-based
session_id = f"user_{user_id}_{timestamp}"
```

### Conversation History Structure
```python
conversation_history = [
    {
        "role": "user",
        "content": "Hello!",
        "timestamp": "2024-01-01T12:00:00.000000"
    },
    {
        "role": "assistant", 
        "content": "Hi there! How can I help you?",
        "timestamp": "2024-01-01T12:00:01.000000"
    }
]
```

### Memory Statistics
```python
def get_session_stats(self) -> Dict[str, Any]:
    return {
        "total_sessions": len(self.conversations),
        "total_messages": sum(len(conv) for conv in self.conversations.values()),
        "active_sessions": [sid for sid in self.conversations.keys() if len(self.conversations[sid]) > 0]
    }
```

---

## Next Steps

### Coming in Phase 3.4.3: Redis Store
- Production-ready persistence
- Distributed memory management
- Advanced caching strategies
- Multi-user session handling

### Integration with LangGraph
- Custom memory nodes
- State persistence patterns
- Memory-aware routing
- Conversation summarization

---

## Quick Reference Commands

### File-Based Store
```bash
🧑 You: new_session        # Start new session
🧑 You: sessions           # List all sessions  
🧑 You: exit              # Quit application
```

### In-Memory Store
```bash
🧑 You: new_session <name>  # Start named session
🧑 You: switch <session>    # Switch to session
🧑 You: sessions           # List all sessions
🧑 You: clear              # Clear current session
🧑 You: stats              # Show statistics
🧑 You: help               # Show help
🧑 You: exit               # Quit application
```

---

## Libraries Used
- `langgraph` 0.5.0 - Graph-based agent framework
- `langchain-core` 0.3.67 - Core LangChain components
- `langchain-google-genai` - Gemini integration
- `pathlib` - File system operations
- `collections.deque` - Memory-efficient queues
- `datetime` - Timestamp management
- `json` - Data serialization