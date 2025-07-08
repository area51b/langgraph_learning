from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import TypedDict, List, Literal
import os
import re
import json

# Load Gemini API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define state schema
class SmartAgentState(TypedDict):
    current_input: str
    response: str
    user_id: str
    intent: str
    confidence: float
    context: dict
    conversation_history: List[dict]

# Initialize store and LLM
store = InMemoryStore()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.1)

# Intent Classification Tools
@tool
def math_calculator(expression: str) -> str:
    """Perform mathematical calculations"""
    try:
        import math
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Math error: {str(e)}"

@tool
def get_weather_info(location: str) -> str:
    """Get weather information (simulated)"""
    weather_data = {
        "new york": "New York: 22°C, Partly cloudy with light rain expected",
        "london": "London: 15°C, Overcast with occasional drizzle",
        "tokyo": "Tokyo: 28°C, Sunny with high humidity",
        "sydney": "Sydney: 25°C, Clear skies with gentle breeze"
    }
    location_lower = location.lower()
    for city, weather in weather_data.items():
        if city in location_lower:
            return weather
    return f"Weather info for {location}: Not available in simulation"

# Intent Classification Node
def intent_classifier_node(state: SmartAgentState) -> SmartAgentState:
    """Classify user intent using LLM"""
    current_input = state["current_input"]
    
    # Intent classification prompt
    classification_prompt = f"""
    Analyze the user's input and classify it into one of these categories:
    
    1. MATH - Mathematical calculations, arithmetic, algebra
    2. WEATHER - Weather queries, climate information
    3. PERSONAL - Personal information, preferences, memory-related
    4. GENERAL - General conversation, questions, chitchat
    5. CREATIVE - Creative writing, stories, poems
    6. HELP - Help requests, instructions, how-to questions
    
    User input: "{current_input}"
    
    Respond with ONLY a JSON object in this format:
    {{"intent": "CATEGORY", "confidence": 0.85, "reasoning": "brief explanation"}}
    """
    
    try:
        response = llm.invoke(classification_prompt)
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            classification = json.loads(json_match.group())
            intent = classification.get("intent", "GENERAL")
            confidence = classification.get("confidence", 0.5)
            reasoning = classification.get("reasoning", "Default classification")
        else:
            intent = "GENERAL"
            confidence = 0.5
            reasoning = "Failed to parse classification"
        
        return {
            "current_input": current_input,
            "response": "",
            "user_id": state["user_id"],
            "intent": intent,
            "confidence": confidence,
            "context": {"reasoning": reasoning},
            "conversation_history": state.get("conversation_history", [])
        }
    except Exception as e:
        return {
            "current_input": current_input,
            "response": f"Error in intent classification: {str(e)}",
            "user_id": state["user_id"],
            "intent": "GENERAL",
            "confidence": 0.1,
            "context": {"error": str(e)},
            "conversation_history": state.get("conversation_history", [])
        }

# Specialized Handler Nodes
def math_handler_node(state: SmartAgentState) -> SmartAgentState:
    """Handle mathematical queries"""
    current_input = state["current_input"]
    user_id = state["user_id"]
    
    try:
        # Extract mathematical expression
        math_prompt = f"""
        Extract and solve the mathematical expression from: "{current_input}"
        If it's a word problem, convert it to a mathematical expression first.
        Provide the calculation and result.
        """
        
        response = llm.invoke(math_prompt)
        
        # Try to find and execute any mathematical expressions
        expressions = re.findall(r'[\d+\-*/().\s]+', current_input)
        if expressions:
            calc_result = math_calculator.invoke({"expression": expressions[0].strip()})
            final_response = f"🧮 {response.content}\n\n{calc_result}"
        else:
            final_response = f"🧮 {response.content}"
        
        return {
            "current_input": current_input,
            "response": final_response,
            "user_id": user_id,
            "intent": state["intent"],
            "confidence": state["confidence"],
            "context": state["context"],
            "conversation_history": state["conversation_history"]
        }
    except Exception as e:
        return {
            "current_input": current_input,
            "response": f"🧮 Math handler error: {str(e)}",
            "user_id": user_id,
            "intent": state["intent"],
            "confidence": state["confidence"],
            "context": state["context"],
            "conversation_history": state["conversation_history"]
        }

def weather_handler_node(state: SmartAgentState) -> SmartAgentState:
    """Handle weather queries"""
    current_input = state["current_input"]
    user_id = state["user_id"]
    
    try:
        # Extract location from input
        location_prompt = f"""
        Extract the location from this weather query: "{current_input}"
        If no specific location is mentioned, assume "current location".
        Respond with just the location name.
        """
        
        location_response = llm.invoke(location_prompt)
        location = location_response.content.strip()
        
        # Get weather info
        weather_info = get_weather_info.invoke({"location": location})
        
        response = f"🌤️ {weather_info}"
        
        return {
            "current_input": current_input,
            "response": response,
            "user_id": user_id,
            "intent": state["intent"],
            "confidence": state["confidence"],
            "context": {**state["context"], "location": location},
            "conversation_history": state["conversation_history"]
        }
    except Exception as e:
        return {
            "current_input": current_input,
            "response": f"🌤️ Weather handler error: {str(e)}",
            "user_id": user_id,
            "intent": state["intent"],
            "confidence": state["confidence"],
            "context": state["context"],
            "conversation_history": state["conversation_history"]
        }

def personal_handler_node(state: SmartAgentState) -> SmartAgentState:
    """Handle personal information and memory queries"""
    current_input = state["current_input"]
    user_id = state["user_id"]
    
    try:
        # Get conversation history from store
        namespace = "personal_info"
        memory_key = f"user_{user_id}"
        
        stored_item = store.get(namespace, memory_key)
        personal_info = stored_item.value if stored_item else {}
        
        # Analyze if this is storing or retrieving personal info
        analysis_prompt = f"""
        Analyze if the user is:
        1. STORING personal information (telling me about themselves)
        2. RETRIEVING personal information (asking about what I know about them)
        
        User input: "{current_input}"
        Current stored info: {personal_info}
        
        Respond with JSON: {{"action": "STORE" or "RETRIEVE", "key": "name/preference/etc", "value": "if storing"}}
        """
        
        response = llm.invoke(analysis_prompt)
        
        # Extract action
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            action_data = json.loads(json_match.group())
            action = action_data.get("action", "RETRIEVE")
            key = action_data.get("key", "general")
            value = action_data.get("value", "")
            
            if action == "STORE" and value:
                personal_info[key] = value
                store.put(namespace, memory_key, personal_info)
                response_text = f"👤 Got it! I'll remember that {key}: {value}"
            else:
                if personal_info:
                    info_list = [f"{k}: {v}" for k, v in personal_info.items()]
                    response_text = f"👤 Here's what I know about you: {', '.join(info_list)}"
                else:
                    response_text = "👤 I don't have any personal information stored about you yet."
        else:
            response_text = "👤 I understand you're sharing something personal, but I couldn't parse the details."
        
        return {
            "current_input": current_input,
            "response": response_text,
            "user_id": user_id,
            "intent": state["intent"],
            "confidence": state["confidence"],
            "context": {**state["context"], "personal_info": personal_info},
            "conversation_history": state["conversation_history"]
        }
    except Exception as e:
        return {
            "current_input": current_input,
            "response": f"👤 Personal handler error: {str(e)}",
            "user_id": user_id,
            "intent": state["intent"],
            "confidence": state["confidence"],
            "context": state["context"],
            "conversation_history": state["conversation_history"]
        }

def creative_handler_node(state: SmartAgentState) -> SmartAgentState:
    """Handle creative writing requests"""
    current_input = state["current_input"]
    user_id = state["user_id"]
    
    try:
        creative_prompt = f"""
        The user is asking for creative content: "{current_input}"
        
        Provide a creative response - this could be:
        - A short story or poem
        - Creative writing
        - Imaginative scenarios
        - Artistic descriptions
        
        Be creative and engaging!
        """
        
        response = llm.invoke(creative_prompt)
        
        return {
            "current_input": current_input,
            "response": f"✨ {response.content}",
            "user_id": user_id,
            "intent": state["intent"],
            "confidence": state["confidence"],
            "context": state["context"],
            "conversation_history": state["conversation_history"]
        }
    except Exception as e:
        return {
            "current_input": current_input,
            "response": f"✨ Creative handler error: {str(e)}",
            "user_id": user_id,
            "intent": state["intent"],
            "confidence": state["confidence"],
            "context": state["context"],
            "conversation_history": state["conversation_history"]
        }

def help_handler_node(state: SmartAgentState) -> SmartAgentState:
    """Handle help and instruction requests"""
    current_input = state["current_input"]
    user_id = state["user_id"]
    
    try:
        help_prompt = f"""
        The user is asking for help: "{current_input}"
        
        Provide helpful, step-by-step instructions or guidance.
        Be practical and actionable.
        """
        
        response = llm.invoke(help_prompt)
        
        return {
            "current_input": current_input,
            "response": f"🆘 {response.content}",
            "user_id": user_id,
            "intent": state["intent"],
            "confidence": state["confidence"],
            "context": state["context"],
            "conversation_history": state["conversation_history"]
        }
    except Exception as e:
        return {
            "current_input": current_input,
            "response": f"🆘 Help handler error: {str(e)}",
            "user_id": user_id,
            "intent": state["intent"],
            "confidence": state["confidence"],
            "context": state["context"],
            "conversation_history": state["conversation_history"]
        }

def general_handler_node(state: SmartAgentState) -> SmartAgentState:
    """Handle general conversation"""
    current_input = state["current_input"]
    user_id = state["user_id"]
    
    try:
        # Get conversation history for context
        namespace = "general_conversation"
        memory_key = f"user_{user_id}"
        
        stored_item = store.get(namespace, memory_key)
        history = stored_item.value if stored_item else []
        
        # Add current exchange to history
        history.append({"user": current_input, "timestamp": str(os.times())})
        
        # Keep only last 5 exchanges
        history = history[-5:]
        store.put(namespace, memory_key, history)
        
        response = llm.invoke(current_input)
        
        return {
            "current_input": current_input,
            "response": f"💬 {response.content}",
            "user_id": user_id,
            "intent": state["intent"],
            "confidence": state["confidence"],
            "context": state["context"],
            "conversation_history": history
        }
    except Exception as e:
        return {
            "current_input": current_input,
            "response": f"💬 General handler error: {str(e)}",
            "user_id": user_id,
            "intent": state["intent"],
            "confidence": state["confidence"],
            "context": state["context"],
            "conversation_history": state["conversation_history"]
        }

# Conditional routing function
def route_by_intent(state: SmartAgentState) -> Literal["math", "weather", "personal", "creative", "help", "general"]:
    """Route to appropriate handler based on intent"""
    intent = state["intent"]
    confidence = state["confidence"]
    
    # If confidence is too low, route to general
    if confidence < 0.6:
        return "general"
    
    # Route based on intent
    routing_map = {
        "MATH": "math",
        "WEATHER": "weather", 
        "PERSONAL": "personal",
        "CREATIVE": "creative",
        "HELP": "help",
        "GENERAL": "general"
    }
    
    return routing_map.get(intent, "general")

# Build the graph
builder = StateGraph(SmartAgentState)

# Add nodes
builder.add_node("intent_classifier", intent_classifier_node)
builder.add_node("math", math_handler_node)
builder.add_node("weather", weather_handler_node)
builder.add_node("personal", personal_handler_node)
builder.add_node("creative", creative_handler_node)
builder.add_node("help", help_handler_node)
builder.add_node("general", general_handler_node)

# Add edges
builder.set_entry_point("intent_classifier")
builder.add_conditional_edges(
    "intent_classifier",
    route_by_intent,
    {
        "math": "math",
        "weather": "weather",
        "personal": "personal", 
        "creative": "creative",
        "help": "help",
        "general": "general"
    }
)

# All handlers end the flow
builder.set_finish_point("math")
builder.set_finish_point("weather")
builder.set_finish_point("personal")
builder.set_finish_point("creative")
builder.set_finish_point("help")
builder.set_finish_point("general")

# Compile graph
graph = builder.compile(store=store)

# Utility functions
def show_personal_info(user_id: str):
    """Show stored personal information"""
    namespace = "personal_info"
    memory_key = f"user_{user_id}"
    
    try:
        stored_item = store.get(namespace, memory_key)
        personal_info = stored_item.value if stored_item else {}
        
        if personal_info:
            print("\n👤 Personal Information:")
            for key, value in personal_info.items():
                print(f"  {key}: {value}")
        else:
            print("\n👤 No personal information stored")
        print()
    except Exception:
        print("\n👤 No personal information stored")
        print()

def clear_personal_info(user_id: str):
    """Clear personal information"""
    namespace = "personal_info"
    memory_key = f"user_{user_id}"
    store.put(namespace, memory_key, {})
    print("🧹 Personal information cleared!")

# Main execution
if __name__ == "__main__":
    print("🧠 Smart Conditional Branching Agent")
    print("This agent can handle different types of queries:")
    print("  🧮 Math - Calculations and math problems")
    print("  🌤️ Weather - Weather information queries")
    print("  👤 Personal - Remember and recall personal info")
    print("  ✨ Creative - Creative writing and stories")
    print("  🆘 Help - Instructions and how-to questions")
    print("  💬 General - General conversation")
    print("\nCommands:")
    print("  'info' - Show stored personal information")
    print("  'clear' - Clear personal information")
    print("  'exit' or 'quit' - End conversation")
    print("\nExample queries:")
    print("  'Calculate 15 * 8 + 25'")
    print("  'What's the weather in Tokyo?'")
    print("  'My name is Alice and I love coffee'")
    print("  'What's my name?'")
    print("  'Write a short story about a robot'")
    print("  'How do I bake a cake?'")
    print()
    
    user_id = "user_001"
    
    while True:
        user_input = input("🧑 You: ")
        
        if user_input.lower() in {"exit", "quit"}:
            break
        elif user_input.lower() == "clear":
            clear_personal_info(user_id)
            continue
        elif user_input.lower() == "info":
            show_personal_info(user_id)
            continue
        
        # Create state for this interaction
        state = {
            "current_input": user_input,
            "response": "",
            "user_id": user_id,
            "intent": "",
            "confidence": 0.0,
            "context": {},
            "conversation_history": []
        }
        
        # Process through graph
        result = graph.invoke(state)
        
        # Print response with intent information
        print(f"🤖 Agent: {result['response']}")
        print(f"📊 Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        
        if result['context'].get('reasoning'):
            print(f"🔍 Reasoning: {result['context']['reasoning']}")
        
        print()
