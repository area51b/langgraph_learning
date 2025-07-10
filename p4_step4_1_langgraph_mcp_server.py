import asyncio
import json
from typing import Any, Dict, List, Optional, TypedDict
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult, Resource, GetResourceResult
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ========== LangGraph State & Nodes ==========
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    current_step: str
    result: Optional[str]
    error: Optional[str]

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

def reasoning_node(state: AgentState) -> AgentState:
    """Node that performs reasoning with Gemini"""
    messages = state["messages"]
    
    try:
        # Get the latest human message
        latest_message = messages[-1]["content"] if messages else ""
        
        # Create reasoning prompt
        reasoning_prompt = f"""
        You are a helpful AI assistant. Analyze the following request and provide a thoughtful response:
        
        Request: {latest_message}
        
        Please provide a clear, helpful response.
        """
        
        response = llm.invoke(reasoning_prompt)
        
        # Add AI response to messages
        messages.append({
            "role": "assistant",
            "content": response.content
        })
        
        return {
            "messages": messages,
            "current_step": "completed",
            "result": response.content,
            "error": None
        }
        
    except Exception as e:
        return {
            "messages": messages,
            "current_step": "error",
            "result": None,
            "error": str(e)
        }

def analysis_node(state: AgentState) -> AgentState:
    """Node that performs detailed analysis"""
    messages = state["messages"]
    
    try:
        latest_message = messages[-1]["content"] if messages else ""
        
        analysis_prompt = f"""
        Perform a detailed analysis of the following:
        
        Topic: {latest_message}
        
        Please provide:
        1. Key points
        2. Implications
        3. Recommendations
        4. Summary
        
        Format your response clearly with numbered sections.
        """
        
        response = llm.invoke(analysis_prompt)
        
        messages.append({
            "role": "assistant",
            "content": response.content
        })
        
        return {
            "messages": messages,
            "current_step": "completed",
            "result": response.content,
            "error": None
        }
        
    except Exception as e:
        return {
            "messages": messages,
            "current_step": "error",
            "result": None,
            "error": str(e)
        }

def creative_node(state: AgentState) -> AgentState:
    """Node that generates creative content"""
    messages = state["messages"]
    
    try:
        latest_message = messages[-1]["content"] if messages else ""
        
        creative_prompt = f"""
        Generate creative content based on this request:
        
        Request: {latest_message}
        
        Be creative, engaging, and original. Use vivid language and interesting perspectives.
        """
        
        response = llm.invoke(creative_prompt)
        
        messages.append({
            "role": "assistant",
            "content": response.content
        })
        
        return {
            "messages": messages,
            "current_step": "completed",
            "result": response.content,
            "error": None
        }
        
    except Exception as e:
        return {
            "messages": messages,
            "current_step": "error",
            "result": None,
            "error": str(e)
        }

# ========== Create LangGraph Workflows ==========
def create_reasoning_graph():
    """Create reasoning workflow"""
    builder = StateGraph(AgentState)
    builder.add_node("reasoning", reasoning_node)
    builder.set_entry_point("reasoning")
    builder.set_finish_point("reasoning")
    return builder.compile()

def create_analysis_graph():
    """Create analysis workflow"""
    builder = StateGraph(AgentState)
    builder.add_node("analysis", analysis_node)
    builder.set_entry_point("analysis")
    builder.set_finish_point("analysis")
    return builder.compile()

def create_creative_graph():
    """Create creative workflow"""
    builder = StateGraph(AgentState)
    builder.add_node("creative", creative_node)
    builder.set_entry_point("creative")
    builder.set_finish_point("creative")
    return builder.compile()

# ========== MCP Server Setup ==========
server = Server("langgraph-mcp-server")

# Global graphs
reasoning_graph = None
analysis_graph = None
creative_graph = None

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available tools provided by this MCP server"""
    return [
        Tool(
            name="reasoning_agent",
            description="Use LangGraph reasoning agent powered by Gemini to analyze and respond to questions",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to reason about"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="analysis_agent",
            description="Use LangGraph analysis agent to perform detailed analysis with structured output",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to analyze in detail"
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="creative_agent",
            description="Use LangGraph creative agent to generate creative content, stories, poems, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The creative prompt or request"
                    }
                },
                "required": ["prompt"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    """Handle tool calls from MCP clients"""
    global reasoning_graph, analysis_graph, creative_graph
    
    try:
        if name == "reasoning_agent":
            query = arguments.get("query", "")
            
            state = {
                "messages": [{"role": "user", "content": query}],
                "current_step": "reasoning",
                "result": None,
                "error": None
            }
            
            result = reasoning_graph.invoke(state)
            
            if result["error"]:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {result['error']}")]
                )
            
            return CallToolResult(
                content=[TextContent(type="text", text=result["result"])]
            )
        
        elif name == "analysis_agent":
            topic = arguments.get("topic", "")
            
            state = {
                "messages": [{"role": "user", "content": topic}],
                "current_step": "analysis",
                "result": None,
                "error": None
            }
            
            result = analysis_graph.invoke(state)
            
            if result["error"]:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {result['error']}")]
                )
            
            return CallToolResult(
                content=[TextContent(type="text", text=result["result"])]
            )
        
        elif name == "creative_agent":
            prompt = arguments.get("prompt", "")
            
            state = {
                "messages": [{"role": "user", "content": prompt}],
                "current_step": "creative",
                "result": None,
                "error": None
            }
            
            result = creative_graph.invoke(state)
            
            if result["error"]:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {result['error']}")]
                )
            
            return CallToolResult(
                content=[TextContent(type="text", text=result["result"])]
            )
        
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")]
            )
    
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool execution error: {str(e)}")]
        )

@server.list_resources()
async def handle_list_resources() -> List[Resource]:
    """List available resources"""
    return [
        Resource(
            uri="langgraph://workflows/info",
            name="LangGraph Workflows Info",
            description="Information about available LangGraph workflows",
            mimeType="application/json"
        ),
        Resource(
            uri="langgraph://graphs/structure",
            name="Graph Structure",
            description="Detailed structure of LangGraph workflows",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> GetResourceResult:
    """Handle resource requests"""
    if uri == "langgraph://workflows/info":
        info = {
            "workflows": [
                {
                    "name": "reasoning_agent",
                    "description": "General reasoning and question answering",
                    "nodes": ["reasoning"],
                    "model": "gemini-2.0-flash-exp"
                },
                {
                    "name": "analysis_agent", 
                    "description": "Detailed analysis with structured output",
                    "nodes": ["analysis"],
                    "model": "gemini-2.0-flash-exp"
                },
                {
                    "name": "creative_agent",
                    "description": "Creative content generation",
                    "nodes": ["creative"],
                    "model": "gemini-2.0-flash-exp"
                }
            ],
            "total_workflows": 3,
            "server_version": "1.0.0"
        }
        
        return GetResourceResult(
            contents=[TextContent(type="text", text=json.dumps(info, indent=2))]
        )
    
    elif uri == "langgraph://graphs/structure":
        structure = {
            "reasoning_graph": {
                "entry_point": "reasoning",
                "finish_point": "reasoning",
                "nodes": ["reasoning"],
                "edges": []
            },
            "analysis_graph": {
                "entry_point": "analysis",
                "finish_point": "analysis", 
                "nodes": ["analysis"],
                "edges": []
            },
            "creative_graph": {
                "entry_point": "creative",
                "finish_point": "creative",
                "nodes": ["creative"],
                "edges": []
            }
        }
        
        return GetResourceResult(
            contents=[TextContent(type="text", text=json.dumps(structure, indent=2))]
        )
    
    else:
        return GetResourceResult(
            contents=[TextContent(type="text", text=f"Resource not found: {uri}")]
        )

async def main():
    """Main function to start the MCP server"""
    global reasoning_graph, analysis_graph, creative_graph
    
    # Initialize LangGraph workflows
    print("🚀 Initializing LangGraph workflows...")
    reasoning_graph = create_reasoning_graph()
    analysis_graph = create_analysis_graph()
    creative_graph = create_creative_graph()
    print("✅ LangGraph workflows compiled successfully!")
    
    # Start MCP server
    print("🌟 Starting LangGraph MCP Server...")
    print("📡 Server provides 3 tools: reasoning_agent, analysis_agent, creative_agent")
    print("📚 Server provides 2 resources: workflow info and graph structure")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="langgraph-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
