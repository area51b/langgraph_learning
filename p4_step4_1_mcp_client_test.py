#!/usr/bin/env python3
"""
MCP Client to test the LangGraph MCP Server
This demonstrates how to interact with MCP tools and resources
"""

import asyncio
import json
from typing import Any, Dict, List
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import subprocess
import sys

console = Console()

class LangGraphMCPClient:
    def __init__(self):
        self.session = None
        self.server_process = None
    
    async def start_server(self):
        """Start the MCP server as a subprocess"""
        try:
            console.print("🚀 [cyan]Starting LangGraph MCP Server...[/cyan]")
            self.server_process = subprocess.Popen(
                [sys.executable, "p4_step4_1_langgraph_mcp_server.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give server time to start
            await asyncio.sleep(2)
            console.print("✅ [green]Server started successfully![/green]")
            
        except Exception as e:
            console.print(f"❌ [red]Failed to start server: {e}[/red]")
            raise
    
    async def connect(self):
        """Connect to the MCP server"""
        try:
            console.print("🔗 [cyan]Connecting to MCP server...[/cyan]")
            
            # Connect via stdio
            read_stream, write_stream = await stdio_client(self.server_process)
            
            # Create session
            self.session = ClientSession(read_stream, write_stream)
            
            # Initialize the session
            await self.session.initialize()
            
            console.print("✅ [green]Connected to MCP server![/green]")
            
        except Exception as e:
            console.print(f"❌ [red]Failed to connect: {e}[/red]")
            raise
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server"""
        if not self.session:
            raise Exception("Not connected to MCP server")
        
        try:
            result = await self.session.list_tools()
            return [tool.model_dump() for tool in result.tools]
        except Exception as e:
            console.print(f"❌ [red]Error listing tools: {e}[/red]")
            return []
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources from the MCP server"""
        if not self.session:
            raise Exception("Not connected to MCP server")
        
        try:
            result = await self.session.list_resources()
            return [resource.model_dump() for resource in result.resources]
        except Exception as e:
            console.print(f"❌ [red]Error listing resources: {e}[/red]")
            return []
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on the MCP server"""
        if not self.session:
            raise Exception("Not connected to MCP server")
        
        try:
            result = await self.session.call_tool(name, arguments)
            
            # Extract text content from result
            if result.content:
                return result.content[0].text if result.content else "No content returned"
            return "No response from tool"
            
        except Exception as e:
            return f"Error calling tool: {e}"
    
    async def read_resource(self, uri: str) -> str:
        """Read a resource from the MCP server"""
        if not self.session:
            raise Exception("Not connected to MCP server")
        
        try:
            result = await self.session.read_resource(uri)
            
            # Extract text content from result
            if result.contents:
                return result.contents[0].text if result.contents else "No content returned"
            return "No content in resource"
            
        except Exception as e:
            return f"Error reading resource: {e}"
    
    async def close(self):
        """Close the connection and stop the server"""
        if self.session:
            await self.session.close()
        
        if self.server_process:
            self.server_process.terminate()
            try:
                await asyncio.wait_for(asyncio.create_task(
                    asyncio.to_thread(self.server_process.wait)
                ), timeout=5.0)
            except asyncio.TimeoutError:
                self.server_process.kill()

async def test_mcp_server():
    """Test the MCP server functionality"""
    client = LangGraphMCPClient()
    
    try:
        console.print("🧪 [bold cyan]Testing LangGraph MCP Server[/bold cyan]")
        console.print()
        
        # Start server and connect
        await client.start_server()
        await client.connect()
        
        # Test 1: List Tools
        console.print("1️⃣ [yellow]Available Tools[/yellow]")
        tools = await client.list_tools()
        
        if tools:
            table = Table(title="MCP Tools")
            table.add_column("Tool Name", style="cyan")
            table.add_column("Description", style="white")
            
            for tool in tools:
                table.add_row(tool["name"], tool["description"])
            
            console.print(table)
        else:
            console.print("❌ [red]No tools found[/red]")
        
        console.print()
        
        # Test 2: List Resources
        console.print("2️⃣ [yellow]Available Resources[/yellow]")
        resources = await client.list_resources()
        
        if resources:
            table = Table(title="MCP Resources")
            table.add_column("Resource Name", style="cyan")
            table.add_column("URI", style="magenta")
            table.add_column("Description", style="white")
            
            for resource in resources:
                table.add_row(resource["name"], resource["uri"], resource["description"])
            
            console.print(table)
        else:
            console.print("❌ [red]No resources found[/red]")
        
        console.print()
        
        # Test 3: Call Tools
        console.print("3️⃣ [yellow]Testing Tools[/yellow]")
        
        test_cases = [
            {
                "tool": "reasoning_agent",
                "args": {"query": "What is artificial intelligence and how does it work?"},
                "description": "Reasoning about AI"
            },
            {
                "tool": "analysis_agent", 
                "args": {"topic": "The impact of climate change on global economies"},
                "description": "Economic analysis"
            },
            {
                "tool": "creative_agent",
                "args": {"prompt": "Write a short poem about programming"},
                "description": "Creative writing"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            console.print(f"\n📤 [blue]Test {i}:[/blue] {test_case['description']}")
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task(f"Calling {test_case['tool']}...", total=None)
                result = await client.call_tool(test_case["tool"], test_case["args"])
                progress.update(task, completed=True)
            
            console.print(Panel(
                result,
                title=f"🤖 {test_case['tool']} Response",
                border_style="green"
            ))
        
        console.print()
        
        # Test 4: Read Resources
        console.print("4️⃣ [yellow]Reading Resources[/yellow]")
        
        resource_uris = [
            "langgraph://workflows/info",
            "langgraph://graphs/structure"
        ]
        
        for uri in resource_uris:
            console.print(f"\n📖 [blue]Reading:[/blue] {uri}")
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task("Reading resource...", total=None)
                content = await client.read_resource(uri)
                progress.update(task, completed=True)
            
            try:
                # Try to parse as JSON for better formatting
                parsed = json.loads(content)
                formatted_content = json.dumps(parsed, indent=2)
            except:
                formatted_content = content
            
            console.print(Panel(
                formatted_content,
                title=f"📄 Resource: {uri}",
                border_style="blue"
            ))
        
        console.print()
        console.print("🎉 [bold green]All MCP tests completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"❌ [red]Test failed: {e}[/red]")
    
    finally:
        await client.close()

async def interactive_mcp_chat():
    """Interactive chat using MCP tools"""
    client = LangGraphMCPClient()
    
    try:
        console.print("🚀 [bold cyan]Interactive LangGraph MCP Chat[/bold cyan]")
        console.print("Available commands:")
        console.print("  /reason <query>  - Use reasoning agent")
        console.print("  /analyze <topic> - Use analysis agent") 
        console.print("  /create <prompt> - Use creative agent")
        console.print("  /tools          - List available tools")
        console.print("  /resources      - List available resources")
        console.print("  /help           - Show this help")
        console.print("  /exit           - Exit the chat")
        console.print()
        
        # Start server and connect
        await client.start_server()
        await client.connect()
        
        while True:
            try:
                user_input = console.input("🧑 [bold blue]You:[/bold blue] ")
                
                if user_input.lower() in ["/exit", "/quit"]:
                    console.print("👋 [yellow]Goodbye![/yellow]")
                    break
                
                if user_input.lower() == "/help":
                    console.print("Available commands:")
                    console.print("  /reason <query>  - Use reasoning agent")
                    console.print("  /analyze <topic> - Use analysis agent")
                    console.print("  /create <prompt> - Use creative agent")
                    console.print("  /tools          - List available tools")
                    console.print("  /resources      - List available resources")
                    continue
                
                if user_input.lower() == "/tools":
                    tools = await client.list_tools()
                    for tool in tools:
                        console.print(f"  🔧 [cyan]{tool['name']}[/cyan]: {tool['description']}")
                    continue
                
                if user_input.lower() == "/resources":
                    resources = await client.list_resources()
                    for resource in resources:
                        console.print(f"  📄 [cyan]{resource['name']}[/cyan]: {resource['uri']}")
                    continue
                
                # Handle tool commands
                if user_input.startswith("/reason "):
                    query = user_input[8:].strip()
                    if query:
                        result = await client.call_tool("reasoning_agent", {"query": query})
                        console.print(f"🤖 [bold green]Reasoning Agent:[/bold green] {result}")
                    else:
                        console.print("❌ [red]Please provide a query after /reason[/red]")
                
                elif user_input.startswith("/analyze "):
                    topic = user_input[9:].strip()
                    if topic:
                        result = await client.call_tool("analysis_agent", {"topic": topic})
                        console.print(f"🤖 [bold green]Analysis Agent:[/bold green] {result}")
                    else:
                        console.print("❌ [red]Please provide a topic after /analyze[/red]")
                
                elif user_input.startswith("/create "):
                    prompt = user_input[8:].strip()
                    if prompt:
                        result = await client.call_tool("creative_agent", {"prompt": prompt})
                        console.print(f"🤖 [bold green]Creative Agent:[/bold green] {result}")
                    else:
                        console.print("❌ [red]Please provide a prompt after /create[/red]")
                
                elif user_input.strip():
                    console.print("❓ [yellow]Unknown command. Type /help for available commands.[/yellow]")
                
                console.print()
                
            except KeyboardInterrupt:
                console.print("\n👋 [yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"❌ [red]Error: {e}[/red]")
    
    except Exception as e:
        console.print(f"❌ [red]Failed to start interactive chat: {e}[/red]")
    
    finally:
        await client.close()

async def main():
    """Main function to choose between test and interactive mode"""
    console.print("🌟 [bold cyan]LangGraph MCP Client[/bold cyan]")
    console.print()
    console.print("Choose mode:")
    console.print("1. 🧪 Run MCP tests")
    console.print("2. 💬 Interactive MCP chat")
    
    choice = console.input("\nEnter your choice (1 or 2): ")
    
    if choice == "1":
        await test_mcp_server()
    elif choice == "2":
        await interactive_mcp_chat()
    else:
        console.print("❌ [red]Invalid choice. Please run again.[/red]")

if __name__ == "__main__":
    asyncio.run(main())
