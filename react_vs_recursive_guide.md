# ReAct Pattern vs Recursive/Looping Agents - Learning Guide

## Overview

Both patterns involve loops in LangGraph, but they serve **fundamentally different purposes** and have distinct architectures. Understanding the difference is crucial for building effective AI agents.

## ReAct Pattern (Step 3.2)

### Purpose
- **Reason** about a problem, then **Act** (use tools), then **Observe** results
- Problem-solving through iterative tool usage and reasoning
- External information gathering and synthesis

### Flow Pattern
```
Reason → Act → Observe → Reason → Act → Observe → ...
```

### Key Characteristics
- **External Focus**: Uses tools to interact with the world
- **Information Gathering**: Searches, calculates, retrieves data
- **Decision Making**: Chooses which tools to use based on reasoning
- **Synthesis**: Combines information from multiple sources

### Example Workflow
```
Question: "What's the weather in New York and how does it compare to London?"

Step 1: REASON → "I need weather data for both cities"
Step 2: ACT → Use get_weather("New York") 
Step 3: OBSERVE → "New York: 72°F, partly cloudy"
Step 4: REASON → "Now I need London weather to compare"
Step 5: ACT → Use get_weather("London")
Step 6: OBSERVE → "London: 15°C, rainy"
Step 7: REASON → "I have both, now I can compare"
Step 8: ACT → final_answer("NY is warmer and less rainy than London")
```

### Use Cases
- Research agents
- Question-answering bots
- Data analysis workflows
- Multi-step problem solving
- Tool orchestration

### State Schema Example
```python
class ReActState(TypedDict):
    question: str
    thought: str
    action: str
    action_input: str
    observation: str
    final_answer: str
    step_count: int
    history: List[Dict[str, Any]]
    tools_used: List[str]
```

## Recursive/Looping Agents (Step 3.3)

### Purpose
- **Improve** or **refine** the same output iteratively
- Self-correction and quality enhancement
- Internal optimization of a single artifact

### Flow Pattern
```
Generate → Critique → Improve → Critique → Improve → ...
```

### Key Characteristics
- **Internal Focus**: Self-evaluation and improvement
- **Quality Enhancement**: Iterative refinement of output
- **Self-Correction**: Identifies and fixes its own mistakes
- **Convergence**: Works toward a quality threshold

### Example Workflow
```
Topic: "Essay about AI"

Step 1: GENERATE → Write initial essay
Step 2: CRITIQUE → "Needs better examples and conclusion"
Step 3: IMPROVE → Rewrite with better examples
Step 4: CRITIQUE → "Good! But introduction could be stronger"
Step 5: IMPROVE → Enhance introduction
Step 6: CRITIQUE → "Perfect! SATISFIED"
```

### Use Cases
- Content refinement (essays, code, reports)
- Iterative design processes
- Quality assurance workflows
- Self-improving systems
- Creative writing enhancement

### State Schema Example
```python
class RecursiveState(TypedDict):
    topic: str
    current_output: str
    critique: str
    iteration_count: int
    max_iterations: int
    improvement_history: List[str]
    is_satisfied: bool
```

## Key Differences Summary

| Aspect | ReAct Pattern | Recursive/Looping |
|--------|---------------|-------------------|
| **Purpose** | Problem-solving with tools | Self-improvement of output |
| **Tools** | External tools (search, calc, API) | Internal critique/improvement |
| **Loop Type** | Reason→Act→Observe | Generate→Critique→Improve |
| **End Goal** | Answer using gathered info | Perfect a single artifact |
| **Information Source** | External APIs and tools | Internal LLM evaluation |
| **Decision Making** | Which tool to use next | How to improve current output |
| **Stopping Condition** | Question answered | Quality threshold met |
| **Example Use Case** | Research agent, QA bot | Essay writer, code reviewer |

## Graph Architecture Comparison

### ReAct Pattern Graph
```python
# Nodes
builder.add_node("reason", reason)
builder.add_node("act", act)
builder.add_node("observe", observe_and_update)
builder.add_node("finalize", finalize_react)

# Flow
builder.set_entry_point("reason")
builder.add_edge("reason", "act")
builder.add_edge("act", "observe")
builder.add_conditional_edges(
    "observe",
    should_continue,
    {
        "continue": "reason",  # Back to reasoning
        "finish": "finalize"
    }
)
```

### Recursive Pattern Graph
```python
# Nodes
builder.add_node("generate", generate_content)
builder.add_node("critique", critique_content)
builder.add_node("improve", improve_content)
builder.add_node("finalize", finalize_content)

# Flow
builder.set_entry_point("generate")
builder.add_edge("generate", "critique")
builder.add_conditional_edges(
    "critique",
    should_continue,
    {
        "continue": "improve",  # Improve current output
        "finish": "finalize"
    }
)
builder.add_edge("improve", "critique")  # Back to critique
```

## When to Use Which Pattern

### Use ReAct Pattern When:
- You need to gather information from multiple sources
- The problem requires external tools or APIs
- You're building a research or analysis agent
- The task involves decision-making about which actions to take
- You need to combine data from different domains

### Use Recursive/Looping When:
- You want to improve the quality of a single output
- The task involves iterative refinement
- You're building content creation tools
- Quality matters more than speed
- You want self-correcting behavior

## Code Examples

### ReAct Pattern Implementation
```python
def reason(state: ReActState) -> ReActState:
    """Decide what tool to use next"""
    # Analyze current situation and choose action
    
def act(state: ReActState) -> ReActState:
    """Execute the chosen tool"""
    # Use search, calculator, API, etc.
    
def observe(state: ReActState) -> ReActState:
    """Process tool results"""
    # Update state with observations
```

### Recursive Pattern Implementation
```python
def generate(state: RecursiveState) -> RecursiveState:
    """Create initial output"""
    # Generate first version
    
def critique(state: RecursiveState) -> RecursiveState:
    """Evaluate current output"""
    # Analyze and provide feedback
    
def improve(state: RecursiveState) -> RecursiveState:
    """Enhance based on critique"""
    # Refine and improve
```

## Best Practices

### For ReAct Pattern:
- Keep tool inventory organized and well-documented
- Implement proper error handling for tool failures
- Use structured reasoning prompts
- Maintain conversation context across steps
- Implement tool result validation

### For Recursive Pattern:
- Set clear quality criteria for stopping
- Implement max iteration limits to prevent infinite loops
- Track improvement history for transparency
- Use specific, actionable critique prompts
- Monitor convergence toward quality goals

## Advanced Combinations

You can combine both patterns:
- **ReAct + Recursive**: Use ReAct to gather information, then recursively improve the synthesis
- **Recursive + ReAct**: Improve a plan recursively, then execute it with ReAct
- **Parallel Processing**: Run multiple recursive agents, then use ReAct to synthesize results

## Troubleshooting Common Issues

### ReAct Pattern Issues:
- **Infinite loops**: Tool keeps returning same result
- **Tool failures**: External APIs not responding
- **Poor reasoning**: Agent chooses wrong tools
- **Context loss**: Forgetting previous observations

### Recursive Pattern Issues:
- **Never satisfied**: Critique always finds problems
- **Degradation**: Output gets worse with iterations
- **Repetitive cycles**: Same improvements repeatedly
- **Vague critique**: Non-actionable feedback

## Testing Strategy

### ReAct Pattern Testing:
1. Test individual tools separately
2. Verify reasoning logic with mock tools
3. Test with various question types
4. Validate tool selection accuracy
5. Check final answer quality

### Recursive Pattern Testing:
1. Test with different quality thresholds
2. Verify improvement convergence
3. Test maximum iteration limits
4. Validate critique accuracy
5. Check output quality progression

## Performance Considerations

### ReAct Pattern:
- Tool latency affects overall speed
- Parallel tool calls can improve performance
- Cache frequently used tool results
- Implement timeout handling

### Recursive Pattern:
- Each iteration adds computational cost
- Quality vs. speed trade-off
- Monitor iteration efficiency
- Implement early stopping for good-enough results

---

*This guide is part of the LangGraph Learning Plan - Phase 3: Agentic Workflows & LangGraph Patterns*