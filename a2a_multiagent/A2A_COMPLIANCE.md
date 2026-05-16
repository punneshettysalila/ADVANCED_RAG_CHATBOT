# A2A Protocol Compliance & Architecture

## Overview

This multi-agent system follows the **Agent-to-Agent (A2A) protocol** with a strict **no auto-routing** policy. Every query is processed by ALL agents in parallel, ensuring transparency and comprehensive responses.

---

## Key Principle: No Auto-Routing

### ✅ What We Do (Current Implementation)
- **Parallel Execution**: All three agents (Research, Summarizer, General QA) process every query simultaneously
- **Transparent**: Users see responses from all agents
- **A2A Compliant**: No hidden routing logic — explicit agent coordination
- **ThreadPoolExecutor**: Concurrent execution for speed

### ❌ What We Don't Do
- **Auto-Classification**: No LLM classifies queries to route to specific agents
- **Conditional Routing**: No logic that sends query to Agent A OR Agent B
- **Sequential Pipelines**: No "if research needed, then summarize" logic
- **Agent Selection**: No automatic selection of "best agent" for a query

---

## A2A Protocol Compliance

### 1. Message Format ✅

**A2AMessage Structure:**
```python
{
    "message_id": "uuid",
    "role": "user" | "agent",
    "agent_id": "research-agent" | "summarizer-agent" | "general-qa-agent",
    "timestamp": 1234567890.123,
    "parts": [
        {"type": "text", "text": "..."},
        {"type": "code", "code": "...", "language": "python"},
        {"type": "data", "data": {...}, "mime_type": "application/json"},
        {"type": "error", "message": "...", "code": 500}
    ]
}
```

**Multi-Part Messages Supported:**
- `TextPart`: Plain text responses
- `CodePart`: Code snippets with language
- `DataPart`: Structured data with MIME type
- `ErrorPart`: Error messages with codes

---

### 2. Task Lifecycle ✅

**Task States:**
```
SUBMITTED → WORKING → COMPLETED
            ↓
          FAILED
            ↓
         CANCELLED
```

**A2ATask Structure:**
```python
{
    "task_id": "uuid",
    "session_id": "uuid",
    "state": "submitted" | "working" | "completed" | "failed" | "cancelled",
    "assigned_agent": null,  # NOT used for routing — tracking only
    "created_at": 1234567890.123,
    "updated_at": 1234567890.123,
    "messages": [A2AMessage, ...],
    "metadata": {}
}
```

**Important**: `assigned_agent` is **NOT** used for routing. In parallel execution mode, multiple agents process the same task. This field is for tracking/logging only.

---

### 3. Agent Discovery ✅

**Agent Card Format:**
```python
{
    "agent_id": "research-agent",
    "name": "Research Agent",
    "description": "Finds, explains and synthesises information",
    "capabilities": ["research"],
    "version": "1.0.0",
    "model": "mistralai/Mistral-7B-Instruct-v0.3"
}
```

**Discovery Endpoint:**
```
GET /agents
```

**Registry Methods:**
- `register(card)` - Register an agent
- `discover(capability)` - Find agents by capability
- `get(agent_id)` - Get specific agent
- `list_all()` - List all agents

---

### 4. Agent Capabilities ✅

Defined capabilities:
- `RESEARCH` - Deep information gathering
- `SUMMARIZE` - Distill complex information
- `GENERAL_QA` - Conversational responses
- `ORCHESTRATE` - Coordinate multiple agents
- `CODE_GEN` - Code generation (extensible)

---

## Architecture

### Execution Flow

```
User Query
    ↓
Streamlit UI (streamlit_app.py)
    ↓ POST /tasks/send
FastAPI Server (server.py)
    ↓
ParallelOrchestratorAgent
    │
    ├── Create A2ATask with user message
    │
    ├── ThreadPoolExecutor.submit()
    │   ├→ ResearchAgent.run(task_copy_1)
    │   ├→ SummarizerAgent.run(task_copy_2)
    │   └→ GeneralQAAgent.run(task_copy_3)
    │
    ├── Wait for all agents to complete
    │
    └── Combine all results
    
    ↓ SSE Stream
Streamlit UI renders all agent responses
```

### Why Parallel? Why No Routing?

**1. Transparency**
- Users see how EACH agent interprets their query
- No "black box" classification hiding which agent ran
- Full visibility into multi-agent reasoning

**2. Completeness**
- No single LLM classifier decides what the user "really wants"
- All perspectives are valuable (research depth + summary + conversational)
- Avoids classification errors that miss important aspects

**3. A2A Protocol Alignment**
- A2A emphasizes explicit agent coordination
- Auto-routing via hidden LLM classification violates transparency
- Parallel execution is honest: "here's what each agent thinks"

**4. User Control**
- Users can ignore agent outputs they don't need
- Better than system guessing and hiding agents

---

## File Organization

### Active Files (Current System)

| File | Purpose | A2A Role |
|------|---------|----------|
| `a2a/protocol.py` | Core A2A types | Protocol implementation |
| `agents/base.py` | BaseAgent class | Agent interface |
| `agents/specialized.py` | Agent implementations | Specialized capabilities |
| `agents/orchestrator_parallel.py` | **ACTIVE** orchestrator | Parallel coordination |
| `server.py` | FastAPI server | HTTP gateway + discovery |
| `streamlit_app.py` | Web UI | User interface |

### Deprecated Files

| File | Status | Reason |
|------|--------|--------|
| `agents/orchestrator.py` | **DEPRECATED** | Contains LangGraph auto-routing logic — violates no-routing policy |

**⚠️ DO NOT USE `orchestrator.py`** — it implements auto-classification routing which is incompatible with the parallel execution design.

---

## Code Examples

### Creating a Task (No Routing)

```python
# orchestrator_parallel.py
def _make_task(self, query: str, session_id: str) -> A2ATask:
    """Create a task from a query. No agent assignment per A2A protocol."""
    task = A2ATask(session_id=session_id)
    task.add_message(A2AMessage(role="user", parts=[TextPart(text=query)]))
    # Note: assigned_agent field is NOT set — parallel execution means
    # no single agent is "assigned" to the task
    return task
```

### Parallel Execution (No Routing)

```python
# orchestrator_parallel.py
def orchestrate(self, task: A2ATask) -> Dict:
    """Execute ALL agents in parallel and combine their results."""
    query = task.get_latest_user_message() or ""
    
    # Create SEPARATE tasks for each agent (independent execution)
    research_task = self._make_task(query, task.session_id)
    summarizer_task = self._make_task(query, task.session_id)
    qa_task = self._make_task(query, task.session_id)
    
    # Execute ALL agents simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_research = executor.submit(self._run_agent, self.research, research_task, "Research Agent")
        future_summarizer = executor.submit(self._run_agent, self.summarizer, summarizer_task, "Summarizer Agent")
        future_qa = executor.submit(self._run_agent, self.general_qa, qa_task, "General QA Agent")
        
        # Wait for ALL to complete
        results['research'] = future_research.result()
        results['summarizer'] = future_summarizer.result()
        results['general_qa'] = future_qa.result()
    
    # Combine ALL results (no filtering)
    return combined_response
```

### Agent Registration

```python
# orchestrator_parallel.py
def __init__(self, registry: AgentRegistry, hf_token: Optional[str] = None):
    super().__init__(...)
    
    # Initialize all specialized agents
    self.research = ResearchAgent(hf_token=hf_token)
    self.summarizer = SummarizerAgent(hf_token=hf_token)
    self.general_qa = GeneralQAAgent(hf_token=hf_token)
    
    # Register ALL agents for A2A discovery
    for agent in [self, self.research, self.summarizer, self.general_qa]:
        registry.register(agent.card)
```

---

## API Endpoints

### Agent Discovery (A2A Standard)

```bash
GET /agents
```

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "orchestrator",
      "name": "Parallel Orchestrator",
      "description": "Executes all agents simultaneously",
      "capabilities": ["orchestration"],
      "version": "1.0.0",
      "model": "mistralai/Mistral-7B-Instruct-v0.3"
    },
    {
      "agent_id": "research-agent",
      "name": "Research Agent",
      "description": "Finds, explains and synthesises information",
      "capabilities": ["research"],
      "version": "1.0.0",
      "model": "mistralai/Mistral-7B-Instruct-v0.3"
    },
    ...
  ]
}
```

### Task Submission (SSE Streaming)

```bash
POST /tasks/send
Content-Type: application/json

{
  "message": "Explain quantum computing",
  "session_id": "optional-uuid"
}
```

**Response (Server-Sent Events):**
```
data: {"type": "status", "message": "🚀 Starting all agents in parallel...", "step": "initialize"}

data: {"type": "status", "message": "🔍 Research Agent working...", "step": "research"}

data: {"type": "status", "message": "📝 Summarizer Agent working...", "step": "summarize"}

data: {"type": "status", "message": "💬 General QA Agent working...", "step": "qa"}

data: {"type": "chunk", "text": "\n## 🔍 Research Agent\n\n", "agent_id": "research-agent"}

data: {"type": "chunk", "text": "Quantum computing...", "agent_id": "research-agent"}

...

data: {"type": "done", "agents_used": ["research-agent", "summarizer-agent", "general-qa-agent"], "reasoning": "All agents executed in parallel", "intent": "parallel_execution", "task_id": "uuid"}
```

---

## Testing A2A Compliance

### Verify No Auto-Routing

**Test 1: All Agents Always Run**
```python
# Query ANY type of question
query1 = "What is 2+2?"  # Simple math
query2 = "Explain quantum mechanics"  # Research-heavy
query3 = "Hi, how are you?"  # Conversational

# For ALL queries, verify response contains:
assert "## 🔍 Research Agent" in response
assert "## 📝 Summarizer Agent" in response
assert "## 💬 General QA Agent" in response
```

**Test 2: Check agents_used Field**
```python
result = orchestrator.orchestrate(task)
assert result["agents_used"] == ["research-agent", "summarizer-agent", "general-qa-agent"]
assert result["intent"] == "parallel_execution"
assert result["reasoning"] == "All agents executed in parallel"
```

### Verify A2A Message Format

```python
# Check message structure
message = task.messages[0]
assert message.role in ["user", "agent"]
assert message.message_id is not None
assert message.timestamp > 0
assert len(message.parts) > 0
assert message.parts[0].type in ["text", "code", "data", "error"]
```

### Verify Task Lifecycle

```python
task = A2ATask()
assert task.state == TaskState.SUBMITTED

task.set_state(TaskState.WORKING)
assert task.state == TaskState.WORKING

task.set_state(TaskState.COMPLETED)
assert task.state == TaskState.COMPLETED
assert task.updated_at > task.created_at
```

---

## Migration Notes

### Previous Architecture (DEPRECATED)

The old `orchestrator.py` used LangGraph with auto-routing:

```python
# DEPRECATED - DO NOT USE
def _classify_node(self, state):
    """Use LLM to classify query → research_and_summarize | general_qa | summarize_only"""
    intent = llm_classify(query)  # ❌ Auto-routing
    return intent

def _route_after_classify(self, state):
    """Route to ONE agent based on classification"""
    if intent == "general_qa":
        return "general_qa"  # ❌ Only one agent runs
    elif intent == "summarize_only":
        return "summarize"
    else:
        return "research"
```

### Current Architecture (ACTIVE)

```python
# ACTIVE - orchestrator_parallel.py
def orchestrate(self, task):
    """Execute ALL agents in parallel — no classification, no routing"""
    with ThreadPoolExecutor() as executor:
        # ALL agents submit simultaneously
        future_research = executor.submit(...)
        future_summarizer = executor.submit(...)
        future_qa = executor.submit(...)
        # ALL results collected
```

---

## Extending the System

### Adding a New Agent (Maintains No-Routing Policy)

```python
# 1. Create specialized agent
class CodeGenAgent(BaseAgent):
    def __init__(self, hf_token=None):
        super().__init__(
            agent_id="codegen-agent",
            name="Code Generator",
            description="Generates code from natural language",
            capabilities=[AgentCapability.CODE_GEN],
            model="mistralai/Mistral-7B-Instruct-v0.3",
            hf_token=hf_token,
        )

# 2. Add to orchestrator (maintains parallel execution)
class ParallelOrchestratorAgent(BaseAgent):
    def __init__(self, registry, hf_token=None):
        ...
        self.codegen = CodeGenAgent(hf_token=hf_token)  # ← Add agent
        
        for agent in [self, self.research, self.summarizer, self.general_qa, self.codegen]:
            registry.register(agent.card)  # ← Register for discovery

# 3. Execute in parallel (no routing logic)
def orchestrate(self, task):
    ...
    codegen_task = self._make_task(query, task.session_id)
    with ThreadPoolExecutor(max_workers=4) as executor:  # ← Increase workers
        ...
        future_codegen = executor.submit(self._run_agent, self.codegen, codegen_task, "CodeGen Agent")
        results['codegen'] = future_codegen.result()  # ← Collect result
```

**Key Point**: New agents are added to parallel execution pool, NOT to routing logic.

---

## Summary

✅ **A2A Protocol Compliant**
- Proper message format with multi-part support
- Task lifecycle management (submitted → working → completed)
- Agent discovery via registry and HTTP endpoint
- Capability-based agent classification

✅ **No Auto-Routing**
- ALL agents process EVERY query in parallel
- NO LLM classification routing
- NO conditional agent selection
- Transparent multi-agent coordination

✅ **Transparent Design**
- Users see all agent responses
- No hidden routing logic
- Explicit parallel execution

---

## References

- **A2A Protocol**: Google's Agent-to-Agent communication standard
- **Active Orchestrator**: `agents/orchestrator_parallel.py`
- **Protocol Implementation**: `a2a/protocol.py`
- **Server**: `server.py` (FastAPI + SSE streaming)
- **UI**: `streamlit_app.py`
