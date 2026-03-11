# HiveMind: Swarm Intelligence for LLM Research

> A decentralized multi-agent framework that orchestrates autonomous research agents exploring LLM training optimizations, sharing discoveries through pub/sub messaging, and collectively converging toward better models.

```
           ┌─────────────────────────────────────────────────────────────┐
           │                    HiveMind Swarm                           │
           ├─────────────────────────────────────────────────────────────┤
           │                                                           │
           │   ┌──────────┐    ┌──────────┐    ┌──────────┐            │
           │   │ Agent 0  │    │ Agent 1  │    │ Agent N  │            │
           │   │  Claude  │    │  Claude  │    │  Claude  │            │
           │   │ + Train  │    │ + Train  │    │ + Train  │            │
           │   └────┬─────┘    └────┬─────┘    └────┬─────┘            │
           │        │              │              │                   │
           │        └──────────────┼──────────────┘                   │
           │                       ▼                                  │
           │        ┌────────────────────────┐                       │
           │        │   Redis Pub/Sub Bus    │                       │
           │        │  - Discoveries         │                       │
           │        │  - Status Updates      │                       │
           │        │  - Agent Output Stream │                       │
           │        └────────────────────────┘                       │
           │                  ▲    ▲    ▲                            │
           │                  │    │    │                            │
           │        ┌─────────┴──┬─┴────┴──────┐                    │
           │        │            │             │                    │
           │   ┌────▼────┐ ┌─────▼────┐ ┌──────▼─────┐             │
           │   │Agent 0  │ │ Agent 1  │ │  Agent N   │             │
           │   │History  │ │ History   │ │  History   │             │
           │   └─────────┘ └───────────┘ └────────────┘             │
           │                                                           │
           └─────────────────────────────────────────────────────────────┘
```

---

## The Big Picture

**HiveMind** extends Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) from a single autonomous agent into a **decentralized swarm** of research agents that:

1. **Explore independently** — Each agent runs in an isolated git worktree, making incremental changes to `train.py`
2. **Share discoveries immediately** — When an agent finds an improvement, it broadcasts via Redis pub/sub
3. **Maintain collective memory** — Every agent keeps a complete history of all discoveries in memory
4. **Learn from each other** — Agents receive the full experiment history as context for their next iteration

The result is a **self-organizing research collective** where parallel exploration meets collaborative learning.

---

## Architecture Overview

### Core Components

| Component | File | Purpose |
|-----------|------|--------|
| **HiveAgent** | `hive_agent.py` | Autonomous research agent with experiment loop |
| **Orchestrator** | `orchestrator.py` | Spawns agents, coordinates discovery sharing |
| **PubSub** | `pubsub.py` | Redis-based message bus for inter-agent communication |
| **Discovery** | `discovery.py` | Structured data format for experiment results |
| **Dashboard** | `dashboard.py` | Real-time monitoring with SSE streaming |
| **Config** | `config.py` | Centralized configuration management |

### The Experiment Loop

Each agent runs this continuous loop:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Single Agent Loop                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                │
│   1. BUILD CONTEXT                                             │
│      └─ Gather all received discoveries from memory            │
│      └─ Format as markdown experiment history                  │
│                                                                │
│   2. SPIN UP CLAUDE                                            │
│      └─ Invoke claude CLI with experiment history as prompt    │
│      └─ Claude analyzes and modifies train.py                  │
│      └─ Stream output to frontend in real-time                 │
│                                                                │
│   3. RUN EXPERIMENT                                            │
│      └─ Execute train.py for 5 minutes (fixed time budget)     │
│      └─ Measure val_bpb (lower is better)                      │
│                                                                │
│   4. EVALUATE                                                  │
│      └─ Compare against baseline val_bpb                       │
│      └─ If improvement → Create Discovery object               │
│                                                                │
│   5. BROADCAST                                                 │
│      └─ Publish discovery to Redis pub/sub                     │
│      └─ All other agents receive and store in memory           │
│                                                                │
│   6. RESET AND REPEAT                                          │
│      └─ Git checkout to discard changes                        │
│      └─ Loop back to step 1 with updated discovery history     │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### Discovery Format

Every discovery contains both human-readable and machine-readable information:

```python
{
    "agent_id": "0",
    "timestamp": 1710259234.123,
    "description": "Increased DEPTH from 8 to 12, improved val_bpb by 0.002",
    "diff": "@@ -450,7 +450,7 @@\n-DEPTH = 8\n+DEPTH = 12",
    "baseline_bpb": 0.998,
    "new_bpb": 0.996,
    "improvement": 0.002,
    "training_seconds": 300.1,
    "mfu_percent": 39.8,
    "modified_parameters": ["DEPTH"],
    "experiment_type": "architecture"
}
```

---

## How It Works

### 1. Agent Isolation (Git Worktrees)

Each agent operates in its own git worktree branch:

```bash
$ git worktree list
./hivemind    main                    (main branch)
./worktrees/agent0_20260311...  hivemind/agent0_20260311...  (agent 0)
./worktrees/agent1_20260311...  hivemind/agent1_20260311...  (agent 1)
./worktrees/agent2_20260311...  hivemind/agent2_20260311...  (agent 2)
```

This provides:
- **Complete isolation**: Agents don't interfere with each other
- **Clean resets**: Each iteration starts from HEAD
- **Parallel execution**: Multiple agents run truly concurrently

### 2. Discovery Broadcasting (Redis Pub/Sub)

When Agent 0 discovers an improvement:

```python
# Agent 0 publishes discovery
await self.pubsub.publish_discovery(discovery)

# Orchestrator receives and distributes to all agents
async for discovery in self.pubsub.subscribe_discoveries():
    for agent in self.agents:
        await agent.receive_discovery(discovery)  # Stored in memory
```

### 3. Context Building (In Memory)

Before each Claude invocation, the agent builds context from ALL received discoveries:

```python
async def build_changes_context(self) -> str:
    """Build the changes context from received discoveries (in memory)."""
    if not self.received_discoveries:
        return "No prior discoveries yet."

    lines = ["# Experiment History", "## Discoveries from Other Agents", ""]

    for disc in sorted(self.received_discoveries, key=lambda d: d.timestamp):
        status = "✅" if disc.is_improvement else "❌"
        lines.append(f"### Agent {disc.agent_id} - {disc.description}")
        lines.append(f"- **Result**: {status} val_bpb {disc.baseline_bpb:.6f} → {disc.new_bpb:.6f}")
        # ... more formatting

    return "\n".join(lines)
```

This context is then passed to Claude as the prompt:

```
Analyze the following experiment history and propose improvements.

# Experiment History
## Discoveries from Other Agents

### Agent 1 - Increased DEPTH to 12
- **Result**: ✅ val_bpb 0.998000 → 0.996000 (Δ: 0.002000)
- **Type**: architecture
- **Modified**: DEPTH

Based on this context, modify train.py to improve val_bpb.
```

### 4. Real-Time Output Streaming

Claude's output is streamed line-by-line to the frontend:

```python
def run_claude_streaming():
    process = subprocess.Popen(
        ["claude", "-p"],
        input=prompt_content,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered
    )

    for line in process.stdout:
        # Stream each line to frontend via Redis
        asyncio.run_coroutine_threadsafe(
            self.pubsub.publish_output(str(self.agent_id), "claude", line),
            loop
        ).result(timeout=1.0)
```

---

## Quick Start

### Prerequisites

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/hivemind.git
cd hivemind

# 2. Ensure autoresearch submodule is initialized
git submodule update --init --recursive

# 3. Start Redis
redis-server

# 4. Install dependencies
uv pip install -r requirements.txt
```

### Running the Swarm

```bash
# Run orchestrator with 3 agents
python main.py --orchestrator --num-agents 3

# Run dashboard only
python main.py --dashboard

# Run both orchestrator and dashboard
python main.py --both --num-agents 3

# Debug mode (skip claude and experiment execution)
python main.py --both --num-agents 2 --experiments 10 --debug
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-agents` | 3 | Number of parallel agents |
| `--experiments` | -1 | Experiments per agent (-1 = infinite) |
| `--redis-url` | redis://localhost:6379 | Redis server URL |
| `--dashboard-port` | 8000 | Dashboard port |
| `--debug` | False | Skip claude/experiment (mock data) |

---

## Dashboard

The dashboard provides real-time visibility into the swarm:

```
┌─────────────────────────────────────────────────────────────────┐
│  HiveMind Dashboard                              [Live]         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Agents:                                                        │
│  ┌─────────┬─────────┬──────────┬──────────┐                   │
│  │ Agent   │ State   │ Experiments│ Discoveries│                │
│  ├─────────┼─────────┼──────────┼──────────┤                │
│  │   0     │ 🟢 Exp  │    12     │     3     │                │
│  │   1     │ 🟡 Set   │    8      │     1     │                │
│  │   2     │ 🟢 Exp  │    11     │     2     │                │
│  └─────────┴─────────┴──────────┴──────────┘                │
│                                                                 │
│  Best val_bpb Over Time:                                        │
│  ┌────┐                                                       │
│  │    │\                                                     │
│  │    │  \      Best: 0.994200                               │
│  │    │    \                                                 │
│  └────┴────┴────┴────┴────┴────┴────┴────┘                   │
│     0m   5m  10m  15m  20m  25m  30m  35m                     │
│                                                                 │
│  Live Output (Agent 0):                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ > Analyzing experiment history...                       │    │
│  │ > Found 3 improvements from other agents               │    │
│  │ > Considering DEPTH modification based on Agent 1's... │    │
│  │ > Modifying train.py...                                │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Overall system status |
| `/api/agents` | GET | All agent statuses |
| `/api/agents/{id}` | GET | Specific agent status |
| `/api/discoveries` | GET | All discoveries (paginated) |
| `/api/discoveries/best` | GET | Top discoveries by val_bpb |
| `/api/metrics/bpb-over-time` | GET | val_bpb progression |
| `/api/agent-output/{id}` | GET | Agent output history |
| `/api/agent-output/{id}/stream` | GET | Real-time SSE stream |

---

## Design Decisions

### Why Pub/Sub Instead of Centralized Storage?

**Pub/Sub** allows immediate broadcast without coordination overhead. Each agent maintains its own in-memory history, enabling:
- **Low latency**: Discoveries propagate instantly
- **Decentralization**: No single point of failure
- **Scalability**: Easy to add agents across machines

### Why In-Memory History Instead of Files?

Storing discoveries in memory (rather than writing `changes.md` files):
- **Performance**: No disk I/O on every iteration
- **Consistency**: Single source of truth per agent
- **Simplicity**: Clean state management

### Why Git Worktrees Instead of Clones?

Git worktrees share the object database:
- **Efficiency**: No redundant git objects
- **Speed**: Faster setup/teardown
- **Clean isolation**: Each agent has independent working directory

### Why Stream Output to Frontend?

Real-time streaming provides:
- **Visibility**: Watch agents think in real-time
- **Debugging**: See what's happening inside Claude
- **Engagement**: Watch the swarm evolve live

---

## Project Structure

```
hivemind/
├── autoresearch/          # Karpathy's autoresearch (submodule)
│   ├── train.py           # Training script (agents modify this)
│   ├── prepare.py         # Data prep (read-only)
│   └── CLAUDE.md          # System prompt for agents
├── hive_agent.py          # Individual research agent
├── orchestrator.py        # Agent manager and coordinator
├── pubsub.py              # Redis pub/sub message bus
├── discovery.py           # Discovery data structures
├── dashboard.py           # FastAPI monitoring UI
├── config.py              # Configuration management
├── main.py                # Entry point
├── test/
│   ├── test_integration.py
│   └── test_unit.py
└── worktrees/             # Agent worktrees (created at runtime)
    ├── agent0_20260311_...
    ├── agent1_20260311_...
    └── agent2_20260311_...
```

---

## Running Tests

```bash
# Run all tests
pytest test/ -v

# Run integration tests only
pytest test/test_integration.py -v

# Run unit tests only
pytest test/test_unit.py -v
```

---

## Future Extensions

### Cross-Machine Sharing

Agents on different machines can share discoveries by pointing to the same Redis server:

```python
# Machine 1
python main.py --orchestrator --num-agents 2 --redis-url redis://server:6379

# Machine 2
python main.py --orchestrator --num-agents 2 --redis-url redis://server:6379
```

### Agent Specialization

Agents could develop preferences for certain exploration types:
- **Architecture agents**: Focus on model structure changes
- **Hyperparameter agents**: Focus on learning rates, batch sizes
- **Optimizer agents**: Focus on optimization algorithms

### Discovery Aggregation

Periodically aggregate discoveries and update master branch with best findings.

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- **Andrej Karpathy** for [autoresearch](https://github.com/karpathy/autoresearch) - the foundation this project builds upon
- **Redis** for the pub/sub infrastructure
- **FastAPI** for the dashboard backend
- **Claude** for the autonomous research capabilities

---

## Contributing

Contributions welcome! Areas of interest:
- Better visualization in dashboard
- Agent specialization strategies
- Discovery aggregation algorithms
- Multi-GPU experiment support
