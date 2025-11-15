# HMASPT Agent System Documentation

## Overview

The Hierarchical Multi-Agent System consists of four main components that work together to identify and trade cointegrated stock pairs.

## Agents

### 1. MessageBus (`src/agents/message_bus.py`)

**Purpose**: Thread-safe communication infrastructure for inter-agent messaging.

**Components**:
- **MessageBus**: Central event queue and command routing
- **JSONLogger**: JSONL-based event logging for auditing
- **Graph**: Decision flow visualization
- **SwarmOrchestrator**: Multi-agent decision merging

**Usage**:
```python
from src.agents import MessageBus, JSONLogger

bus = MessageBus()
logger = JSONLogger(path="traces/events.jsonl")

# Publish events
bus.publish({"type": "event", "data": "..."})

# Send commands to specific agents
bus.send_command("operator", {"command": "pause"})

# Retrieve commands
cmd = bus.get_command_for("operator")
```

### 2. OperatorAgent (`src/agents/operator_agent.py`)

**Purpose**: Executes trades using Reinforcement Learning (PPO algorithm).

**Dependencies**: `gymnasium`, `stable-baselines3` (optional)

**Features**:
- **PairTradingEnv**: Gymnasium environment for pairs trading
  - Observation: z-score, volatility, half-life, correlation
  - Actions: -1 (short), 0 (neutral), +1 (long)
  - Rewards: Spread changes minus transaction costs

- **OperatorAgent**: Trains and evaluates trading models
  - PPO-based strategy learning
  - Performance metrics (Sharpe, Sortino, max drawdown)
  - Command handling (pause, resume, adjust costs)

**Usage**:
```python
from src.agents import OperatorAgent, MessageBus, JSONLogger

bus = MessageBus()
logger = JSONLogger(path="traces/operator.jsonl")
operator = OperatorAgent(bus, logger, storage_dir="models/")

# Train on a pair
trace = operator.train_on_pair(prices_df, "STOCK001", "STOCK002")
print(f"Sharpe Ratio: {trace['sharpe']:.2f}")
print(f"Max Drawdown: {trace['max_drawdown']:.2%}")
```

### 3. SupervisorAgent (`src/agents/supervisor_agent.py`)

**Purpose**: Monitors portfolio health and coordinates agent actions.

**Dependencies**: `langchain` (optional, for LLM-based analysis)

**Features**:
- Portfolio risk monitoring (drawdown, VaR, CVaR)
- Rule-based intervention triggers
- Optional LLM-based decision support
- Command issuing to Selector and Operator

**Monitoring Metrics**:
- Total returns
- Maximum drawdown
- Value at Risk (VaR 95%)
- Conditional Value at Risk (CVaR)
- Number of underperforming pairs

**Actions**:
- `retrain_selector`: Re-train pair selection model
- `reduce_risk`: Increase transaction costs
- `freeze_agents`: Pause all trading
- `resume_agents`: Resume operations

**Usage**:
```python
from src.agents import SupervisorAgent, MessageBus, JSONLogger

bus = MessageBus()
logger = JSONLogger(path="traces/supervisor.jsonl")
supervisor = SupervisorAgent(
    bus, logger,
    max_total_drawdown=0.20,
    storage_dir="./storage"
)

# Evaluate portfolio
operator_traces = [...]  # List of performance traces
summary = supervisor.evaluate_portfolio(operator_traces)

print(f"Total Return: {summary['metrics']['total_return']:.2%}")
print(f"Actions Taken: {len(summary['actions'])}")
print(f"Explanation: {summary['explanation']}")
```

### 4. SelectorAgent (Future)

**Status**: Planned - will use Temporal Graph Neural Networks (TGNN)

**Purpose**: Identify cointegrated stock pairs using graph-based learning.

**Planned Features**:
- Temporal graph construction from price correlations
- Memory-based TGNN for dynamic embeddings
- Pair scoring and validation
- Statistical tests (cointegration, half-life, ADF)

## System Workflow

```
1. SelectorAgent identifies promising pairs
   └─> Builds temporal graphs from price data
   └─> Trains TGNN model
   └─> Scores and validates pairs

2. OperatorAgent trains trading strategies
   └─> Creates RL environments for each pair
   └─> Trains PPO models
   └─> Evaluates performance

3. SupervisorAgent monitors and coordinates
   └─> Collects performance metrics
   └─> Detects risk conditions
   └─> Issues commands via MessageBus
   └─> Agents respond to commands

4. MessageBus enables communication
   └─> Events logged for audit trail
   └─> Commands routed to specific agents
   └─> Decision graphs exported
```

## Configuration

System parameters in `src/config.py`:

```python
CONFIG = {
    "cointegration_pvalue_threshold": 0.05,
    "half_life_min": 1,
    "half_life_max": 60,
    "transaction_cost": 0.0005,
    "rl_policy": "MlpPolicy",
    "rl_timesteps": 50000,
    "windows": [60]
}
```

## Event Logging

All agents log events in JSONL format:

```json
{
  "timestamp": "2024-01-15T10:30:00.000000",
  "agent": "operator",
  "event": "pair_trained",
  "details": {
    "pair": ["STOCK001", "STOCK002"],
    "sharpe": 1.45,
    "max_drawdown": 0.08
  }
}
```

## Dependencies

### Core (Required)
- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - ML utilities
- `statsmodels` - Statistical tests
- `scipy` - Scientific computing

### Advanced (Optional)
- `gymnasium` - RL environments
- `stable-baselines3` - PPO training
- `torch`, `torch-geometric` - TGNN models
- `langchain` - LLM integration

## Example: Complete System

```python
from src.agents import (
    MessageBus, JSONLogger,
    OperatorAgent, SupervisorAgent
)

# Initialize infrastructure
bus = MessageBus()
logger = JSONLogger()

# Create agents
operator = OperatorAgent(bus, logger)
supervisor = SupervisorAgent(bus, logger, max_total_drawdown=0.15)

# Train on pairs (example)
pairs = [("STOCK001", "STOCK002"), ("STOCK003", "STOCK004")]
traces = [operator.train_on_pair(prices, x, y) for x, y in pairs]

# Supervisor evaluation
summary = supervisor.evaluate_portfolio(traces)
print(f"Portfolio Health: {summary['metrics']}")

# Process any commands issued
while True:
    cmd = bus.get_command_for("operator")
    if cmd is None:
        break
    operator.apply_command(cmd)
```

## Notes

- This is a research/educational system
- Not suitable for live trading without extensive testing
- Always backtest thoroughly before deploying
- Monitor risk metrics continuously
