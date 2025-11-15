"""
HMASPT Agents Module

Multi-agent system for pairs trading:
- MessageBus: Inter-agent communication
- SelectorAgent: TGNN-based pair selection (requires PyTorch)
- OperatorAgent: RL-based trading execution
- SupervisorAgent: Portfolio monitoring and coordination
"""

from .message_bus import MessageBus, JSONLogger, Graph, SwarmOrchestrator
from .operator_agent import OperatorAgent, train_operator_on_pairs
from .supervisor_agent import SupervisorAgent
from .selector_agent import SelectorAgent

try:
    from .operator_agent import PairTradingEnv
except:
    pass

try:
    from .selector_agent import MemoryTGNN
except:
    pass

__all__ = [
    "MessageBus",
    "JSONLogger",
    "Graph",
    "SwarmOrchestrator",
    "SelectorAgent",
    "OperatorAgent",
    "SupervisorAgent",
    "train_operator_on_pairs",
]
