"""
HMASPT Agents Module

Multi-agent system for pairs trading:
- MessageBus: Inter-agent communication
- OperatorAgent: RL-based trading execution
- SupervisorAgent: Portfolio monitoring and coordination
- SelectorAgent: TGNN-based pair selection (requires PyTorch)
"""

from .message_bus import MessageBus, JSONLogger, Graph, SwarmOrchestrator
from .operator_agent import OperatorAgent, train_operator_on_pairs

try:
    from .operator_agent import PairTradingEnv
except:
    pass

from .supervisor_agent import SupervisorAgent

__all__ = [
    "MessageBus",
    "JSONLogger",
    "Graph",
    "SwarmOrchestrator",
    "OperatorAgent",
    "SupervisorAgent",
    "train_operator_on_pairs",
]
