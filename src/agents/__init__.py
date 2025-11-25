"""
HMASPT Agents Module

Multi-agent system for pairs trading:
- MessageBus: Inter-agent communication
- SelectorAgent: TGNN-based pair selection
- OperatorAgent: RL-based trading execution
- SupervisorAgent: Portfolio monitoring and coordination
"""

from .message_bus import MessageBus, JSONLogger, Graph, SwarmOrchestrator
from .operator_agent import OperatorAgent, train_operator_on_pairs, PairTradingEnv
from .supervisor_agent import SupervisorAgent
from .selector_agent import OptimizedSelectorAgent, SimplifiedMemoryTGNN

__all__ = [
    "MessageBus",
    "JSONLogger",
    "Graph",
    "SelectorAgent",
    "OperatorAgent",
    "SupervisorAgent",
    "train_operator_on_pairs",
    "validate_pairs",
    "run_operator_holdout"
]
