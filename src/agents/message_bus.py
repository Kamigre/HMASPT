"""
Message Bus and supporting infrastructure for inter-agent communication.
Provides thread-safe messaging, logging, and orchestration utilities.
"""

import os
import json
import queue
import threading
import datetime
from typing import Dict, List, Any, Optional


class MessageBus:
    """
    Minimal thread-safe message bus for inter-agent communication.
    
    Features:
    - publish(event): Put an event into the global event queue
    - send_command(target, command): Put a command into target-specific queue
    - get_command_for(target): Non-blocking retrieval of commands
    - drain_events(): Get all pending events
    """

    def __init__(self):
        self.global_queue = queue.Queue()
        self.command_queues: Dict[str, queue.Queue] = {}
        self.lock = threading.Lock()

    def publish(self, event: Dict[str, Any]):
        """Publish a generic event (informational) to the global queue."""
        self.global_queue.put(event)

    def send_command(self, target: str, command: Dict[str, Any]):
        """Send a command dict to a specific agent (creates queue lazily)."""
        with self.lock:
            if target not in self.command_queues:
                self.command_queues[target] = queue.Queue()
            self.command_queues[target].put(command)

        cmd_event = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": "message_bus",
            "type": "command_sent",
            "target": target,
            "command": command,
        }
        self.publish(cmd_event)

    def get_command_for(self, target: str, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """Try to get a command for target — non-blocking by default (timeout=0)."""
        with self.lock:
            q = self.command_queues.get(target)
        if q is None:
            return None
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_events(self) -> List[Dict[str, Any]]:
        """Return all events currently in the global queue (non-blocking)."""
        events = []
        while True:
            try:
                events.append(self.global_queue.get_nowait())
            except queue.Empty:
                break
        return events


class JSONLogger:
    """
    JSONL trace logger for agent events.
    Logs events in JSON Lines format for easy parsing and analysis.
    """
    
    def __init__(self, path: str = "traces/agent_trace.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def log(self, agent: str, event: str, details: Dict[str, Any]):
        """Log an agent event in JSONL format."""
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": agent,
            "event": event,
            "details": details
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


class Graph:
    """
    Lightweight graph recorder for tracking agent decision flows.
    Exports to JSON format for visualization.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.nodes = []
        self.edges = []

    def add_node(self, node_name: str, node_type: str = "action"):
        """Add a node to the graph."""
        self.nodes.append({"name": node_name, "type": node_type})

    def add_edge(self, from_node: str, to_node: str):
        """Add an edge between two nodes."""
        self.edges.append({"from": from_node, "to": to_node})

    def export(self, path: str):
        """Export graph to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"nodes": self.nodes, "edges": self.edges}, f, indent=2)


class SwarmOrchestrator:
    """
    Lightweight swarm system to merge decisions from multiple agents.
    Strategy 'consensus': keep all unique actions.
    """
    
    def __init__(self, agents: List[str] = None, strategy: str = "consensus"):
        self.agents = agents or ["rules", "llm"]
        self.strategy = strategy

    def merge_actions(self, actions_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge actions from multiple agents, removing duplicates."""
        seen = set()
        merged = []
        for a in actions_list:
            key = (a.get("action"), a.get("reason", ""))
            if key not in seen:
                merged.append(a)
                seen.add(key)
        return merged
