import os
import json
import datetime
from typing import Dict, Any

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
