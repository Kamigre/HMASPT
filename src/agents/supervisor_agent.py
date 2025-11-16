"""
Supervisor Agent for monitoring and coordinating other agents.
Uses OpenAI API for decision-making.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not installed. Run: pip install openai")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import CONFIG
from agents.message_bus import MessageBus, JSONLogger, Graph, SwarmOrchestrator


@dataclass
class SupervisorAgent:
    """
    Supervisor Agent for monitoring and coordinating Selector and Operator agents.
    
    Features:
    - Portfolio risk monitoring
    - Rule-based interventions
    - OpenAI-based analysis
    - Command issuing via MessageBus
    """
    
    message_bus: MessageBus = None
    logger: JSONLogger = None
    max_total_drawdown: float = 0.20
    storage_dir: str = "./storage"
    openai_api_key: Optional[str] = None
    model: str = "gpt-4o-mini"  # or "gpt-4o", "gpt-3.5-turbo"
    temperature: float = 0.1
    use_openai: bool = True

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.swarm = SwarmOrchestrator(agents=["rules", "llm"], strategy="consensus")
        self.graph = Graph(name="supervisor_decisions")

        # Initialize OpenAI client
        if self.use_openai and OPENAI_AVAILABLE:
            try:
                # Get API key from parameter, environment, or config
                api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
                
                if not api_key:
                    print("Warning: No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
                    self.use_openai = False
                else:
                    # Initialize OpenAI client (for openai >= 1.0.0)
                    self.client = openai.OpenAI(api_key=api_key)
                    print(f"✅ OpenAI API initialized with model: {self.model}")
            except Exception as e:
                self.use_openai = False
                print(f"Warning: Could not initialize OpenAI client: {str(e)}")
        else:
            self.use_openai = False
        
        self.logger.log("supervisor", "init", {
            "max_total_drawdown": self.max_total_drawdown,
            "storage_dir": self.storage_dir,
            "openai_enabled": self.use_openai,
            "model": self.model if self.use_openai else None
        })

    def _action_to_commands(self, action: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert high-level action to concrete MessageBus commands.
        
        Example mappings:
        - retrain_selector -> {target: "selector", command: "retrain_tgn"}
        - reduce_risk -> {target: "operator", command: "adjust_transaction_cost"}
        - freeze_agents -> pause both selector and operator
        """
        act = action.get("action", "")
        commands = []

        if act == "retrain_selector":
            commands.append({"target": "selector", "command": "retrain_tgn", "epochs": action.get("epochs", 1)})
        elif act == "reduce_risk":
            new_tc = action.get("new_transaction_cost", CONFIG.get("transaction_cost", 0.005) + 0.002)
            commands.append({"target": "operator", "command": "adjust_transaction_cost", "new_value": new_tc})
        elif act in ["freeze_agents", "pause_agents"]:
            commands.append({"target": "selector", "command": "pause"})
            commands.append({"target": "operator", "command": "pause"})
        elif act == "resume_agents":
            commands.append({"target": "selector", "command": "resume"})
            commands.append({"target": "operator", "command": "resume"})
        elif act == "increase_capital_allocation":
            commands.append({"target": "operator", "command": "adjust_position_size", "new_value": action.get("new_value", 1.0)})
        elif "target" in action and "command" in action:
            commands.append(action)

        return commands

    def _call_openai(self, messages: List[Dict[str, str]], response_format: Optional[str] = None) -> str:
        """
        Call OpenAI API with given messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            response_format: Optional format specification ('json_object' for JSON mode)
            
        Returns:
            Response text from the model
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            # Add JSON mode if supported (GPT-4 and newer models)
            if response_format == "json_object" and "gpt-4" in self.model or "gpt-3.5" in self.model:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")

    def llm_based_analysis(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Query OpenAI for recommended action.
        Falls back to rule-based analysis if OpenAI is unavailable.
        """
        if not self.use_openai:
            # Fallback to rule-based logic
            if metrics.get("negatives", 0) > metrics.get("n_traces", 0) * 0.5:
                return {"action": "retrain_selector", "reason": "fallback_rule"}
            if metrics.get("max_drawdown", 0) > self.max_total_drawdown:
                return {"action": "reduce_risk", "reason": "fallback_rule"}
            return {"action": "no_op", "reason": "fallback_rule"}

        messages = [
            {
                "role": "system",
                "content": "You are an expert quantitative supervisor overseeing algorithmic trading agents. Respond only with valid JSON."
            },
            {
                "role": "user",
                "content": f"""Given these portfolio metrics: {json.dumps(metrics, default=str)}

                Suggest one high-level action to improve portfolio stability or returns.

                Respond with a JSON object with this structure:
                {{"action": "action_name", "reason": "explanation"}}

                Valid actions include:
                - "rebalance": Rebalance portfolio positions
                - "freeze_agents": Pause all trading agents
                - "retrain_selector": Retrain the stock selection model
                - "reduce_risk": Reduce position sizes and risk exposure
                - "increase_capital_allocation": Increase position sizes
                - "no_op": No action needed

                Provide your recommendation as JSON only."""
                            }
        ]
        
        try:
            response_text = self._call_openai(messages, response_format="json_object")
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                m = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        pass
                return {"action": "review_required", "reason": "LLM_unstructured_output"}
        except Exception as e:
            self.logger.log("supervisor", "openai_error", {"error": str(e)})
            # Fallback to rule-based
            if metrics.get("max_drawdown", 0) > self.max_total_drawdown:
                return {"action": "reduce_risk", "reason": "openai_error_fallback"}
            return {"action": "no_op", "reason": "openai_error_fallback"}

    def generate_explanation(self, metrics: Dict[str, Any], actions: List[Dict[str, Any]]) -> str:
        """
        Generate natural language explanation for actions using OpenAI.
        """
        if not self.use_openai or not actions:
            return f"Rule-based analysis: {len(actions)} interventions recommended."

        messages = [
            {
                "role": "system",
                "content": "You are an expert quantitative supervisor overseeing algorithmic trading agents."
            },
            {
                "role": "user",
                "content": f"""Given the portfolio metrics below, explain in natural language why the following actions are being taken.

                Metrics:
                {json.dumps(metrics, indent=2, default=str)}

                Actions:
                {json.dumps(actions, indent=2, default=str)}

                Explain clearly and concisely the reasoning behind each action, including risk management implications."""
                            }
        ]
        
        try:
            return self._call_openai(messages)
        except Exception as e:
            return f"Simple rule-based explanation: {len(actions)} actions triggered based on portfolio metrics. (OpenAI explanation failed: {str(e)})"

    def evaluate_portfolio(self, operator_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate portfolio performance and issue commands to agents.
        
        Returns summary of metrics and actions taken.
        """
        total_return = sum(t.get("cum_reward", 0.0) for t in operator_traces)
        max_dd = max([t.get("max_drawdown", 0.0) for t in operator_traces] + [0.0])
        returns = [t.get("cum_reward", 0.0) for t in operator_traces if "cum_reward" in t]

        var_95, cvar_95 = 0.0, 0.0
        if returns:
            var_95 = float(np.percentile(returns, 5))
            cvar_95 = float(np.mean([r for r in returns if r <= var_95]) if any(r <= var_95 for r in returns) else var_95)

        metrics_summary = {
            "total_return": float(total_return),
            "max_drawdown": float(max_dd),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "negatives": int(sum(1 for t in operator_traces if t.get("cum_reward", 0.0) < 0)),
            "n_traces": len(operator_traces)
        }

        # Rule-based actions
        actions_rules = []
        if metrics_summary["negatives"] > len(operator_traces) * 0.6:
            actions_rules.append({"action": "retrain_selector", "reason": "systemic_underperformance"})
        if max_dd > self.max_total_drawdown:
            actions_rules.append({"action": "reduce_risk", "reason": "portfolio_drawdown_limit_exceeded"})

        # LLM-based action
        llm_action = self.llm_based_analysis(metrics_summary)
        actions_llm = [llm_action] if llm_action else []

        # Merge actions
        actions = self.swarm.merge_actions(actions_rules + actions_llm)

        # Update graph
        self.graph.add_node("PortfolioMetrics", "metrics")
        for act in actions:
            node_name = f"{act.get('action')} ({act.get('reason','')})"
            self.graph.add_node(node_name)
            self.graph.add_edge("PortfolioMetrics", node_name)

        # Generate explanation
        explanation = self.generate_explanation(metrics_summary, actions)

        # Issue commands
        for act in actions:
            commands = self._action_to_commands(act)
            for cmd in commands:
                target = cmd.pop("target", None)
                if target:
                    self.message_bus.send_command(target, cmd)

        summary = {
            "metrics": metrics_summary,
            "actions": actions,
            "explanation": explanation,
            "commands_issued": sum(len(self._action_to_commands(a)) for a in actions)
        }

        self.logger.log("supervisor", "portfolio_evaluated", summary)
        self.graph.export(os.path.join(self.storage_dir, "supervisor_graph.json"))

        return summary
