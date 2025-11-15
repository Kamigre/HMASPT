"""
Supervisor Agent for monitoring and coordinating other agents.
Optionally uses LLM for decision-making (requires LangChain).
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np

try:
    from langchain import LLMChain, PromptTemplate
    from langchain.llms import LlamaCpp
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not installed. Supervisor will use rule-based logic only.")

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
    - Optional LLM-based analysis (requires LangChain)
    - Command issuing via MessageBus
    """
    
    message_bus: MessageBus
    logger: JSONLogger
    max_total_drawdown: float = 0.20
    storage_dir: str = "./storage"
    llm_model_path: Optional[str] = None
    temperature: float = 0.1
    llm: Optional[Any] = None
    explainer_chain: Optional[Any] = None

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.swarm = SwarmOrchestrator(agents=["rules", "llm"], strategy="consensus")
        self.graph = Graph(name="supervisor_decisions")

        if self.llm_model_path and LANGCHAIN_AVAILABLE:
            try:
                callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
                
                self.llm = LlamaCpp(
                    model_path=self.llm_model_path,
                    temperature=self.temperature,
                    callback_manager=callback_manager,
                    verbose=False,
                )
                
                self.explainer_chain = LLMChain(
                    llm=self.llm,
                    prompt=PromptTemplate(
                        input_variables=["metrics", "actions"],
                        template=(
                            "You are an expert quantitative supervisor overseeing algorithmic trading agents.\n"
                            "Given the portfolio metrics below, explain in natural language why the following actions are being taken.\n\n"
                            "Metrics:\n{metrics}\n\n"
                            "Actions:\n{actions}\n\n"
                            "Explain clearly and concisely the reasoning behind each action, including risk management implications."
                        )
                    ),
                )
                print(f"✅ LLM loaded: {self.llm_model_path}")
            except Exception as e:
                self.llm = None
                self.explainer_chain = None
                print(f"Warning: Could not load LLM model: {str(e)}")
        
        self.logger.log("supervisor", "init", {
            "max_total_drawdown": self.max_total_drawdown,
            "storage_dir": self.storage_dir,
            "llm_enabled": self.llm is not None
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

    def llm_based_analysis(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Query LLM for recommended action (if available).
        Falls back to rule-based analysis if LLM is unavailable.
        """
        if self.llm is None:
            if metrics.get("negatives", 0) > metrics.get("n_traces", 0) * 0.5:
                return {"action": "retrain_selector", "reason": "fallback_rule"}
            if metrics.get("max_drawdown", 0) > self.max_total_drawdown:
                return {"action": "reduce_risk", "reason": "fallback_rule"}
            return {"action": "no_op", "reason": "fallback_rule"}

        prompt = (
            f"Given these portfolio metrics: {json.dumps(metrics, default=str)}, "
            "suggest one high-level action (in JSON format) to improve portfolio stability or returns. "
            "Examples: {\"action\":\"rebalance\"}, {\"action\":\"freeze_agents\"}, {\"action\":\"retrain_selector\"}."
        )
        
        try:
            raw = self.llm(prompt)
            text = raw if isinstance(raw, str) else str(raw)
            
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                import re
                m = re.search(r"\{.*\}", text, flags=re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        pass
                return {"action": "review_required", "reason": "LLM_unstructured_output"}
        except Exception as e:
            self.logger.log("supervisor", "llm_error", {"error": str(e)})
            if metrics.get("max_drawdown", 0) > self.max_total_drawdown:
                return {"action": "reduce_risk", "reason": "llm_error_fallback"}
            return {"action": "no_op", "reason": "llm_error_fallback"}

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

        actions_rules = []
        if metrics_summary["negatives"] > len(operator_traces) * 0.6:
            actions_rules.append({"action": "retrain_selector", "reason": "systemic_underperformance"})
        if max_dd > self.max_total_drawdown:
            actions_rules.append({"action": "reduce_risk", "reason": "portfolio_drawdown_limit_exceeded"})

        llm_action = self.llm_based_analysis(metrics_summary)
        actions_llm = [llm_action] if llm_action else []

        actions = self.swarm.merge_actions(actions_rules + actions_llm)

        self.graph.add_node("PortfolioMetrics", "metrics")
        for act in actions:
            node_name = f"{act.get('action')} ({act.get('reason','')})"
            self.graph.add_node(node_name)
            self.graph.add_edge("PortfolioMetrics", node_name)

        explanation = ""
        if self.explainer_chain is not None and actions:
            try:
                explanation = self.explainer_chain.run(
                    metrics=json.dumps(metrics_summary, default=str),
                    actions=json.dumps(actions, default=str)
                )
            except Exception as e:
                explanation = f"Simple rule-based explanation: {len(actions)} actions triggered based on portfolio metrics."
        else:
            explanation = f"Rule-based analysis: {len(actions)} interventions recommended."

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
