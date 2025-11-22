"""
Supervisor Agent for monitoring and coordinating other agents.
Uses Google Gemini API for decision-making.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Generative AI package not installed. Run: pip install google-generativeai")

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
    - Gemini-based analysis
    - Command issuing via MessageBus
    """
    
    message_bus: MessageBus = None
    logger: JSONLogger = None
    max_total_drawdown: float = 0.20
    storage_dir: str = "./storage"
    gemini_api_key: Optional[str] = None
    model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    use_gemini: bool = True

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        self.graph = Graph(name="supervisor_decisions")

        # Initialize Gemini client
        if self.use_gemini and GEMINI_AVAILABLE:
            try:
                # Get API key from parameter, environment, or config
                api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                
                if not api_key:
                    print("Warning: No Gemini API key provided. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
                    self.use_gemini = False
                else:
                    # Configure Gemini
                    genai.configure(api_key=api_key)
                    
                    # Initialize model with generation config
                    generation_config = {
                        "temperature": self.temperature,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 2048,
                    }
                    
                    self.client = genai.GenerativeModel(
                        model_name=self.model,
                        generation_config=generation_config
                    )
                    print(f"✅ Gemini API initialized with model: {self.model}")
            except Exception as e:
                self.use_gemini = False
                print(f"Warning: Could not initialize Gemini client: {str(e)}")
        else:
            self.use_gemini = False
        
        self.logger.log("supervisor", "init", {
            "max_total_drawdown": self.max_total_drawdown,
            "storage_dir": self.storage_dir,
            "gemini_enabled": self.use_gemini,
            "model": self.model if self.use_gemini else None
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

        if act == "reduce_risk":
            new_tc = action.get("new_transaction_cost", CONFIG.get("transaction_cost", 0.005) + 0.002)
            commands.append({"target": "operator", "command": "adjust_transaction_cost", "new_value": new_tc})
        elif act in ["freeze_agents", "pause_agents"]:
            commands.append({"target": "selector", "command": "pause"})
            commands.append({"target": "operator", "command": "pause"})
        elif act == "resume_agents":
            commands.append({"target": "selector", "command": "resume"})
            commands.append({"target": "operator", "command": "resume"})
        elif "target" in action and "command" in action:
            commands.append(action)

        return commands

    def _call_gemini(self, prompt: str, system_instruction: Optional[str] = None, json_mode: bool = False) -> str:
        """
        Call Gemini API with given prompt.
        
        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction for the model
            json_mode: Whether to request JSON output
            
        Returns:
            Response text from the model
        """
        try:
            # Add JSON instruction if needed
            if json_mode:
                prompt = f"{prompt}\n\nIMPORTANT: Respond with valid JSON only, no additional text or formatting."
            
            # Create a new model instance with system instruction if provided
            if system_instruction:
                generation_config = {
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
                model = genai.GenerativeModel(
                    model_name=self.model,
                    generation_config=generation_config,
                    system_instruction=system_instruction
                )
            else:
                model = self.client
            
            response = model.generate_content(prompt)
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts'):
                return ''.join(part.text for part in response.parts if hasattr(part, 'text'))
            else:
                raise Exception("Unexpected response format from Gemini")
                
        except Exception as e:
            raise Exception(f"Gemini API call failed: {str(e)}")

    def generate_explanation(self, metrics: Dict[str, Any], actions: List[Dict[str, Any]]) -> str:
        """
        Generate natural language explanation for actions using Gemini.
        """
        if not self.use_gemini or not actions:
            return f"Rule-based analysis: {len(actions)} interventions recommended."

        system_instruction = "You are an expert quantitative supervisor overseeing algorithmic trading agents."
        
        prompt = f"""Given the portfolio metrics below, explain in natural language why the following actions are being taken.

                  Metrics:
                  {json.dumps(metrics, indent=2, default=str)}

                  Actions:
                  {json.dumps(actions, indent=2, default=str)}

                  Explain clearly and concisely the reasoning behind each action, including risk management implications."""
        
        try:
            return self._call_gemini(prompt, system_instruction=system_instruction)
        except Exception as e:
            return f"Simple rule-based explanation: {len(actions)} actions triggered based on portfolio metrics. (Gemini explanation failed: {str(e)})"

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
        actions = []
        if max_dd > self.max_total_drawdown:
            actions.append({"action": "reduce_risk", "reason": "portfolio_drawdown_limit_exceeded"})

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
    # ---------------- Validation ----------------
    def validate_pairs(
        self,
        df_pairs: pd.DataFrame,
        validation_window: Tuple[pd.Timestamp, pd.Timestamp],
        half_life_max: float = 60,
        min_crossings_per_year: int = 24
    ) -> pd.DataFrame:
        """
        Validates statistically whether candidate pairs are cointegrated.
        - Runs ADF test for stationarity of the spread.
        - Computes half-life (mean reversion speed).
        - Counts zero-crossings per year (signal frequency).
    
        The Selector produces a dataframe of pairs with scores.
        The Supervisor validates them statistically before allowing the Operator to trade them.
        """
    
        start, end = validation_window
        validated = []
    
        # Extract clean price matrix
        prices = self.df.pivot(
            index="date",
            columns="ticker",
            values="adj_close"
        ).sort_index()
    
        for _, row in df_pairs.iterrows():
            # Allow Supervisor to break if a pause/freeze command arrives
            self._check_for_commands()
    
            x, y = row["x"], row["y"]
    
            # Ensure both tickers exist
            if x not in prices.columns or y not in prices.columns:
                continue
    
            # Extract validation window prices
            series_x = prices[x].loc[start:end].dropna()
            series_y = prices[y].loc[start:end].dropna()
    
            # Require minimum length for statistical validity
            if min(len(series_x), len(series_y)) < 60:
                continue
    
            spread = compute_spread(series_x, series_y)
            if spread is None or len(spread) == 0:
                continue
    
            # Crossing frequency
            centered = spread - spread.mean()
            crossings = (centered.shift(1) * centered < 0).sum()
    
            days = (
                (series_x.index[-1] - series_x.index[0]).days
                if len(series_x.index) > 1 else 0
            )
    
            crossings_per_year = float(crossings) / max(days / 252.0, 1e-9)
    
            if crossings_per_year < min_crossings_per_year:
                continue
    
            # ADF test
            try:
                adf_res = adfuller(spread.dropna())
                adf_stat, adf_p = adf_res[0], adf_res[1]
            except Exception:
                adf_p = 1.0
    
            # Half-life
            hl = compute_half_life(spread.values)
            try:
                hl_val = float(hl)
            except Exception:
                hl_val = float("inf")
    
            # Decision
            pass_criteria = (adf_p < 0.05) and (hl_val < half_life_max)
    
            validated.append({
                "x": x,
                "y": y,
                "score": float(row.get("score", np.nan)),
                "adf_p": float(adf_p),
                "half_life": hl_val,
                "crossings_per_year": float(crossings_per_year),
                "pass": bool(pass_criteria)
            })
    
        # Logging
        self._log_event("pairs_validated", {"n_validated": len(validated)})
    
        return pd.DataFrame(validated)
