"""
Supervisor Agent for monitoring and coordinating other agents. Explains actions
and risks.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import google.generativeai as genai
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import CONFIG
from agents.message_bus import MessageBus, JSONLogger, Graph, SwarmOrchestrator
from utils import half_life as compute_half_life, compute_spread
from statsmodels.tsa.stattools import adfuller

@dataclass
class SupervisorAgent:

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
        if self.use_gemini:
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

        act = action.get("action", "")
        commands = []

        if act == "reduce_risk":
            new_tc = action.get("new_transaction_cost", CONFIG.get("transaction_cost", 0.005) + 0.002)
            commands.append({"target": "operator", "command": "adjust_transaction_cost", "new_value": new_tc})
        elif act == "pause_agent":
            commands.append({"target": "selector", "command": "pause"})
            commands.append({"target": "operator", "command": "pause"})
        elif act == "resume_agent":
            commands.append({"target": "selector", "command": "resume"})
            commands.append({"target": "operator", "command": "resume"})
        elif "target" in action and "command" in action:
            commands.append(action)

        return commands

    def _call_gemini(self, prompt: str, system_instruction: Optional[str] = None, json_mode: bool = False) -> str:

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
        Generate natural language explanation of portfolio performance and actions using Gemini.
        
        Args:
            metrics: Dictionary of portfolio metrics
            actions: List of action dictionaries
            
        Returns:
            Natural language explanation string
        """
        if not self.use_gemini:
            if not actions:
                return "Portfolio is performing within acceptable parameters. No interventions required."
            return f"Rule-based analysis: {len(actions)} interventions recommended based on risk thresholds."

        system_instruction = """You are an expert quantitative supervisor overseeing algorithmic trading agents. 
          Your role is to analyze portfolio performance metrics and explain risk management decisions in clear, 
          professional language. Focus on actionable insights and risk implications."""
              
        prompt = f"""Analyze the following portfolio performance metrics and explain the rationale behind the recommended actions.

          PORTFOLIO METRICS:
          {json.dumps(metrics, indent=2, default=str)}

          RECOMMENDED ACTIONS:
          {json.dumps(actions, indent=2, default=str)}

          Provide a concise executive summary that:
          1. Highlights the most important performance indicators (Sharpe, Sortino, drawdown, win rate)
          2. Identifies key risks or concerns based on the metrics
          3. Explains why each action is being taken and its risk management purpose
          4. Offers a brief outlook on portfolio health

          Keep the explanation professional, clear, and actionable. Limit to 3-4 paragraphs."""
        
        try:
            explanation = self._call_gemini(prompt, system_instruction=system_instruction)
            return explanation
        except Exception as e:
            # Fallback explanation if Gemini fails
            fallback = f"Portfolio Analysis Summary:\n\n"
            fallback += f"Performance: Sharpe {metrics.get('sharpe_ratio', 0):.2f}, "
            fallback += f"Sortino {metrics.get('sortino_ratio', 0):.2f}, "
            fallback += f"Win Rate {metrics.get('win_rate', 0):.1%}\n\n"
            
            if actions:
                fallback += f"Risk Management: {len(actions)} action(s) triggered - "
                fallback += ", ".join([a['action'] for a in actions])
                fallback += "\n\n"
            else:
                fallback += "No risk interventions required at this time.\n\n"
            
            fallback += f"(Note: Gemini explanation unavailable: {str(e)})"
            return fallback


    def evaluate_portfolio(self, operator_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate portfolio performance across all operator runs with comprehensive aggregated metrics.
        
        Args:
            operator_traces: List of trace dictionaries from all trading pairs
            
        Returns:
            Dictionary containing metrics, actions, explanation, and commands issued
        """
        
        # Group traces by pair for per-pair analysis
        traces_by_pair = {}
        for t in operator_traces:
            pair = t.get("pair", "unknown")
            if pair not in traces_by_pair:
                traces_by_pair[pair] = []
            traces_by_pair[pair].append(t)
        
        # Aggregate metrics across all pairs
        all_returns = [t.get("return", 0.0) for t in operator_traces if "return" in t]
        all_pnls = [t.get("pnl", 0.0) for t in operator_traces if "pnl" in t]
        all_cum_rewards = [t.get("cum_reward", 0.0) for t in operator_traces if "cum_reward" in t]
        
        # Portfolio-level metrics
        total_pnl = sum(all_pnls)
        total_steps = len(operator_traces)
        max_dd = max([t.get("max_drawdown", 0.0) for t in operator_traces] + [0.0])
        
        # Calculate aggregate Sharpe and Sortino from all returns
        sharpe_ratio = self._calculate_portfolio_sharpe(all_returns)
        sortino_ratio = self._calculate_portfolio_sortino(all_returns)
        
        # Risk metrics: VaR and CVaR
        var_95, cvar_95 = 0.0, 0.0
        if all_returns:
            var_95 = float(np.percentile(all_returns, 5))
            cvar_95 = float(np.mean([r for r in all_returns if r <= var_95]) 
                          if any(r <= var_95 for r in all_returns) else var_95)
        
        # Win rate and profitability metrics
        positive_returns = sum(1 for r in all_returns if r > 0)
        negative_returns = sum(1 for r in all_returns if r < 0)
        win_rate = positive_returns / len(all_returns) if all_returns else 0.0
        
        # Per-pair summary statistics
        pair_summaries = []
        for pair, traces in traces_by_pair.items():
            pair_pnl = sum(t.get("pnl", 0.0) for t in traces)
            pair_cum_return = traces[-1].get("cum_reward", 0.0) if traces else 0.0
            pair_max_dd = max([t.get("max_drawdown", 0.0) for t in traces] + [0.0])
            pair_steps = len(traces)
            
            # Calculate per-pair Sharpe and Sortino
            pair_returns = [t.get("return", 0.0) for t in traces if "return" in t]
            pair_sharpe = self._calculate_portfolio_sharpe(pair_returns)
            pair_sortino = self._calculate_portfolio_sortino(pair_returns)
            
            pair_summaries.append({
                "pair": pair,
                "total_pnl": float(pair_pnl),
                "final_cum_return": float(pair_cum_return),
                "max_drawdown": float(pair_max_dd),
                "steps": pair_steps,
                "sharpe": float(pair_sharpe),
                "sortino": float(pair_sortino)
            })
        
        # Compile comprehensive metrics summary
        metrics_summary = {
            # Portfolio-level aggregates
            "total_pnl": float(total_pnl),
            "total_steps": int(total_steps),
            "max_drawdown": float(max_dd),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            
            # Risk metrics
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            
            # Performance distribution
            "win_rate": float(win_rate),
            "positive_returns": int(positive_returns),
            "negative_returns": int(negative_returns),
            "avg_return": float(np.mean(all_returns)) if all_returns else 0.0,
            "median_return": float(np.median(all_returns)) if all_returns else 0.0,
            "std_return": float(np.std(all_returns)) if all_returns else 0.0,
            
            # Trading activity
            "n_pairs": len(traces_by_pair),
            "n_traces": len(operator_traces),
            "avg_steps_per_pair": float(total_steps / len(traces_by_pair)) if traces_by_pair else 0.0,
            
            # Per-pair breakdown
            "pair_summaries": pair_summaries
        }
        
        # Rule-based actions based on aggregated metrics
        actions = []
        
        if max_dd > self.max_total_drawdown:
            actions.append({
                "action": "reduce_risk",
                "reason": "portfolio_drawdown_limit_exceeded",
                "severity": "high"
            })
        
        if sharpe_ratio < 0:
            actions.append({
                "action": "review_strategy",
                "reason": "negative_sharpe_ratio",
                "severity": "medium"
            })
        
        if win_rate < 0.4:
            actions.append({
                "action": "review_entry_exit",
                "reason": "low_win_rate",
                "severity": "medium"
            })
        
        if cvar_95 < -0.05:  # If tail risk exceeds 5%
            actions.append({
                "action": "reduce_position_sizes",
                "reason": "excessive_tail_risk",
                "severity": "high"
            })
        
        # Update graph with portfolio metrics
        self.graph.add_node("PortfolioMetrics", "metrics")
        for pair_summary in pair_summaries:
            pair_node = f"Pair: {pair_summary['pair']}"
            self.graph.add_node(pair_node, "pair")
            self.graph.add_edge("PortfolioMetrics", pair_node)
        
        for act in actions:
            node_name = f"{act.get('action')} ({act.get('reason', '')})"
            self.graph.add_node(node_name, "action")
            self.graph.add_edge("PortfolioMetrics", node_name)
        
        # Generate explanation
        explanation = self.generate_explanation(metrics_summary, actions)
        
        # Issue commands based on actions
        commands_issued = 0
        for act in actions:
            commands = self._action_to_commands(act)
            for cmd in commands:
                target = cmd.pop("target", None)
                if target:
                    self.message_bus.send_command(target, cmd)
                    commands_issued += 1
        
        # Compile final summary
        summary = {
            "metrics": metrics_summary,
            "actions": actions,
            "explanation": explanation,
            "commands_issued": commands_issued
        }
        
        # Log comprehensive evaluation
        self.logger.log("supervisor", "portfolio_evaluated", summary)
        self.graph.export(os.path.join(self.storage_dir, "supervisor_graph.json"))
        
        return summary


def _calculate_portfolio_sharpe(self, returns: List[float], 
                                risk_free_rate: float = None) -> float:
    """Calculate annualized Sharpe ratio from returns."""
    if risk_free_rate is None:
        risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
    
    if len(returns) == 0:
        return 0.0
    
    rf_daily = risk_free_rate / 252
    excess_returns = [r - rf_daily for r in returns]
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    return (mean_excess / std_excess) * np.sqrt(252)


def _calculate_portfolio_sortino(self, returns: List[float], 
                                 risk_free_rate: float = None) -> float:
    """Calculate annualized Sortino ratio from returns."""
    if risk_free_rate is None:
        risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
    
    if len(returns) == 0:
        return 0.0
    
    rf_daily = risk_free_rate / 252
    excess_returns = [r - rf_daily for r in returns]
    downside_returns = [r for r in excess_returns if r < 0]
    
    if len(downside_returns) == 0:
        return float('inf')
    
    mean_excess = np.mean(excess_returns)
    downside_std = np.sqrt(np.mean([r**2 for r in downside_returns]))
    
    if downside_std == 0:
        return float('inf')
    
    return (mean_excess / downside_std) * np.sqrt(252)
    
    def _calculate_portfolio_sharpe(self, returns: List[float], 
                                    risk_free_rate: float = None) -> float:
        """Calculate annualized Sharpe ratio from returns."""
        if risk_free_rate is None:
            risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
        
        if len(returns) == 0:
            return 0.0
        
        rf_daily = risk_free_rate / 252
        excess_returns = [r - rf_daily for r in returns]
        
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        
        if std_excess == 0:
            return 0.0
        
        return (mean_excess / std_excess) * np.sqrt(252)
    
    def _calculate_portfolio_sortino(self, returns: List[float], 
                                     risk_free_rate: float = None) -> float:
        """Calculate annualized Sortino ratio from returns."""
        if risk_free_rate is None:
            risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
        
        if len(returns) == 0:
            return 0.0
        
        rf_daily = risk_free_rate / 252
        excess_returns = [r - rf_daily for r in returns]
        downside_returns = [r for r in excess_returns if r < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        mean_excess = np.mean(excess_returns)
        downside_std = np.sqrt(np.mean([r**2 for r in downside_returns]))
        
        if downside_std == 0:
            return float('inf')
        
        return (mean_excess / downside_std) * np.sqrt(252)

    def validate_pairs(self, df_pairs: pd.DataFrame, validation_window: Tuple[pd.Timestamp, pd.Timestamp],
        half_life_max: float = 60, min_crossings_per_year: int = 24) -> pd.DataFrame:
    
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
