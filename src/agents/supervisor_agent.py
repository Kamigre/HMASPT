import os
import json
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import google.generativeai as genai
from statsmodels.tsa.stattools import adfuller

# Local imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import CONFIG
    from agents.message_bus import JSONLogger
    from utils import half_life as compute_half_life, compute_spread
except ImportError:
    CONFIG = {"risk_free_rate": 0.04}
    JSONLogger = None
    compute_half_life = lambda x: 10
    compute_spread = lambda x, y: x - y

@dataclass
class SupervisorAgent:
    
    logger: Optional[JSONLogger] = None
    df: pd.DataFrame = None 
    storage_dir: str = "./storage"
    gemini_api_key: Optional[str] = None
    model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    use_gemini: bool = True
    
    # Check frequency: Every 3 days
    check_frequency: int = 3 
    
    monitoring_state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        # Gemini setup
        if self.use_gemini:
            try:
                api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.client = genai.GenerativeModel(model_name=self.model)
                else:
                    self.use_gemini = False
            except Exception:
                self.use_gemini = False

    def _log(self, event: str, details: Dict[str, Any]):
        if self.logger:
            self.logger.log("supervisor", event, details)

    # ===================================================================
    # 1. PAIR VALIDATION (Pre-Trading Check)
    # ===================================================================
    
    def validate_pairs(
        self, 
        df_pairs: pd.DataFrame, 
        validation_window: Tuple[pd.Timestamp, pd.Timestamp],
        half_life_max: float = 60,
        min_crossings_per_year: int = 12
    ) -> pd.DataFrame:
        
        start, end = validation_window
        validated = []

        # Pivot price data for fast access
        if self.df is None or self.df.empty:
            print("⚠️ Supervisor has no data for validation.")
            return df_pairs

        prices = self.df.pivot(
            index="date",
            columns="ticker",
            values="adj_close"
        ).sort_index()

        print(f"\n🔍 Validating {len(df_pairs)} pairs...")
        
        for idx, row in df_pairs.iterrows():
            x, y = row["x"], row["y"]

            if x not in prices.columns or y not in prices.columns:
                continue

            series_x = prices[x].loc[start:end].dropna()
            series_y = prices[y].loc[start:end].dropna()

            if min(len(series_x), len(series_y)) < 60:
                continue

            spread = compute_spread(series_x, series_y)
            if spread is None or len(spread) == 0:
                continue

            # Check stationarity (ADF Test)
            try:
                adf_res = adfuller(spread.dropna())
                adf_p = adf_res[1]
            except:
                adf_p = 1.0

            # Check mean reversion speed (Half-Life)
            hl = compute_half_life(spread.values)
            
            # Check Crossing Frequency
            centered = spread - spread.mean()
            crossings = (centered.shift(1) * centered < 0).sum()
            days = (series_x.index[-1] - series_x.index[0]).days
            crossings_per_year = float(crossings) / max(days / 365.0, 1e-9)

            # Decision Logic
            pass_criteria = (adf_p < 0.05) and (float(hl) < half_life_max) and (crossings_per_year >= min_crossings_per_year)

            validated.append({
                "x": x, "y": y,
                "score": float(row.get("score", np.nan)),
                "adf_p": float(adf_p),
                "half_life": float(hl),
                "crossings_per_year": crossings_per_year,
                "pass": bool(pass_criteria)
            })

        result_df = pd.DataFrame(validated)
        n_passed = result_df["pass"].sum() if len(result_df) > 0 else 0
        
        self._log("pairs_validated", {
            "n_total": len(df_pairs),
            "n_validated": len(result_df),
            "n_passed": int(n_passed)
        })
        
        print(f"✅ Validation complete: {n_passed}/{len(result_df)} pairs passed")
        return result_df

    # ===================================================================
    # 2. OPERATOR MONITORING
    # ===================================================================
    
    def check_operator_performance(
        self, 
        operator_traces: List[Dict[str, Any]],
        pair: Tuple[str, str],
        phase: str = "holdout"
    ) -> Dict[str, Any]:
        
        # 1. Config & Data Check
        rules = CONFIG.get("supervisor_rules", {}).get(phase, {}) if "supervisor_rules" in CONFIG else {}
        min_obs = rules.get("min_observations", 10)
        
        if len(operator_traces) < min_obs:
            return {"action": "continue", "severity": "info", "reason": "insufficient_data", "metrics": {}}

        pair_key = f"{pair[0]}-{pair[1]}"
        latest_trace = operator_traces[-1]
        days_in_pos = latest_trace.get('days_in_position', 0)
        
        # 2. State Initialization & Grace Period
        if pair_key not in self.monitoring_state:
            self.monitoring_state[pair_key] = {'strikes': 0, 'grace_period': True}

        # 3-day burn-in grace period
        if days_in_pos <= 3:
            self.monitoring_state[pair_key]['strikes'] = 0
            self.monitoring_state[pair_key]['grace_period'] = True
        else:
            self.monitoring_state[pair_key]['grace_period'] = False

        # Calculate LIVE metrics (only for immediate monitoring)
        metrics = self._compute_live_metrics(operator_traces)
        
        # ============================================================
        # A. IMMEDIATE KILL (Structural Breaks) - CHECK EVERY DAY
        # ============================================================
        
        # 1. Structural Break (Z-Score > 3.0)
        spread_history = [t['current_spread'] for t in operator_traces]
        if len(spread_history) > 20:
            spread_series = pd.Series(spread_history)
            rolling_mean = spread_series.rolling(window=20).mean().iloc[-1]
            rolling_std = spread_series.rolling(window=20).std().iloc[-1]
            
            if rolling_std > 1e-8:
                current_z = abs(latest_trace['current_spread'] - rolling_mean) / rolling_std
                
                if current_z > 3.0:
                    self._log("intervention_triggered", {"pair": pair, "reason": "structural_break_zscore", "z": current_z})
                    return {
                        'action': 'stop',
                        'severity': 'critical',
                        'reason': f'Structural Break: Z-Score {current_z:.2f} > 3.0',
                        'metrics': metrics
                    }

        # 2. Hard Drawdown Kill (> 15%)
        # Explicit kill switch regardless of strikes
        if metrics['drawdown'] > 0.15:
             return {
                'action': 'stop',
                'severity': 'critical',
                'reason': f'Hard Stop: Drawdown {metrics["drawdown"]:.1%} > 15%',
                'metrics': metrics
            }

        # ============================================================
        # B. PERIODIC REVIEW (Strikes System)
        # ============================================================
        
        is_check_day = (days_in_pos > 0) and (days_in_pos % self.check_frequency == 0)
        
        if not is_check_day:
            return {'action': 'continue', 'severity': 'info', 'reason': 'off_cycle', 'metrics': metrics}

        # Stalemate Check (30 days)
        if days_in_pos > 30:
            unrealized_pnl = latest_trace.get('unrealized_pnl', 0.0)
            if unrealized_pnl <= 0:
                 return {
                    'action': 'stop', 
                    'severity': 'warning',
                    'reason': f'Stalemate ({days_in_pos} days) & Negative PnL. Capital rotation.',
                    'metrics': metrics
                }

        # VIOLATION LOGIC
        violation = False
        violation_reason = ""
        
        # Warning Threshold: 10% Drawdown
        if metrics['drawdown'] > 0.1: 
            violation = True
            violation_reason = f"Drawdown {metrics['drawdown']:.1%} > 10%"
        
        # Efficiency Threshold: Bad Sharpe after 15 days
        elif metrics['sharpe'] < -0.5 and days_in_pos > 15: 
            violation = True
            violation_reason = f"Sharpe {metrics['sharpe']:.2f} (Inefficient Risk)"

        # TWO-STRIKE SYSTEM
        if violation:
            if self.monitoring_state[pair_key]['grace_period']:
                return {'action': 'continue', 'severity': 'info', 'reason': 'Grace Period', 'metrics': metrics}
            
            self.monitoring_state[pair_key]['strikes'] += 1
            strikes = self.monitoring_state[pair_key]['strikes']
            
            if strikes == 1:
                # STRIKE 1: WARN ONLY (No resizing)
                return {
                    'action': 'warn',
                    'severity': 'warning',
                    'reason': f'Strike 1/2: {violation_reason}. Monitoring closely.',
                    'metrics': metrics
                }
            elif strikes >= 2:
                # STRIKE 2: STOP
                return {
                    'action': 'stop',
                    'severity': 'critical',
                    'reason': f'Strike 2/2: {violation_reason}. Failed.',
                    'metrics': metrics
                }
        else:
            # Heal strikes if performance recovers (drawdown < 2.5% - half of warning)
            if self.monitoring_state[pair_key]['strikes'] > 0 and metrics['drawdown'] < 0.025:
                self.monitoring_state[pair_key]['strikes'] -= 1
                
        return {
            'action': 'continue',
            'severity': 'info',
            'reason': 'Performance nominal',
            'metrics': metrics
        }
        
    def _compute_live_metrics(self, traces):
        """Simple lightweight calculation for live monitoring only."""
        returns = [t.get("daily_return", 0) for t in traces]
        portfolio_values = [t.get("portfolio_value", 0) for t in traces]
        
        current_pv = portfolio_values[-1] if portfolio_values else 0
        peak_pv = max(portfolio_values) if portfolio_values else 1
        
        drawdown = (peak_pv - current_pv) / max(peak_pv, 1e-8)
        
        # Approximation for live check
        if len(returns) < 2: 
            sharpe = 0.0
        else:
            std = np.std(returns, ddof=1)
            sharpe = (np.mean(returns) / std * np.sqrt(252)) if std > 1e-8 else 0.0

        return {
            'drawdown': drawdown,
            'sharpe': sharpe,
            'total_steps': len(traces)
        }

    # ===================================================================
    # 3. FINAL EVALUATION (Post-Trading Aggregation)
    # ===================================================================
    
    def evaluate_portfolio(self, operator_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate complete portfolio performance by extracting PRE-CALCULATED
        metrics from the Operator's output.
        """
        
        # Group traces by pair
        traces_by_pair = {}
        for t in operator_traces:
            pair = t.get('pair')
            # Handle cases where pair might be a tuple or string "A-B"
            if isinstance(pair, (list, tuple)):
                pair_str = f"{pair[0]}-{pair[1]}"
            else:
                pair_str = str(pair)
            
            traces_by_pair.setdefault(pair_str, []).append(t)

        pair_summaries = []
        global_pnl = 0.0
        global_returns_acc = [] # Just for global VaR/CVaR if needed
        total_steps = 0
        
        # Iterate through pairs and extract Operator's calculated metrics
        for pair_str, traces in traces_by_pair.items():
            
            # Sort traces to ensure we get the final state
            sorted_traces = sorted(traces, key=lambda x: x.get('step', 0))
            final_trace = sorted_traces[-1]
            
            # 1. Identify where the metrics are stored.
            # They might be in a dedicated "holdout_complete" event or embedded in the last trace
            # Logic: We prefer explicit pre-calculated keys.
            
            # Default values (safe fallbacks)
            p_metrics = {
                "sharpe": 0.0,
                "sortino": 0.0, 
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "final_return": 0.0,
                "total_trades": 0
            }
            
            # Try to find a trace that looks like a Summary/Metrics object
            summary_trace = next((t for t in sorted_traces if "sharpe" in t and "win_rate" in t), None)
            
            if summary_trace:
                # Option A: We found a dedicated summary trace (from train_on_pairs or holdout_metrics)
                p_metrics["sharpe"] = summary_trace.get("sharpe", 0.0)
                p_metrics["sortino"] = summary_trace.get("sortino", 0.0)
                p_metrics["win_rate"] = summary_trace.get("win_rate", 0.0)
                p_metrics["max_drawdown"] = summary_trace.get("max_drawdown", 0.0)
                
                # Check for various keys for return (cum_return vs final_return)
                p_metrics["final_return"] = summary_trace.get("final_return", summary_trace.get("cum_return", 0.0))
                
            else:
                # Option B: We only have step traces. Extract cumulative values from the last step.
                # Operator instructions say "Take them from operator", implying we shouldn't recalculate.
                # However, step traces usually have cumulative return/drawdown maintained by the Env.
                p_metrics["final_return"] = final_trace.get("cum_return", 0.0) * 100 # usually stored as decimal in step, % in summary
                p_metrics["max_drawdown"] = final_trace.get("max_drawdown", 0.0)
                
                # If Sharpe isn't in the step trace, we leave it as 0.0 or mark as N/A to obey "no calc" rule,
                # but we will try to look for the 'sharpe' key just in case it was added to the last step.
                if "sharpe" in final_trace:
                     p_metrics["sharpe"] = final_trace["sharpe"]
            
            # PnL Calculation (Summing realized PnL from steps is reliable)
            pair_pnl = sum(t.get("realized_pnl_this_step", 0) for t in traces)
            global_pnl += pair_pnl
            total_steps += len(traces)
            
            pair_summaries.append({
                "pair": pair_str,
                "total_pnl": pair_pnl,
                "cum_return": p_metrics["final_return"],
                "sharpe": p_metrics["sharpe"],
                "sortino": p_metrics["sortino"],
                "max_drawdown": p_metrics["max_drawdown"],
                "win_rate": p_metrics["win_rate"],
                "steps": len(traces)
            })

        # Global stats aggregation (averaging pair metrics where appropriate)
        if pair_summaries:
            avg_sharpe = float(np.mean([p['sharpe'] for p in pair_summaries]))
            avg_sortino = float(np.mean([p['sortino'] for p in pair_summaries]))
            avg_return = float(np.mean([p['cum_return'] for p in pair_summaries]))
            avg_win_rate = float(np.mean([p['win_rate'] for p in pair_summaries]))
            max_dd_global = max([p['max_drawdown'] for p in pair_summaries] + [0])
        else:
            avg_sharpe = 0.0
            avg_sortino = 0.0
            avg_return = 0.0
            avg_win_rate = 0.0
            max_dd_global = 0.0

        metrics = {
            "total_pnl": global_pnl,
            "sharpe_ratio": avg_sharpe, # Average of Operator's Sharpes
            "sortino_ratio": avg_sortino,
            "max_drawdown": max_dd_global,
            "avg_return": avg_return,
            "win_rate": avg_win_rate,
            "total_steps": total_steps,
            "n_pairs": len(traces_by_pair),
            "pair_summaries": pair_summaries
        }
        
        actions = self._generate_portfolio_actions(metrics)
        explanation = self._generate_explanation(metrics, actions)
        
        return {"metrics": metrics, "actions": actions, "explanation": explanation}

    def _generate_portfolio_actions(self, metrics: Dict) -> List[Dict]:
        actions = []
        if metrics['max_drawdown'] > 0.30:
            actions.append({"action": "reduce_risk", "reason": "Portfolio drawdown > 30%", "severity": "high"})
        if metrics['sharpe_ratio'] < 0:
            actions.append({"action": "halt_trading", "reason": "Negative Average Sharpe Ratio", "severity": "high"})
        return actions
        
    def _generate_explanation(self, metrics: Dict, actions: List[Dict]) -> str:
        if not self.use_gemini:
            return self._fallback_explanation(metrics, actions)
        
        prompt = f"""
                You are the Chief Risk Officer (CRO) at a Quantitative Hedge Fund. 
                Your mandate is capital preservation and risk-adjusted growth.
                
                Analyze the following Pairs Trading Portfolio results.
                NOTE: These metrics are aggregated from the Operator's pre-calculated execution logs.
                
                --- METRICS ---
                {json.dumps(metrics, indent=2, default=str)}
                
                --- AUTOMATED ACTIONS TRIGGERED ---
                {json.dumps(actions, indent=2)}
                
                Produce a strict, institutional-grade Executive Risk Memo. 
                Avoid generic pleasantries. Focus on data interpretation.
                
                Structure your response into these three specific sections:
                
                ### 1. Performance Attribution
                - Evaluate the quality of returns (Average Sharpe > 2.0 is target).
                - Analyze the "Quality of Earnings": Compare average Win Rate vs. Total PnL.
                - Are we seeing consistent performance across pairs, or are outliers driving the stats?
                
                ### 2. Risk Decomposition
                - Analyze Tail Risk: Comment on the Max Drawdown relative to the average return.
                - Assess "Stalemate Risk": Look at 'total_steps'. Are capital turnover rates healthy?
                - Identify if specific pairs (in summary list) are failing.
                
                ### 3. CRO Verdict & Adjustments
                - Review the AUTOMATED ACTIONS above. Do you concur, or do you recommend a manual override?
                - Provide a final "Traffic Light" signal: GREEN (Scale Up), YELLOW (Maintain/Monitor), or RED (De-risk/Halt).
                
                """
        
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception:
            return self._fallback_explanation(metrics, actions)

    def _fallback_explanation(self, metrics, actions):
        return f"Portfolio Avg Sharpe: {metrics['sharpe_ratio']:.2f}. Max Drawdown: {metrics['max_drawdown']:.2%}. Avg Win Rate: {metrics['win_rate']:.1%}."

    def _basic_check(self, operator_traces, pair):
        return {"action": "continue", "reason": "basic_check_pass", "metrics": {}}
