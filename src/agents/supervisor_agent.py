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
            crossings_per_year = float(crossings) / max(days / 252.0, 1e-9)

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
        elif metrics['sharpe'] < -1.5 and days_in_pos > 15: 
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
                    'reason': f'Strike 2/2: {violation_reason}. Validation Failed.',
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
        returns = [t.get("daily_return", 0) for t in traces]
        portfolio_values = [t.get("portfolio_value", 0) for t in traces]
        
        current_pv = portfolio_values[-1] if portfolio_values else 0
        peak_pv = max(portfolio_values) if portfolio_values else 1
        
        drawdown = (peak_pv - current_pv) / max(peak_pv, 1e-8)
        
        return {
            'drawdown': drawdown,
            'sharpe': self._calculate_sharpe(returns),
            'total_steps': len(traces)
        }

    # ===================================================================
    # 3. FINAL EVALUATION (Post-Trading Aggregation)
    # ===================================================================
    
    def evaluate_portfolio(self, operator_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate complete portfolio performance."""
        
        # --- 1. PRE-CALCULATE GLOBAL TIME SERIES ---
        # We must aggregate value by time step to get true Portfolio Returns.
        # This handles correlations (e.g., hedging) correctly for Sharpe/Sortino.
        portfolio_by_step = {}
        for t in operator_traces:
            step = t['step']
            # Sum the portfolio_value of all pairs active at this step
            portfolio_by_step.setdefault(step, 0.0)
            portfolio_by_step[step] += t.get('portfolio_value', 0.0)

        # Sort steps and calculate GLOBAL returns stream
        sorted_steps = sorted(portfolio_by_step.keys())
        global_returns = []
        
        if len(sorted_steps) > 1:
            for i in range(1, len(sorted_steps)):
                curr_step = sorted_steps[i]
                prev_step = sorted_steps[i-1]
                
                curr_pv = portfolio_by_step[curr_step]
                prev_pv = portfolio_by_step[prev_step]
                
                # Avoid division by zero
                if prev_pv > 0:
                    ret = (curr_pv - prev_pv) / prev_pv
                else:
                    ret = 0.0
                global_returns.append(ret)

        # --- 2. GROUP TRACES BY PAIR ---
        traces_by_pair = {}
        for t in operator_traces:
            traces_by_pair.setdefault(t['pair'], []).append(t)

        pair_summaries = []
        
        # Initialize global accumulator for Total Realized PnL
        total_portfolio_realized_pnl = 0.0

        # --- 3. PROCESS EACH PAIR INDIVIDUALLY ---
        for pair, traces in traces_by_pair.items():
            pair_returns = []
            
            # Sort traces by step to ensure chronological order
            traces = sorted(traces, key=lambda x: x['step'])

            # Calculate returns per step for Pair-Specific Sharpe/Sortino
            for i in range(1, len(traces)):
                pv_curr = traces[i].get("portfolio_value", 0)
                pv_prev = traces[i-1].get("portfolio_value", 0)
                
                if pv_prev > 0:
                    ret = (pv_curr - pv_prev) / pv_prev
                else:
                    ret = 0.0
                
                pair_returns.append(ret)
                # NOTE: We do NOT append to global_returns here anymore.

            # --- CALCULATE PAIR PNL ---
            final_trace = traces[-1]
            pair_total_pnl = final_trace.get("realized_pnl", 0)

            # Add to global total
            total_portfolio_realized_pnl += pair_total_pnl

            # Calculate Pair Specific Metrics
            initial = traces[0]['portfolio_value']
            final = final_trace['portfolio_value']
            cum_ret = (final - initial) / initial if initial > 0 else 0
            
            pair_summaries.append({
                "pair": pair,
                "total_pnl": pair_total_pnl,
                "cum_return": cum_ret,
                "sharpe": self._calculate_sharpe(pair_returns),
                "sortino": self._calculate_sortino(pair_returns),
                "max_drawdown": max([t.get("max_drawdown", 0) for t in traces] + [0]),
                "steps": len(traces)
            })

        # --- 4. CALCULATE GLOBAL PORTFOLIO METRICS ---
        # Note: We now use `global_returns` for portfolio-wide risk metrics
        metrics = {
            "total_pnl": total_portfolio_realized_pnl,
            "sharpe_ratio": self._calculate_sharpe(global_returns),
            "sortino_ratio": self._calculate_sortino(global_returns),
            "max_drawdown": max([p['max_drawdown'] for p in pair_summaries] + [0]),
            "avg_return": float(np.mean(global_returns)) if global_returns else 0.0,
            "total_steps": len(operator_traces),
            "n_pairs": len(traces_by_pair),
            "pair_summaries": pair_summaries
        }
        
        # --- 5. CALCULATE WIN RATE (Based on closed trades) ---
        # Filter for steps where a trade was explicitly closed/adjusted
        closed_trades = [t for t in operator_traces if t.get("realized_pnl_this_step", 0) != 0]
        if closed_trades:
            # A win is defined as Positive PnL AFTER transaction costs
            wins = sum(1 for t in closed_trades if (t.get("realized_pnl_this_step", 0) - t.get("transaction_costs", 0)) > 0)
            metrics["win_rate"] = wins / len(closed_trades)
        else:
            metrics["win_rate"] = 0.0
        
        # --- 6. CALCULATE RISK METRICS (VaR/CVaR) ---
        # We use global_returns here to represent the risk of the whole portfolio
        if global_returns:
            metrics["var_95"] = float(np.percentile(global_returns, 5))
            tail_losses = [r for r in global_returns if r <= metrics["var_95"]]
            metrics["cvar_95"] = float(np.mean(tail_losses)) if tail_losses else metrics["var_95"]
        else:
            metrics["var_95"] = 0.0
            metrics["cvar_95"] = 0.0

        # --- 7. ADDITIONAL ACTIVITY METRICS ---
        metrics["positive_returns"] = sum(1 for r in global_returns if r > 0)
        metrics["negative_returns"] = sum(1 for r in global_returns if r < 0)
        metrics["median_return"] = float(np.median(global_returns)) if global_returns else 0.0
        metrics["std_return"] = float(np.std(global_returns)) if global_returns else 0.0
        metrics["avg_steps_per_pair"] = metrics["total_steps"] / max(metrics["n_pairs"], 1)
        
        # --- 8. GLOBAL CUMULATIVE RETURN ---
        if sorted_steps:
            start_pv = portfolio_by_step[sorted_steps[0]]
            end_pv = portfolio_by_step[sorted_steps[-1]]
            metrics["cum_return"] = (end_pv - start_pv) / start_pv if start_pv > 0 else 0.0
        else:
            metrics["cum_return"] = 0.0

        actions = self._generate_portfolio_actions(metrics)
        explanation = self._generate_explanation(metrics, actions)
        
        return {"metrics": metrics, "actions": actions, "explanation": explanation}

    def _generate_portfolio_actions(self, metrics: Dict) -> List[Dict]:
        actions = []
        if metrics['max_drawdown'] > 0.30:
            actions.append({"action": "reduce_risk", "reason": "Portfolio drawdown > 30%", "severity": "high"})
        if metrics['sharpe_ratio'] < 0:
            actions.append({"action": "halt_trading", "reason": "Negative Sharpe Ratio", "severity": "high"})
        return actions
        
    def _calculate_sharpe(self, returns: List[float]) -> float:
        if len(returns) < 2: return 0.0
        rf = CONFIG.get("risk_free_rate", 0.04) / 252
        exc = np.array(returns) - rf
        std = np.std(exc, ddof=1)
        return (np.mean(exc) / std) * np.sqrt(252) if std > 1e-8 else 0.0

    def _calculate_sortino(self, returns: List[float]) -> float:
        if len(returns) < 2: return 0.0
        rf = CONFIG.get("risk_free_rate", 0.04) / 252
        exc = np.array(returns) - rf
        down = exc[exc < 0]
        std = np.sqrt(np.mean(down**2)) if len(down) > 0 else 0.0
        return (np.mean(exc) / std) * np.sqrt(252) if std > 1e-8 else 0.0

    def _generate_explanation(self, metrics: Dict, actions: List[Dict]) -> str:
        if not self.use_gemini:
            return self._fallback_explanation(metrics, actions)
        
        prompt = f"""
                You are the Chief Risk Officer (CRO) at a Quantitative Hedge Fund. 
                Your mandate is capital preservation and risk-adjusted growth.
                
                Analyze the following Pairs Trading Portfolio results:
                
                --- METRICS ---
                {json.dumps(metrics, indent=2, default=str)}
                
                --- AUTOMATED ACTIONS TRIGGERED ---
                {json.dumps(actions, indent=2)}
                
                Produce a strict, institutional-grade Executive Risk Memo. 
                Avoid generic pleasantries. Focus on data interpretation.
                
                Structure your response into these three specific sections:
                
                ### 1. Performance Attribution
                - Evaluate the quality of returns (Sharpe > 2.0 is target).
                - Analyze the "Quality of Earnings": Compare Win Rate vs. Total PnL. (e.g., If Win Rate is high but PnL is low/negative, are we taking small profits and large losses?)
                - Comment on the disparity between Average Return and Median Return (skewness).
                
                ### 2. Risk Decomposition
                - Analyze Tail Risk: specific comment on Max Drawdown vs. VaR/CVaR (95%).
                - Assess "Stalemate Risk": Look at 'avg_steps_per_pair'. Are we holding positions too long for a mean-reversion strategy?
                - Identify if specific pairs are dragging down the aggregate (Concentration of loss).
                
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
        return f"Portfolio Sharpe: {metrics['sharpe_ratio']:.2f}. Drawdown: {metrics['max_drawdown']:.2%}. Win Rate: {metrics['win_rate']:.1%}."

    def _basic_check(self, operator_traces, pair):
        return {"action": "continue", "reason": "basic_check_pass", "metrics": {}}
