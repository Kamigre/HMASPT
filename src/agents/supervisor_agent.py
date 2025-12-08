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
    # Fallback for standalone testing
    CONFIG = {"risk_free_rate": 0.04}
    JSONLogger = None
    compute_half_life = lambda x: 10
    compute_spread = lambda x, y: x - y

@dataclass
class SupervisorAgent:
    
    logger: Optional[JSONLogger] = None
    df: pd.DataFrame = None  # Full price data for validation
    storage_dir: str = "./storage"
    gemini_api_key: Optional[str] = None
    model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    use_gemini: bool = True
    
    # How often (in days) to perform deep performance reviews
    check_frequency: int = 5 
    
    # Internal state for tracking strikes and warnings per pair
    monitoring_state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize Gemini for explanations
        if self.use_gemini:
            try:
                api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
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
                    print(f"✅ Gemini API initialized")
                else:
                    self.use_gemini = False
                    print("⚠️ No Gemini API key - using fallback explanations")
            except Exception as e:
                self.use_gemini = False
                print(f"⚠️ Gemini init failed: {e}")
        
        self._log("init", {
            "gemini_enabled": self.use_gemini,
            "supervisor_rules_loaded": "supervisor_rules" in CONFIG,
            "check_frequency_days": self.check_frequency
        })

    def _log(self, event: str, details: Dict[str, Any]):
        """Simple logging wrapper."""
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
    # 2. OPERATOR MONITORING (The Intelligent Watchdog)
    # ===================================================================
    
    def check_operator_performance(
        self, 
        operator_traces: List[Dict[str, Any]],
        pair: Tuple[str, str],
        phase: str = "holdout"
    ) -> Dict[str, Any]:
        """
        Monitors trading performance with Z-Score Circuit Breakers and a 
        Three-Strike Warning system.
        """
        
        # 1. Base Setup
        if "supervisor_rules" not in CONFIG:
            return self._basic_check(operator_traces, pair)
        
        rules = CONFIG["supervisor_rules"].get(phase, {})
        
        if len(operator_traces) < rules.get("min_observations", 20):
            return {"action": "continue", "severity": "info", "reason": "insufficient_data", "metrics": {}}

        pair_key = f"{pair[0]}-{pair[1]}"
        latest_trace = operator_traces[-1]
        days_in_pos = latest_trace.get('days_in_position', 0)
        
        # 2. Initialize/Reset State (Grace Period Logic)
        if pair_key not in self.monitoring_state:
            self.monitoring_state[pair_key] = {'strikes': 0, 'grace_period': True}

        # If days_in_position is small (<= 5), reset strikes ("Burn-in period").
        if days_in_pos <= 5:
            self.monitoring_state[pair_key]['strikes'] = 0
            self.monitoring_state[pair_key]['grace_period'] = True
        else:
            self.monitoring_state[pair_key]['grace_period'] = False

        # 3. Compute Metrics
        metrics = self._compute_live_metrics(operator_traces)
        
        # ============================================================
        # A. IMMEDIATE KILL (Structural Breaks) - CHECK EVERY DAY
        # ============================================================
        # We NEVER skip this. If Z > 5, the model is broken now.
        
        # We calculate spread stats independently of the operator to act as a double-check
        spread_history = [t['current_spread'] for t in operator_traces]
        if len(spread_history) > 30:
            spread_series = pd.Series(spread_history)
            rolling_mean = spread_series.rolling(window=30).mean().iloc[-1]
            rolling_std = spread_series.rolling(window=30).std().iloc[-1]
            
            if rolling_std > 1e-8:
                current_z = abs(latest_trace['current_spread'] - rolling_mean) / rolling_std
                
                if current_z > 5:
                    self._log("intervention_triggered", {"pair": pair, "reason": "structural_break_zscore", "z": current_z})
                    return {
                        'action': 'stop',
                        'severity': 'critical',
                        'reason': f'Structural Break: Z-Score {current_z:.1f} > 5 (Instant Kill)',
                        'metrics': metrics
                    }

        # ============================================================
        # FREQUENCY CHECK: Skip "Performance Reviews" on off-days
        # ============================================================
        # If it's not a check day, and we passed the safety check above, we relax.
        is_check_day = (days_in_pos > 0) and (days_in_pos % self.check_frequency == 0)
        
        if not is_check_day:
            return {
                'action': 'continue',
                'severity': 'info',
                'reason': f'Off-cycle day (Day {days_in_pos})',
                'metrics': metrics
            }

        # ============================================================
        # B. STALEMATE CHECK (Modified) - Checked on Check Days
        # ============================================================
        if days_in_pos > 45:
            unrealized_pnl = latest_trace.get('unrealized_pnl', 0.0)
            
            if unrealized_pnl > 0:
                return {
                    'action': 'continue', 
                    'severity': 'info',
                    'reason': f'Stalemate ({days_in_pos} days) but profitable. Extending hold.',
                    'metrics': metrics
                }
            else:
                 return {
                    'action': 'stop', 
                    'severity': 'warning',
                    'reason': f'Stalemate ({days_in_pos} days) and failing. Closing dead capital.',
                    'metrics': metrics
                }

        # ============================================================
        # C. SEQUENTIAL WARNINGS (P&L Checks) - Checked on Check Days
        # ============================================================
        
        # Define Thresholds
        stop_tier = rules.get("stop_tier", {})
        max_dd_limit = stop_tier.get("catastrophic_drawdown", 0.30)
        
        # Check for Violation
        violation = False
        violation_reason = ""
        
        if metrics['drawdown'] > max_dd_limit:
            violation = True
            violation_reason = f"Drawdown {metrics['drawdown']:.1%} > {max_dd_limit:.1%}"
        
        elif metrics['sharpe'] < -2.0 and days_in_pos > 20: 
            violation = True
            violation_reason = f"Sharpe {metrics['sharpe']:.2f} is disastrous"

        # Apply Three-Strike Logic
        if violation:
            if self.monitoring_state[pair_key]['grace_period']:
                return {
                    'action': 'continue',
                    'severity': 'info',
                    'reason': f'Grace Period: Ignoring {violation_reason}',
                    'metrics': metrics
                }
            
            self.monitoring_state[pair_key]['strikes'] += 1
            strikes = self.monitoring_state[pair_key]['strikes']
            
            if strikes == 1:
                return {
                    'action': 'warn',
                    'severity': 'warning',
                    'reason': f'Strike 1/3: {violation_reason}. Monitoring closely.',
                    'metrics': metrics
                }
            elif strikes == 2:
                return {
                    'action': 'adjust',
                    'severity': 'warning',
                    'reason': f'Strike 2/3: {violation_reason}. Persisting. Suggest size reduction.',
                    'suggestion': 'reduce_size_50_percent',
                    'metrics': metrics
                }
            elif strikes >= 3:
                return {
                    'action': 'stop',
                    'severity': 'critical',
                    'reason': f'Strike 3/3: {violation_reason}. Patience exhausted. Stopping.',
                    'metrics': metrics
                }
        else:
            # GOOD BEHAVIOR: Heal strikes slowly on check days
            if self.monitoring_state[pair_key]['strikes'] > 0:
                self.monitoring_state[pair_key]['strikes'] -= 1
                
        return {
            'action': 'continue',
            'severity': 'info',
            'reason': 'Performance nominal',
            'metrics': metrics
        }
        
    def _compute_live_metrics(self, traces):
        """Helper to calculate metrics efficiently from traces."""
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
        
        # Group traces by pair
        traces_by_pair = {}
        for t in operator_traces:
            traces_by_pair.setdefault(t['pair'], []).append(t)

        all_returns = []
        all_pnls = []
        pair_summaries = []

        # Process each pair
        for pair, traces in traces_by_pair.items():
            pair_returns = []
            pair_pnls = []
            
            # Sort traces by step just in case they are out of order
            traces = sorted(traces, key=lambda x: x['step'])

            for i in range(1, len(traces)):
                pnl = traces[i].get("realized_pnl_this_step", 0)
                
                pv_curr = traces[i].get("portfolio_value", 0)
                pv_prev = traces[i-1].get("portfolio_value", 0)
                
                if pv_prev > 0:
                    ret = (pv_curr - pv_prev) / pv_prev
                else:
                    ret = 0.0
                
                pair_returns.append(ret)
                all_returns.append(ret)
                
                pair_pnls.append(pnl)
                all_pnls.append(pnl)

            # Pair stats
            initial = traces[0]['portfolio_value']
            final = traces[-1]['portfolio_value']
            cum_ret = (final - initial) / initial if initial > 0 else 0
            
            pair_summaries.append({
                "pair": pair,
                "total_pnl": sum(pair_pnls),
                "cum_return": cum_ret,
                "sharpe": self._calculate_sharpe(pair_returns),
                "sortino": self._calculate_sortino(pair_returns),
                "max_drawdown": max([t.get("max_drawdown", 0) for t in traces] + [0]),
                "steps": len(traces)
            })

        # Global stats
        metrics = {
            "total_pnl": sum(all_pnls),
            "sharpe_ratio": self._calculate_sharpe(all_returns),
            "sortino_ratio": self._calculate_sortino(all_returns),
            "max_drawdown": max([p['max_drawdown'] for p in pair_summaries] + [0]),
            "avg_return": float(np.mean(all_returns)) if all_returns else 0,
            "total_steps": len(operator_traces),
            "n_pairs": len(traces_by_pair),
            "pair_summaries": pair_summaries
        }
        
        # --- CORRECT WIN RATE CALCULATION (Matching Visualizer) ---
        # Filter for steps where a trade was explicitly closed/adjusted
        closed_trades = [t for t in operator_traces if t.get("realized_pnl_this_step", 0) != 0]
        if closed_trades:
            # A win is defined as Positive PnL AFTER transaction costs
            wins = sum(1 for t in closed_trades if (t.get("realized_pnl_this_step", 0) - t.get("transaction_costs", 0)) > 0)
            metrics["win_rate"] = wins / len(closed_trades)
        else:
            metrics["win_rate"] = 0.0
        
        # Calculate Risk Metrics (VaR/CVaR)
        if all_returns:
            metrics["var_95"] = float(np.percentile(all_returns, 5))
            tail_losses = [r for r in all_returns if r <= metrics["var_95"]]
            metrics["cvar_95"] = float(np.mean(tail_losses)) if tail_losses else metrics["var_95"]
        else:
            metrics["var_95"] = 0.0
            metrics["cvar_95"] = 0.0

        # Additional activity metrics
        metrics["positive_returns"] = sum(1 for r in all_returns if r > 0)
        metrics["negative_returns"] = sum(1 for r in all_returns if r < 0)
        metrics["median_return"] = float(np.median(all_returns)) if all_returns else 0.0
        metrics["std_return"] = float(np.std(all_returns)) if all_returns else 0.0
        metrics["avg_steps_per_pair"] = metrics["total_steps"] / max(metrics["n_pairs"], 1)
        
        # Store cumulative return properly
        if operator_traces:
            sorted_traces = sorted(operator_traces, key=lambda x: x['step'])
            start_pv = sorted_traces[0].get("portfolio_value", 0)
            end_pv = sorted_traces[-1].get("portfolio_value", 0)
            metrics["cum_return"] = (end_pv - start_pv) / start_pv if start_pv > 0 else 0
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
