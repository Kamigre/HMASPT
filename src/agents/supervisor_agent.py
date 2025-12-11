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
        if len(spread_history) > 10:
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
        elif metrics['sharpe'] < 0 and days_in_pos > 15: 
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
        """
        Evaluate complete portfolio performance using robust 'Delta PnL' aggregation.
        This PREVENTS 'Capital Injection' bugs where new pairs appearing look like profit.
        """
        
        # --- 1. DATA PREPARATION ---
        df_all = pd.DataFrame(operator_traces)
        
        # Handle empty data
        if df_all.empty:
            return {
                "metrics": {
                    "total_pnl": 0.0, "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
                    "max_drawdown": 0.0, "win_rate": 0.0, "cum_return": 0.0,
                    "equity_curve": [], "dates": [], "pair_summaries": []
                },
                "actions": [],
                "explanation": "No data available."
            }
        
        # Normalize time column
        time_col = 'timestamp' if 'timestamp' in df_all.columns else 'step'
        if time_col not in df_all.columns and 'local_step' in df_all.columns:
            time_col = 'local_step'
        
        # Ensure time is sorted
        df_all[time_col] = pd.to_datetime(df_all[time_col]) if time_col == 'timestamp' else df_all[time_col]
        df_all = df_all.sort_values(by=time_col)
        
        # --- 2. ROBUST EQUITY CURVE CALCULATION (The Source of Truth) ---
        # Pivot: Index=Time, Columns=Pair, Values=Portfolio Value
        equity_matrix = df_all.pivot_table(index=time_col, columns='pair', values='portfolio_value')
        
        # Forward fill: If a pair creates no new trace, it holds its last value.
        equity_matrix_ffill = equity_matrix.ffill()
        
        # Calculate Dollar PnL per step (Change in value)
        # diff() ensures that the first appearance of a pair (NaN -> Value) results in NaN change, not Profit.
        dollar_pnl_matrix = equity_matrix_ffill.diff()
        
        # Sum dollar PnL across all pairs for each step
        global_dollar_pnl = dollar_pnl_matrix.sum(axis=1).fillna(0.0)
        
        # Calculate Total Invested Capital per step (Sum of active pairs)
        # We fill NaN with 0 here just for the summation of capital
        total_capital_series = equity_matrix_ffill.fillna(0.0).sum(axis=1)
        
        # Calculate Percentage Returns
        # Return = Global Dollar PnL / Previous Step's Total Capital
        # We shift capital by 1 to represent "Assets at beginning of period"
        prev_capital = total_capital_series.shift(1)
        
        # Avoid division by zero and handle infinity
        global_returns_series = global_dollar_pnl / prev_capital
        global_returns_series = global_returns_series.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        # Clean returns list for metrics
        global_returns = global_returns_series.tolist()
        
        # Reconstruct the "Normalized" Equity Curve (Start at 100)
        cum_returns = (1 + global_returns_series).cumprod()
        normalized_equity_curve = 100 * cum_returns
        
        # --- 3. PROCESS PAIR SUMMARIES ---
        pair_summaries = []
        total_portfolio_realized_pnl = 0.0
        pairs = equity_matrix.columns.tolist()

        for pair in pairs:
            pair_df = df_all[df_all['pair'] == pair]
            if pair_df.empty: continue

            # Returns for this specific pair
            pair_vals = pair_df['portfolio_value'].sort_index()
            pair_ret = pair_vals.pct_change().dropna().tolist()

            # PnL & Metrics
            final_trace = pair_df.iloc[-1]
            pair_total_pnl = final_trace.get("realized_pnl", 0.0)
            total_portfolio_realized_pnl += pair_total_pnl
            
            # Pair-specific Drawdown
            p_vals = pair_df['portfolio_value']
            p_max = p_vals.cummax()
            p_dd = (p_vals - p_max) / p_max
            pair_max_dd = abs(p_dd.min()) if not p_dd.empty else 0.0

            initial_val = pair_df.iloc[0]['portfolio_value']
            final_val = pair_df.iloc[-1]['portfolio_value']
            c_ret = (final_val - initial_val) / initial_val if initial_val > 0 else 0.0

            pair_summaries.append({
                "pair": pair,
                "total_pnl": pair_total_pnl,
                "cum_return": c_ret,
                "sharpe": self._calculate_sharpe(pair_ret),
                "sortino": self._calculate_sortino(pair_ret),
                "max_drawdown": pair_max_dd,
                "steps": len(pair_df)
            })

        # --- 4. CALCULATE GLOBAL METRICS ---
        
        # Max Drawdown (Based on the normalized curve)
        running_max = normalized_equity_curve.cummax()
        dd_series = (normalized_equity_curve - running_max) / running_max
        portfolio_max_dd = abs(dd_series.min()) if not dd_series.empty else 0.0
        
        # Basic Metrics
        metrics = {
            "total_pnl": total_portfolio_realized_pnl,
            "sharpe_ratio": self._calculate_sharpe(global_returns),
            "sortino_ratio": self._calculate_sortino(global_returns),
            "max_drawdown": portfolio_max_dd, 
            "avg_return": float(np.mean(global_returns)) if global_returns else 0.0,
            "total_steps": len(df_all),
            "n_pairs": len(pairs),
            "pair_summaries": pair_summaries
        }
        
        # --- 5. EXPORT CURVE DATA FOR VISUALIZATION ---
        # We embed the time series data directly so the visualizer can opt to use it directly
        metrics["equity_curve"] = normalized_equity_curve.tolist()
        # Convert timestamps to strings for JSON serializability if needed, or keep as Index
        metrics["equity_curve_dates"] = normalized_equity_curve.index.tolist()

        # --- 6. WIN RATE & RISK ---
        if 'realized_pnl_this_step' in df_all.columns:
            closed_trades = df_all[df_all['realized_pnl_this_step'] != 0]
            if not closed_trades.empty:
                costs = closed_trades.get('transaction_costs', 0.0)
                # Count win if Net PnL > 0
                wins = ((closed_trades['realized_pnl_this_step'] - costs) > 0).sum()
                metrics["win_rate"] = wins / len(closed_trades)
            else:
                metrics["win_rate"] = 0.0
        else:
            metrics["win_rate"] = 0.0

        if global_returns:
            metrics["var_95"] = float(np.percentile(global_returns, 5))
            tail_losses = [r for r in global_returns if r <= metrics["var_95"]]
            metrics["cvar_95"] = float(np.mean(tail_losses)) if tail_losses else metrics["var_95"]
        else:
            metrics["var_95"] = 0.0; metrics["cvar_95"] = 0.0

        # Final Cumulative Return
        metrics["cum_return"] = (normalized_equity_curve.iloc[-1] - 100) / 100 if not normalized_equity_curve.empty else 0.0

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
                ### ROLE & OBJECTIVE
                Act as the Chief Risk Officer (CRO) of a Quantitative Hedge Fund. Your sole mandate is capital preservation and risk-adjusted growth. You are addressing the Investment Committee.
                
                ### INPUT DATA
                --- PORTFOLIO METRICS ---
                {json.dumps(metrics, indent=2, default=str)}
                
                --- AUTOMATED SYSTEM ACTIONS ---
                {json.dumps(actions, indent=2)}
                
                ### INSTRUCTIONS
                Produce a high-level, institutional-grade **Risk Memo**.
                * **Tone:** Clinical, academic, and extremely concise. No pleasantries, no "I hope this helps," and no email formatting (Subject/Dear Team).
                * **Format:** Bullet points and bold key figures only.
                * **Length:** Maximum 400 words.
                
                ### REQUIRED SECTIONS
                
                #### 1. Performance Attribution
                * **Return Efficiency:** Analyze Sharpe and Sortino ratios. Is the risk-adjusted return justifiable?
                * **Profit Quality:** Contrast Win Rate against Total PnL. explicitly identify if the strategy suffers from "negative skew" (small frequent wins, rare massive losses).
                * **Distribution:** Evaluate the delta between Average Return and Median Return to determine return skewness.
                
                #### 2. Risk Decomposition
                * **Tail Risk Analysis:** Contrast Max Drawdown against VaR/CVaR (95%). Is the realized drawdown within modeled expectations?
                * **Duration/Stalemate Risk:** Analyze 'avg_steps_per_pair'. Is the holding period consistent with a mean-reversion thesis, or are capital costs eroding alpha?
                * **Concentration:** Identify if losses are systemic or isolated to specific pairs.
                
                #### 3. CRO Verdict & Adjustments
                * **Action Review:** Validate the AUTOMATED ACTIONS listed in the input. State "ENDORSE" or "OVERRIDE" with a single-sentence justification.
                * **Traffic Light Signal:** Conclude with a single word: GREEN (Scale Up), YELLOW (Maintain/Monitor), or RED (De-risk/Halt).
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
