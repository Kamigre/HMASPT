import os
import json
import numpy as np
import pandas as pd
import google.generativeai as genai
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
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
    # 2. OPERATOR MONITORING (Live / Step-by-Step)
    # ===================================================================
    
    def check_operator_performance(
        self, 
        operator_traces: List[Dict[str, Any]],
        pair: Tuple[str, str],
        phase: str = "holdout"
    ) -> Dict[str, Any]:
        
        if not operator_traces:
             return {"action": "continue", "severity": "info", "reason": "no_data"}

        # Extract latest trace data
        latest = operator_traces[-1]
        
        # Handle cases where trace is flat or nested in 'details'
        if 'details' in latest:
            latest = latest['details']

        # Get relevant metrics from the LATEST step
        days_in_pos = latest.get('days_in_position', 0)
        current_drawdown = latest.get('max_drawdown', 0.0) # Usually calculated by Env
        current_spread = latest.get('current_spread', 0.0)
        unrealized_pnl = latest.get('unrealized_pnl', 0.0)
        sharpe = latest.get('sharpe', 0.0) # Might not be in step trace, usually in summary

        pair_key = f"{pair[0]}-{pair[1]}"
        
        # Initialize monitoring state
        if pair_key not in self.monitoring_state:
            self.monitoring_state[pair_key] = {'strikes': 0, 'grace_period': True}

        # 3-day burn-in grace period
        if days_in_pos <= 3:
            self.monitoring_state[pair_key]['strikes'] = 0
            self.monitoring_state[pair_key]['grace_period'] = True
        else:
            self.monitoring_state[pair_key]['grace_period'] = False
        
        # --- A. IMMEDIATE HARD STOPS ---
        
        # 1. Hard Drawdown Kill (> 15%)
        if current_drawdown > 0.15:
             return {
                'action': 'stop',
                'severity': 'critical',
                'reason': f'Hard Stop: Drawdown {current_drawdown:.1%} > 15%',
                'metrics': {'drawdown': current_drawdown}
            }

        # --- B. PERIODIC REVIEW (Strikes System) ---
        is_check_day = (days_in_pos > 0) and (days_in_pos % self.check_frequency == 0)
        
        if not is_check_day:
            return {'action': 'continue', 'severity': 'info', 'reason': 'off_cycle'}

        # Stalemate Check (30 days)
        if days_in_pos > 30 and unrealized_pnl <= 0:
             return {
                'action': 'stop', 
                'severity': 'warning',
                'reason': f'Stalemate ({days_in_pos} days) & Negative PnL. Capital rotation.',
                'metrics': {'unrealized_pnl': unrealized_pnl}
            }

        # VIOLATION LOGIC
        violation = False
        violation_reason = ""
        
        # Warning Threshold: 10% Drawdown
        if current_drawdown > 0.10: 
            violation = True
            violation_reason = f"Drawdown {current_drawdown:.1%} > 10%"
        
        # TWO-STRIKE SYSTEM
        if violation:
            if self.monitoring_state[pair_key]['grace_period']:
                return {'action': 'continue', 'severity': 'info', 'reason': 'Grace Period'}
            
            self.monitoring_state[pair_key]['strikes'] += 1
            strikes = self.monitoring_state[pair_key]['strikes']
            
            if strikes == 1:
                return {
                    'action': 'warn',
                    'severity': 'warning',
                    'reason': f'Strike 1/2: {violation_reason}. Monitoring closely.'
                }
            elif strikes >= 2:
                return {
                    'action': 'stop',
                    'severity': 'critical',
                    'reason': f'Strike 2/2: {violation_reason}. Failed.'
                }
        else:
            # Heal strikes if performance recovers
            if self.monitoring_state[pair_key]['strikes'] > 0 and current_drawdown < 0.05:
                self.monitoring_state[pair_key]['strikes'] -= 1
                
        return {
            'action': 'continue',
            'severity': 'info',
            'reason': 'Performance nominal'
        }

    # ===================================================================
    # 3. FINAL EVALUATION (Post-Trading Aggregation)
    # ===================================================================
    
    def evaluate_portfolio(self, operator_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate portfolio by parsing the Operator's 'episode_complete' events.
        This ensures we use the exact figures the Operator calculated.
        """
        
        pair_summaries = []
        global_pnl = 0.0
        total_steps = 0
        
        # 1. Identify "Summary" traces vs "Step" traces
        # We look for events named 'episode_complete' or 'holdout_complete'
        summary_traces = [
            t for t in operator_traces 
            if t.get('event') in ['episode_complete', 'holdout_complete', 'pair_trained']
            and 'details' in t
        ]

        # Process Summary Traces
        processed_pairs = set()
        
        for trace in summary_traces:
            details = trace['details']
            
            # Handle pair name (list or string)
            raw_pair = details.get('pair')
            if isinstance(raw_pair, list):
                pair_str = f"{raw_pair[0]}-{raw_pair[1]}"
            else:
                pair_str = str(raw_pair)
            
            # Prevent duplicates if multiple events exist for same pair
            if pair_str in processed_pairs: continue
            processed_pairs.add(pair_str)

            # Extract Pre-calculated Metrics
            pnl = float(details.get('total_pnl', details.get('realized_pnl', 0.0)))
            cum_ret = float(details.get('final_cum_return', details.get('cum_return', 0.0)))
            # If Operator logged return as decimal (e.g. 0.05), convert to % for display consistency if needed
            # But here we keep it as raw float for aggregation
            
            sharpe = float(details.get('sharpe', 0.0))
            sortino = float(details.get('sortino', 0.0))
            max_dd = float(details.get('max_drawdown', details.get('drawdown', 0.0)))
            steps = int(details.get('total_steps', details.get('steps', 0)))
            
            # Add to global stats
            global_pnl += pnl
            total_steps += steps
            
            pair_summaries.append({
                "pair": pair_str,
                "total_pnl": pnl,
                "cum_return": cum_ret,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown": max_dd,
                "steps": steps,
                "status": "STOPPED" if details.get('was_stopped') else "COMPLETE"
            })

        # Aggregation Logic
        n_pairs = len(pair_summaries)
        if n_pairs > 0:
            avg_return = np.mean([p['cum_return'] for p in pair_summaries])
            avg_sharpe = np.mean([p['sharpe'] for p in pair_summaries])
            avg_sortino = np.mean([p['sortino'] for p in pair_summaries])
            portfolio_max_dd = max([p['max_drawdown'] for p in pair_summaries])
            
            # Calculate Win Rate based on PnL
            winning_pairs = sum(1 for p in pair_summaries if p['total_pnl'] > 0)
            win_rate = winning_pairs / n_pairs
        else:
            avg_return = 0.0
            avg_sharpe = 0.0
            avg_sortino = 0.0
            portfolio_max_dd = 0.0
            win_rate = 0.0

        metrics = {
            "total_pnl": global_pnl,
            "avg_return": avg_return,
            "sharpe_ratio": avg_sharpe,
            "sortino_ratio": avg_sortino,
            "max_drawdown": portfolio_max_dd,
            "win_rate": win_rate,
            "total_steps": total_steps,
            "n_pairs": n_pairs,
            "pair_summaries": pair_summaries
        }
        
        actions = self._generate_portfolio_actions(metrics)
        explanation = self._generate_explanation(metrics, actions)
        
        return {"metrics": metrics, "actions": actions, "explanation": explanation}

    def _generate_portfolio_actions(self, metrics: Dict) -> List[Dict]:
        actions = []
        if metrics['max_drawdown'] > 0.15:
            actions.append({"action": "reduce_risk", "reason": "Portfolio Max Drawdown > 15%", "severity": "high"})
        if metrics['sharpe_ratio'] < 1.0:
             actions.append({"action": "review_strategy", "reason": "Avg Sharpe Ratio < 1.0", "severity": "medium"})
        return actions
        
    def _generate_explanation(self, metrics: Dict, actions: List[Dict]) -> str:
        if not self.use_gemini:
            return f"Portfolio PnL: ${metrics['total_pnl']:.2f}. Sharpe: {metrics['sharpe_ratio']:.2f}. Win Rate: {metrics['win_rate']:.1%}"
        
        prompt = f"""
                You are the Chief Risk Officer (CRO) at a Quantitative Hedge Fund. 
                
                Analyze the following Pairs Trading Portfolio results based on the Operator's execution logs.
                
                --- PORTFOLIO METRICS ---
                {json.dumps(metrics, indent=2, default=str)}
                
                --- AUTOMATED ACTIONS ---
                {json.dumps(actions, indent=2)}
                
                Write a concise Executive Risk Memo (max 200 words).
                
                Structure:
                1. **Performance Verdict**: Assess the Average Sharpe ({metrics['sharpe_ratio']:.2f}) and Total PnL. Is the strategy viable?
                2. **Risk Assessment**: Comment on the Max Drawdown ({metrics['max_drawdown']:.1%}) and specific pairs that failed (check "STOPPED" status in summaries).
                3. **Strategic Directive**: Give a final recommendation (Scale Up, Maintain, or Halt).
                """
        
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
