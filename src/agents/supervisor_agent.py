import os
import json
import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import google.generativeai as genai
from statsmodels.tsa.stattools import adfuller

from config import CONFIG
from agents.message_bus import JSONLogger
from utils import half_life as compute_half_life, compute_spread


@dataclass
class SupervisorAgent:
    
    logger: JSONLogger = None
    df: pd.DataFrame = None  # Full price data for validation
    max_drawdown: float = 0.20  # Max allowed drawdown before intervention
    min_sharpe: float = -0.5  # Min Sharpe before intervention
    min_win_rate: float = 0.5  # Min win rate before intervention
    storage_dir: str = "./storage"
    gemini_api_key: Optional[str] = None
    model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    use_gemini: bool = True

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
            "max_drawdown": self.max_drawdown,
            "min_sharpe": self.min_sharpe,
            "gemini_enabled": self.use_gemini
        })

    def _log(self, event: str, details: Dict[str, Any]):
        """Simple logging wrapper."""
        if self.logger:
            self.logger.log("supervisor", event, details)

    def format_skip_info(self, pair: Tuple[str, str], decision: Dict, step: int) -> Dict:
      """Format skip information for visualization."""
      return {
          "pair": f"{pair[0]}-{pair[1]}",
          "reason": decision['reason'],
          "step_stopped": step,
          "metrics": decision['metrics']
      }

    # ===================================================================
    # 1. PAIR VALIDATION (used by Selector)
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

        # Pivot price data
        prices = self.df.pivot(
            index="date",
            columns="ticker",
            values="adj_close"
        ).sort_index()

        print(f"\n🔍 Validating {len(df_pairs)} pairs...")
        
        for idx, row in df_pairs.iterrows():
            x, y = row["x"], row["y"]

            # Check tickers exist
            if x not in prices.columns or y not in prices.columns:
                continue

            # Extract validation window
            series_x = prices[x].loc[start:end].dropna()
            series_y = prices[y].loc[start:end].dropna()

            if min(len(series_x), len(series_y)) < 60:
                continue

            # Compute spread
            spread = compute_spread(series_x, series_y)
            if spread is None or len(spread) == 0:
                continue

            # --- Crossing frequency ---
            centered = spread - spread.mean()
            crossings = (centered.shift(1) * centered < 0).sum()
            days = (series_x.index[-1] - series_x.index[0]).days
            crossings_per_year = float(crossings) / max(days / 365.0, 1e-9)

            if crossings_per_year < min_crossings_per_year:
                continue

            # --- ADF test (stationarity) ---
            try:
                adf_res = adfuller(spread.dropna())
                adf_p = adf_res[1]
            except:
                adf_p = 1.0

            # --- Half-life (mean reversion speed) ---
            hl = compute_half_life(spread.values)
            try:
                hl_val = float(hl)
            except:
                hl_val = float("inf")

            # --- Pass criteria ---
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
        pair: Tuple[str, str]
    ) -> Dict[str, Any]:

        if len(operator_traces) < 10:
            return {"action": "continue", "reason": "insufficient_data"}
        
        # Calculate current metrics
        returns = [t.get("return", 0) for t in operator_traces]
        pnls = [t.get("pnl", 0) for t in operator_traces]
        
        # Current drawdown
        cum_returns = np.cumsum(returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (peak - cum_returns) / (np.abs(peak) + 1e-8)
        max_dd = float(np.max(drawdown))
        
        # Current Sharpe
        sharpe = self._calculate_sharpe(returns)
        
        # Win rate
        positive_rets = sum(1 for r in returns if r > 0)
        win_rate = positive_rets / len(returns)
        
        # Total P&L
        total_pnl = sum(pnls)
        
        # --- Decision Logic ---
        
        # # STOP: Excessive drawdown
        # if max_dd > self.max_drawdown:
        #     self._log("intervention_triggered", {
        #         "pair": f"{pair[0]}-{pair[1]}",
        #         "action": "stop",
        #         "reason": "max_drawdown_exceeded",
        #         "drawdown": max_dd,
        #         "threshold": self.max_drawdown
        #     })
        #     return {
        #         "action": "stop",
        #         "reason": f"Drawdown {max_dd:.2%} exceeds limit {self.max_drawdown:.2%}",
        #         "metrics": {"drawdown": max_dd, "sharpe": sharpe, "win_rate": win_rate}
        #     }
        
        # # STOP: Terrible Sharpe
        # if sharpe < self.min_sharpe:
        #     self._log("intervention_triggered", {
        #         "pair": f"{pair[0]}-{pair[1]}",
        #         "action": "stop",
        #         "reason": "sharpe_too_low",
        #         "sharpe": sharpe,
        #         "threshold": self.min_sharpe
        #     })
        #     return {
        #         "action": "stop",
        #         "reason": f"Sharpe {sharpe:.2f} below minimum {self.min_sharpe:.2f}",
        #         "metrics": {"drawdown": max_dd, "sharpe": sharpe, "win_rate": win_rate}
        #     }
        
        # # ADJUST: Poor win rate
        # if win_rate < self.min_win_rate:
        #     self._log("intervention_triggered", {
        #         "pair": f"{pair[0]}-{pair[1]}",
        #         "action": "adjust",
        #         "reason": "low_win_rate",
        #         "win_rate": win_rate,
        #         "threshold": self.min_win_rate
        #     })
        #     return {
        #         "action": "adjust",
        #         "reason": f"Win rate {win_rate:.2%} below target {self.min_win_rate:.2%}",
        #         "suggestion": "Consider reducing position sizes or tightening entry thresholds",
        #         "metrics": {"drawdown": max_dd, "sharpe": sharpe, "win_rate": win_rate}
        #     }
        
        # Continue trading
        return {
            "action": "continue",
            "reason": "performance_acceptable",
            "metrics": {"drawdown": max_dd, "sharpe": sharpe, "win_rate": win_rate}
        }

    # ===================================================================
    # 3. FINAL EVALUATION (after all trading complete)
    # ===================================================================
        
    def evaluate_portfolio(
        self, 
        operator_traces: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        # Group by pair
        traces_by_pair = {}
        for t in operator_traces:
            pair = t.get("pair", "unknown")
            if pair not in traces_by_pair:
                traces_by_pair[pair] = []
            traces_by_pair[pair].append(t)

        # ============================================================
        # FIX: Recalculate true daily returns using pnl / prev_value
        # ============================================================
        all_returns = []
        all_pnls = []

        for i in range(1, len(operator_traces)):
            pnl = operator_traces[i].get("pnl", 0)
            pv_prev = operator_traces[i-1].get("portfolio_value", None)

            if pv_prev is None or pv_prev == 0:
                continue
            
            true_return = pnl / pv_prev
            all_returns.append(true_return)
            all_pnls.append(pnl)

        # If only one step, fallback safely
        if not all_returns:
            all_returns = [0.0]
            all_pnls = [0.0]

        total_pnl = sum(all_pnls)

        sharpe = self._calculate_sharpe(all_returns)
        sortino = self._calculate_sortino(all_returns)

        # Risk metrics
        max_dd = max([t.get("max_drawdown", 0) for t in operator_traces] + [0])
        var_95 = float(np.percentile(all_returns, 5)) if all_returns else 0
        cvar_95 = float(np.mean([r for r in all_returns if r <= var_95])) if any(r <= var_95 for r in all_returns) else var_95

        # Win rate
        positive = sum(1 for r in all_returns if r > 0)
        win_rate = positive / len(all_returns) if all_returns else 0

        # ============================================================
        # Per-pair summaries (with corrected returns)
        # ============================================================
        pair_summaries = []
        for pair, traces in traces_by_pair.items():
            
            pair_pnls = []
            pair_returns = []

            for i in range(1, len(traces)):
                pnl = traces[i].get("pnl", 0)
                pv_prev = traces[i-1].get("portfolio_value", None)

                if pv_prev is None or pv_prev == 0:
                    continue

                true_return = pnl / pv_prev
                pair_pnls.append(pnl)
                pair_returns.append(true_return)

            if not pair_returns:
                pair_returns = [0.0]
                pair_pnls = [0.0]

            pair_sharpe = self._calculate_sharpe(pair_returns)
            pair_sortino = self._calculate_sortino(pair_returns)
            pair_max_dd = max([t.get("max_drawdown", 0) for t in traces] + [0])

            pair_summaries.append({
                "pair": pair,
                "total_pnl": float(sum(pair_pnls)),
                "final_return": float(traces[-1].get("cum_reward", 0)) if traces else 0,
                "sharpe": float(pair_sharpe),
                "sortino": float(pair_sortino),
                "max_drawdown": float(pair_max_dd),
                "steps": len(traces)
            })

        metrics = {
            "total_pnl": float(total_pnl),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "win_rate": float(win_rate),
            "avg_return": float(np.mean(all_returns)) if all_returns else 0,
            "std_return": float(np.std(all_returns)) if all_returns else 0,
            "n_pairs": len(traces_by_pair),
            "total_steps": len(operator_traces),
            "pair_summaries": pair_summaries
        }

        # Additional metrics
        metrics["positive_returns"] = sum(1 for r in all_returns if r > 0)
        metrics["negative_returns"] = sum(1 for r in all_returns if r < 0)
        metrics["median_return"] = float(np.median(all_returns)) if all_returns else 0.0
        metrics["avg_steps_per_pair"] = (
            metrics["total_steps"] / max(metrics["n_pairs"], 1)
        )

        # Safety defaults
        metrics.setdefault("pair_summaries", [])
        metrics.setdefault("max_drawdown", 0.0)
        metrics.setdefault("var_95", 0.0)
        metrics.setdefault("cvar_95", 0.0)
        metrics.setdefault("sharpe_ratio", 0.0)
        metrics.setdefault("sortino_ratio", 0.0)
        metrics.setdefault("win_rate", 0.0)
        metrics.setdefault("avg_return", 0.0)
        metrics.setdefault("std_return", 0.0)

        # =====================================================
        # ADD ALL MISSING METRICS FOR YOUR PRINT SUMMARY
        # =====================================================

        # Trading activity counts
        metrics["positive_returns"] = sum(1 for r in all_returns if r > 0)
        metrics["negative_returns"] = sum(1 for r in all_returns if r < 0)

        # Median return
        metrics["median_return"] = float(np.median(all_returns)) if all_returns else 0.0

        # Avg number of steps per pair
        metrics["avg_steps_per_pair"] = (
            metrics["total_steps"] / max(metrics["n_pairs"], 1)
        )

        # Ensure safety defaults
        metrics.setdefault("pair_summaries", [])
        metrics.setdefault("max_drawdown", 0.0)
        metrics.setdefault("var_95", 0.0)
        metrics.setdefault("cvar_95", 0.0)
        metrics.setdefault("sharpe_ratio", 0.0)
        metrics.setdefault("sortino_ratio", 0.0)
        metrics.setdefault("win_rate", 0.0)
        metrics.setdefault("avg_return", 0.0)
        metrics.setdefault("std_return", 0.0)

        # =====================================================
        # END OF PATCH
        # =====================================================

        # Generate actions based on final performance
        actions = []
        
        if max_dd > self.max_drawdown:
            actions.append({
                "action": "reduce_risk",
                "reason": "Portfolio drawdown exceeded limit",
                "severity": "high"
            })
        
        if sharpe < 0:
            actions.append({
                "action": "review_strategy",
                "reason": "Negative Sharpe ratio indicates consistent losses",
                "severity": "high"
            })
        
        if win_rate < self.min_win_rate:
            actions.append({
                "action": "improve_entry_exit",
                "reason": f"Win rate {win_rate:.2%} below target {self.min_win_rate:.2%}",
                "severity": "medium"
            })
        
        if cvar_95 < -0.05:
            actions.append({
                "action": "reduce_tail_risk",
                "reason": "Excessive tail losses detected",
                "severity": "high"
            })
        
        # Generate explanation
        explanation = self._generate_explanation(metrics, actions)
        
        summary = {
            "metrics": metrics,
            "actions": actions,
            "explanation": explanation
        }
        
        self._log("portfolio_evaluated", summary)
        
        # Save report
        report_path = os.path.join(self.storage_dir, "supervisor_final_report.json")
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary

    # ===================================================================
    # HELPER METHODS
    # ===================================================================
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        rf_daily = CONFIG.get("risk_free_rate", 0.04) / 252
        excess = np.array(returns) - rf_daily
        
        mean_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1)
        
        if std_excess < 1e-8:
            return 0.0
        
        return (mean_excess / std_excess) * np.sqrt(252)

    def _calculate_sortino(self, returns: List[float]) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        
        rf_daily = CONFIG.get("risk_free_rate", 0.04) / 252
        excess = np.array(returns) - rf_daily
        
        mean_excess = np.mean(excess)
        downside = excess[excess < 0]
        
        if len(downside) == 0:
            return 100.0 if mean_excess > 0 else 0.0
        
        downside_std = np.sqrt(np.mean(downside**2))
        
        if downside_std < 1e-8:
            return 100.0 if mean_excess > 0 else 0.0
        
        return (mean_excess / downside_std) * np.sqrt(252)

    def _generate_explanation(self, metrics: Dict, actions: List[Dict]) -> str:
        """Generate natural language explanation using Gemini or fallback."""
        if not self.use_gemini:
            return self._fallback_explanation(metrics, actions)
        
        system_instruction = """You are an expert quantitative supervisor analyzing pairs trading performance. 
        Provide clear, actionable insights about risk and performance."""
        
        prompt = f"""Analyze this pairs trading portfolio and explain the results:
                    
                METRICS:
                {json.dumps(metrics, indent=2)}
                
                RECOMMENDED ACTIONS:
                {json.dumps(actions, indent=2)}
                
                Provide a concise 3-4 paragraph executive summary covering:
                1. Overall performance (Sharpe, Sortino, win rate, drawdown)
                2. Key risks or concerns
                3. Rationale for recommended actions
                4. Brief outlook
                
                Keep it professional and actionable. This is a simulated backtest for research purposes."""
        
        try:
            model = genai.GenerativeModel(
                model_name=self.model,
                generation_config={"temperature": self.temperature},
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)
            
            if response.candidates and response.candidates[0].content.parts:
                return "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, "text"))
            
        except Exception as e:
            print(f"⚠️ Gemini explanation failed: {e}")
        
        return self._fallback_explanation(metrics, actions)

    def _fallback_explanation(self, metrics: Dict, actions: List[Dict]) -> str:
        """Fallback explanation when Gemini unavailable."""
        text = "Portfolio Performance Summary:\n\n"
        
        text += f"The portfolio achieved a Sharpe ratio of {metrics['sharpe_ratio']:.2f} "
        text += f"and Sortino ratio of {metrics['sortino_ratio']:.2f} with a "
        text += f"win rate of {metrics['win_rate']:.1%}. "
        text += f"Maximum drawdown was {metrics['max_drawdown']:.2%}.\n\n"
        
        if actions:
            text += f"Risk Management: {len(actions)} intervention(s) recommended - "
            text += ", ".join([a['action'] for a in actions]) + ".\n\n"
        else:
            text += "No critical risk interventions required.\n\n"
        
        text += f"The portfolio traded {metrics['n_pairs']} pairs over "
        text += f"{metrics['total_steps']} steps with an average return of "
        text += f"{metrics['avg_return']:.4f} per step."
        
        return text
