import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from typing import List, Dict, Any, Tuple
from datetime import datetime

# --- CONFIG IMPORT ---
# Ensure 'config.py' is reachable. Adjust path if necessary.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from config import CONFIG
except ImportError:
    # Fallback default if config is missing
    CONFIG = {"risk_free_rate": 0.04} 

# --- STYLE SETTINGS ---
sns.set_theme(style="darkgrid", context="talk", font_scale=0.8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

class PortfolioVisualizer:
    """
    Creates institutional-grade visual reports for pairs trading performance.
    Correctly handles capital-weighted portfolios with varying allocation sizes.
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "pairs"), exist_ok=True)
        
        # Professional Color Palette
        self.colors = {
            'price': '#2c3e50',
            'profit': '#2ecc71',
            'loss': '#e74c3c',
            'drawdown': '#c0392b',
            'zscore': '#8e44ad',
            'marker_buy': '#27ae60',
            'marker_sell': '#c0392b',
            'marker_forced': '#f39c12'
        }
    
    def visualize_pair(self, traces: List[Dict], pair_name: str, was_skipped: bool = False, skip_info: Dict = None):
        """
        Create detailed visualization for a single pair including Z-Score and Drawdown analysis.
        """
        if len(traces) == 0:
            return
        
        # --- Data Extraction ---
        df = pd.DataFrame(traces)
        steps = df['local_step']
        
        # Calculate derived metrics
        df['cum_return_pct'] = df['cum_return'] * 100
        df['drawdown_pct'] = df['max_drawdown'] * 100
        
        # Identify trades
        trades = df[df['trade_occurred'] == True]
        forced_exit = df[df['forced_close'] == True]

        # Statistics
        total_pnl = df['realized_pnl'].iloc[-1]
        final_ret = df['cum_return'].iloc[-1]
        # Win rate based on non-zero daily returns
        active_days = df[df['daily_return'] != 0]
        win_rate = (active_days['daily_return'] > 0).mean() if not active_days.empty else 0.0
        
        # --- Plotting ---
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # Header Title
        title_color = self.colors['loss'] if was_skipped else 'black'
        status_text = f" | STOPPED: {skip_info['reason']}" if was_skipped and skip_info else ""
        fig.suptitle(f"{pair_name} Performance Analysis{status_text}", 
                     fontsize=20, weight='bold', color=title_color, y=0.95)

        # 1. Equity Curve (Main Plot)
        ax1 = fig.add_subplot(gs[0, :3])
        ax1.plot(steps, 100 * (1 + df['cum_return']), color=self.colors['price'], lw=2, label='Portfolio Value')
        ax1.axhline(100, color='gray', linestyle='--', alpha=0.5)
        
        # Color fill based on profit/loss
        ax1.fill_between(steps, 100, 100 * (1 + df['cum_return']), 
                         where=(df['cum_return'] >= 0), color=self.colors['profit'], alpha=0.1)
        ax1.fill_between(steps, 100, 100 * (1 + df['cum_return']), 
                         where=(df['cum_return'] < 0), color=self.colors['loss'], alpha=0.1)

        # Markers for forced exit
        if not forced_exit.empty:
            ax1.scatter(forced_exit['local_step'], 100 * (1 + forced_exit['cum_return']), 
                        color=self.colors['marker_forced'], s=150, marker='X', label='Forced Exit', zorder=5)

        ax1.set_ylabel('Portfolio Value (Base 100)')
        ax1.set_title('Equity Curve', fontsize=14, weight='bold')
        ax1.legend(loc='upper left')

        # 2. Key Metrics Scorecard (Right Side)
        ax2 = fig.add_subplot(gs[0, 3])
        ax2.axis('off')
        
        metrics_text = [
            ("Total P&L", f"${total_pnl:,.2f}", self.colors['profit'] if total_pnl > 0 else self.colors['loss']),
            ("Return", f"{final_ret*100:.2f}%", self.colors['profit'] if final_ret > 0 else self.colors['loss']),
            ("Max DD", f"{df['max_drawdown'].max()*100:.2f}%", self.colors['drawdown']),
            ("Win Rate", f"{win_rate*100:.1f}%", 'black'),
            ("Sharpe", f"{self._calculate_sharpe(df['daily_return']):.2f}", 'black'),
            ("Trades", f"{len(trades)}", 'black')
        ]
        
        y_pos = 0.9
        ax2.text(0.5, 1.0, "Performance Summary", ha='center', fontsize=16, weight='bold')
        
        for label, value, color in metrics_text:
            ax2.text(0.1, y_pos, label, ha='left', fontsize=14, color='gray')
            ax2.text(0.9, y_pos, value, ha='right', fontsize=14, weight='bold', color=color)
            ax2.plot([0.1, 0.9], [y_pos-0.05, y_pos-0.05], color='lightgray', lw=1)
            y_pos -= 0.15

        # 3. Z-Score & Position Overlay (The "Strategy View")
        ax3 = fig.add_subplot(gs[1, :])
        
        # Calculate Z-Score on the fly if not logged explicitly
        # Assuming we need to visualize the signal that drove the trades
        if 'current_spread' in df.columns:
            spread = df['current_spread']
            # Simple rolling Z-score for visualization context
            zscore = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()
            
            ax3.plot(steps, zscore, color=self.colors['zscore'], alpha=0.6, lw=1.5, label='Spread Z-Score')
            ax3.axhline(2.0, color='gray', linestyle='--', alpha=0.5)
            ax3.axhline(-2.0, color='gray', linestyle='--', alpha=0.5)
            ax3.axhline(0, color='black', lw=1)
            ax3.set_ylabel('Z-Score')
        
        # Plot Positions on secondary axis
        ax3b = ax3.twinx()
        ax3b.fill_between(steps, df['position'], color='gray', alpha=0.2, step='post', label='Position Size')
        ax3b.set_ylabel('Position Size')
        ax3.set_title('Strategy Signal vs Position', fontsize=14, weight='bold')
        
        # 4. Underwater Plot (Drawdown)
        ax4 = fig.add_subplot(gs[2, :2])
        ax4.fill_between(steps, 0, -df['drawdown_pct'], color=self.colors['drawdown'], alpha=0.4)
        ax4.plot(steps, -df['drawdown_pct'], color=self.colors['drawdown'], lw=1)
        ax4.set_title('Underwater Plot (Drawdown %)', fontsize=14, weight='bold')
        ax4.set_ylabel('Drawdown %')

        # 5. Daily Returns Distribution
        ax5 = fig.add_subplot(gs[2, 2:])
        if not df[df['daily_return'] != 0].empty:
            sns.histplot(df[df['daily_return'] != 0]['daily_return'], kde=True, ax=ax5, color=self.colors['price'])
        ax5.axvline(0, color='black', linestyle='--')
        ax5.set_title('Daily Returns Distribution', fontsize=14, weight='bold')

        # 6. Cumulative P&L (Money)
        ax6 = fig.add_subplot(gs[3, :2])
        cum_pnl = df['realized_pnl']
        ax6.plot(steps, cum_pnl, color=self.colors['price'])
        ax6.fill_between(steps, 0, cum_pnl, where=(cum_pnl>=0), color=self.colors['profit'], alpha=0.3)
        ax6.fill_between(steps, 0, cum_pnl, where=(cum_pnl<0), color=self.colors['loss'], alpha=0.3)
        ax6.set_title('Cumulative Realized P&L ($)', fontsize=14, weight='bold')

        # 7. Rolling Volatility
        ax7 = fig.add_subplot(gs[3, 2:])
        rolling_vol = df['daily_return'].rolling(window=30).std() * np.sqrt(252)
        ax7.plot(steps, rolling_vol, color='purple', lw=1.5)
        ax7.set_title('Rolling 30-Day Volatility (Annualized)', fontsize=14, weight='bold')

        # Save
        filename = f"{pair_name.replace('-', '_')}_analysis.png"
        filepath = os.path.join(self.output_dir, "pairs", filename)
        plt.savefig(filepath)
        plt.close()
        
        print(f"   📊 Saved pair analysis: {filepath}")

    def visualize_portfolio(self, all_traces: List[Dict], skipped_pairs: List[Dict], final_summary: Dict):
        """
        Create aggregated portfolio dashboard.
        
        METHODOLOGY:
        1. Accumulates Dollar P&L across all pairs (Capital-Weighted).
        2. Estimates Total Capital based on max position exposure per pair.
        3. Calculates Portfolio % Return as (Total Net P&L / Total Capital).
        """
        metrics = final_summary['metrics']
        pair_summaries = metrics['pair_summaries']
        
        # --- 1. Data Processing & Alignment ---
        df_all = pd.DataFrame(all_traces)
        
        # Determine alignment index (Timestamp preferred, else Step)
        time_col = 'timestamp' if 'timestamp' in df_all.columns else 'local_step'

        # --- A. Estimate Total Allocated Capital ---
        # We need the denominator for Return %.
        # If 'position_value' exists, we sum the max exposure of every pair.
        if 'position_value' in df_all.columns:
            max_exposures = df_all.groupby('pair')['position_value'].max()
            total_capital = max_exposures.sum()
        else:
            # Fallback: Assume $10k per pair if data is missing, just to generate a curve.
            # In production, pass 'initial_capital' explicitly.
            unique_pairs = df_all['pair'].unique()
            total_capital = 10000 * len(unique_pairs)
        
        # Safety check for zero capital
        total_capital = max(total_capital, 1.0) 

        # --- B. Pivot Cumulative Realized P&L ($) ---
        # We pivot on 'realized_pnl' (money).
        pnl_matrix = df_all.pivot_table(
            index=time_col, 
            columns='pair', 
            values='realized_pnl'
        )
        
        # --- C. Handle Stopped Pairs (Forward Fill) ---
        # If Pair A stops at index 50, its P&L remains in the bank for index 51, 52...
        # We ffill to propagate the last known P&L value.
        pnl_matrix = pnl_matrix.fillna(method='ffill').fillna(0.0)
        
        # --- D. Calculate Portfolio Aggregate ---
        # Sum P&L across all pairs for every timestep
        total_portfolio_pnl_dollars = pnl_matrix.sum(axis=1)
        
        # Calculate Percentage Return based on Total Capital
        portfolio_return_pct = (total_portfolio_pnl_dollars / total_capital) * 100
        
        # Re-base to 100 for the standard Equity Curve visualization
        portfolio_equity_curve = 100 + portfolio_return_pct

        # --- Figure Setup ---
        fig = plt.figure(figsize=(22, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        final_pnl = total_portfolio_pnl_dollars.iloc[-1]
        final_pct = portfolio_return_pct.iloc[-1]
        
        fig.suptitle(f"Portfolio Final Report | Net P&L: ${final_pnl:,.2f} | Return: {final_pct:.2f}%", 
                     fontsize=22, weight='bold', y=0.96)

        # 1. Portfolio Aggregate Equity Curve (Dual Axis)
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot Base 100 Curve (Left Axis)
        ax1.plot(portfolio_equity_curve.index, portfolio_equity_curve, 
                 color='#2c3e50', lw=3, label='Portfolio Value (Base 100)')
        
        ax1.fill_between(portfolio_equity_curve.index, 100, portfolio_equity_curve, 
                         alpha=0.2, color='#3498db')
        ax1.axhline(100, linestyle='--', color='gray', alpha=0.5)
        
        # --- Right Axis for Percentage Return ---
        ax1_right = ax1.twinx()
        ylim_low, ylim_high = ax1.get_ylim()
        
        # Synchronize scales
        ax1_right.set_ylim(ylim_low, ylim_high)
        ax1_right.set_yticks(ax1.get_yticks())
        # Convert 100-based index to pure percentage labels
        ax1_right.set_yticklabels([f"{y-100:.0f}%" for y in ax1.get_yticks()])
        ax1_right.grid(False) 
        ax1_right.set_ylabel("Total Return (%)", rotation=270, labelpad=20, weight='bold')

        # --- Final Return Annotation Tag ---
        if not portfolio_equity_curve.empty:
            final_val = portfolio_equity_curve.iloc[-1]
            final_ret = final_val - 100
            
            # Green for profit, Red for loss
            tag_color = '#27ae60' if final_ret >= 0 else '#c0392b'
            
            # Annotation box at the end of the line
            ax1.annotate(f"{final_ret:+.2f}% (${final_pnl:,.0f})", 
                        xy=(portfolio_equity_curve.index[-1], final_val),
                        xytext=(10, 0), textcoords='offset points',
                        va='center', weight='bold', color='white',
                        bbox=dict(boxstyle="round,pad=0.3", fc=tag_color, ec="none"))

        ax1.set_title(f"Aggregated Portfolio Performance (Capital Weighted: Est ${total_capital:,.0f})", fontsize=16, weight='bold')
        ax1.set_ylabel("Portfolio Value (Base 100)")
        ax1.legend(loc='upper left')

        # 2. Pair Performance Ranking (Bar Chart)
        pair_names = [p['pair'] for p in pair_summaries]
        pair_returns = [p['cum_return'] * 100 for p in pair_summaries]
        pair_sharpes = [p['sharpe'] for p in pair_summaries]
        skipped_names = [s['pair'] for s in skipped_pairs]

        ax2 = fig.add_subplot(gs[1, :])
        colors = [self.colors['profit'] if r > 0 else self.colors['loss'] for r in pair_returns]
        edge_colors = ['red' if name in skipped_names else 'none' for name in pair_names]
        
        bars = ax2.bar(pair_names, pair_returns, color=colors, edgecolor=edge_colors, linewidth=2)
        ax2.axhline(0, color='black')
        ax2.set_title("Individual Pair Returns (%)", fontsize=16, weight='bold')
        
        # Add 'Stopped' icon
        for bar, name in zip(bars, pair_names):
            if name in skipped_names:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, '⛔', 
                         ha='center', va='bottom', fontsize=12)

        # 3. Risk-Reward Scatter
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.scatter(pair_sharpes, pair_returns, s=150, alpha=0.7, c='#8e44ad')
        for i, txt in enumerate(pair_names):
            ax3.annotate(txt, (pair_sharpes[i], pair_returns[i]), xytext=(5,5), textcoords='offset points')
        ax3.axvline(0, linestyle='--', color='gray')
        ax3.axhline(0, linestyle='--', color='gray')
        ax3.set_xlabel("Sharpe Ratio")
        ax3.set_ylabel("Total Return (%)")
        ax3.set_title("Risk/Reward Clustering", fontsize=14, weight='bold')

        # 4. Correlation Heatmap (Based on P&L changes)
        ax4 = fig.add_subplot(gs[2, 1])
        # Use daily change in P&L for correlation in capital-weighted portfolio
        pnl_changes = pnl_matrix.diff().fillna(0.0)
        
        if pnl_changes.shape[1] > 1:
            corr_matrix = pnl_changes.corr()
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax4, square=True, cbar=False)
            ax4.set_title("Strategy Correlation Matrix (P&L)", fontsize=14, weight='bold')
        else:
            ax4.text(0.5, 0.5, "Insufficient Pairs for Correlation", ha='center')

        # 5. Drawdown Distribution
        ax5 = fig.add_subplot(gs[2, 2])
        pair_dds = [p['max_drawdown']*100 for p in pair_summaries]
        sns.histplot(pair_dds, bins=10, color=self.colors['drawdown'], kde=True, ax=ax5)
        ax5.set_title("Max Drawdown Distribution", fontsize=14, weight='bold')

        # 6. Global Stats Table
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        flat_metrics = [
            ("Total Net P&L", f"${final_pnl:,.2f}"),
            ("Total Return", f"{final_pct:.2f}%"),
            ("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"),
            ("Win Rate", f"{metrics['win_rate']*100:.2f}%"),
            ("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%"),
            ("VaR (95%)", f"{metrics['var_95']:.4f}"),
            ("Total Trades", f"{metrics['total_steps']}"),
            ("Avg Steps/Pair", f"{metrics['avg_steps_per_pair']:.1f}")
        ]
        
        cell_text = []
        # Format table into 2 columns
        for i in range(0, len(flat_metrics), 2):
            row = [flat_metrics[i][0], flat_metrics[i][1]]
            if i+1 < len(flat_metrics):
                row.extend([flat_metrics[i+1][0], flat_metrics[i+1][1]])
            else:
                row.extend(["", ""])
            cell_text.append(row)
            
        table = ax6.table(cellText=cell_text, colLabels=["Metric", "Value", "Metric", "Value"], 
                          loc='center', cellLoc='center', bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2c3e50')
            else:
                cell.set_facecolor('#ecf0f1' if row % 2 == 0 else 'white')

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"portfolio_dashboard_{timestamp}.png")
        plt.savefig(filepath)
        plt.close()
        
        print(f"\n📊 Saved portfolio dashboard: {filepath}")

    def _calculate_sharpe(self, returns):
        if len(returns) < 2: return 0.0
        rf = CONFIG.get("risk_free_rate", 0.04) / 252
        exc = np.array(returns) - rf
        std = np.std(exc, ddof=1)
        return (np.mean(exc) / std) * np.sqrt(252) if std > 1e-8 else 0.0

def generate_all_visualizations(all_traces: List[Dict], 
                                skipped_pairs: List[Dict],
                                final_summary: Dict,
                                output_dir: str = "reports"):
    """
    Generate complete visual report for portfolio and all pairs.
    """
    print("\n" + "="*70)
    print("GENERATING VISUAL REPORTS")
    print("="*70)
    
    visualizer = PortfolioVisualizer(output_dir)
    
    # Group traces by pair
    traces_by_pair = {}
    for t in all_traces:
        pair = t['pair']
        if pair not in traces_by_pair:
            traces_by_pair[pair] = []
        traces_by_pair[pair].append(t)
    
    # Generate individual pair reports
    print("\n📊 Generating pair-level reports...")
    for pair_name, traces in traces_by_pair.items():
        skip_info = next((s for s in skipped_pairs if s['pair'] == pair_name), None)
        was_skipped = skip_info is not None
        visualizer.visualize_pair(traces, pair_name, was_skipped, skip_info)
    
    # Generate portfolio aggregate
    print("\n📊 Generating portfolio aggregate report...")
    visualizer.visualize_portfolio(all_traces, skipped_pairs, final_summary)
    
    print(f"\n✅ All reports saved to: {output_dir}/")
