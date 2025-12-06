import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from typing import List, Dict, Any, Tuple
from datetime import datetime
import textwrap

# --- CONFIG IMPORT ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from config import CONFIG
except ImportError:
    CONFIG = {"risk_free_rate": 0.04} 

# --- GLOBAL STYLE SETTINGS ---
sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.family'] = 'sans-serif'

class PortfolioVisualizer:
    """
    Creates institutional-grade visual reports for pairs trading performance.
    Features homogenized styling, intelligent data summarization, and AI Text Reports.
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "pairs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "behavior"), exist_ok=True) # New folder for behavior plots
        
        # Institutional Color Palette
        self.colors = {
            'primary': '#2c3e50',      # Dark Slate (Prices/Equity)
            'profit': '#27ae60',       # Emerald Green
            'loss': '#c0392b',         # Pomegranate Red
            'drawdown': '#e74c3c',     # Soft Red
            'zscore': '#8e44ad',       # Purple
            'fill_profit': '#2ecc71',
            'fill_loss': '#e74c3c',
            'neutral': '#95a5a6',
            'accent': '#f39c12',       # Orange (Warnings)
            'text_bg': '#fdfefe',      # Very light grey for text card
            'asset_x': '#2980b9',      # Blue for Asset X
            'asset_y': '#7f8c8d'       # Grey for Asset Y
        }
    
    def visualize_pair(self, traces: List[Dict], pair_name: str, was_skipped: bool = False, skip_info: Dict = None):
        """
        Create detailed visualization for a single pair including Z-Score and Drawdown analysis.
        """
        if len(traces) == 0:
            return
        
        # --- Data Prep ---
        df = pd.DataFrame(traces)
        
        # Fill missing columns for robustness
        required_cols = {
            'forced_close': False, 'trade_occurred': False, 'daily_return': 0.0,
            'realized_pnl': 0.0, 'max_drawdown': 0.0, 'cum_return': 0.0, 'position': 0.0
        }
        for col, val in required_cols.items():
            if col not in df.columns: df[col] = val

        steps = df['local_step']
        df['cum_return_pct'] = df['cum_return'] * 100
        df['drawdown_pct'] = df['max_drawdown'] * 100
        
        trades = df[df['trade_occurred'] == True]
        forced_exit = df[df['forced_close'] == True]

        # Stats
        total_pnl = df['realized_pnl'].iloc[-1]
        final_ret = df['cum_return'].iloc[-1]
        win_rate = (df[df['daily_return'] != 0]['daily_return'] > 0).mean() if not df[df['daily_return'] != 0].empty else 0.0
        
        # --- Plotting ---
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # Title
        status_color = self.colors['loss'] if was_skipped else self.colors['primary']
        status_text = f" | STOPPED: {skip_info['reason']}" if was_skipped and skip_info else ""
        fig.suptitle(f"{pair_name} Performance Analysis{status_text}", 
                     fontsize=22, weight='bold', color=status_color, y=0.96)

        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :3])
        ax1.plot(steps, 100 * (1 + df['cum_return']), color=self.colors['primary'], lw=2.5, label='Portfolio Value')
        ax1.axhline(100, color=self.colors['neutral'], linestyle='--', alpha=0.5)
        
        # Fill
        ax1.fill_between(steps, 100, 100 * (1 + df['cum_return']), 
                         where=(df['cum_return'] >= 0), color=self.colors['fill_profit'], alpha=0.15)
        ax1.fill_between(steps, 100, 100 * (1 + df['cum_return']), 
                         where=(df['cum_return'] < 0), color=self.colors['fill_loss'], alpha=0.15)

        if not forced_exit.empty:
            ax1.scatter(forced_exit['local_step'], 100 * (1 + forced_exit['cum_return']), 
                        color=self.colors['accent'], s=200, marker='X', label='Forced Exit', zorder=5, edgecolor='white')

        ax1.set_ylabel('Equity (Base 100)')
        ax1.set_title('Equity Curve', loc='left')
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(True, alpha=0.3)

        # 2. Scorecard
        ax2 = fig.add_subplot(gs[0, 3])
        ax2.axis('off')
        
        metrics = [
            ("Total P&L", f"${total_pnl:,.2f}", self.colors['profit'] if total_pnl > 0 else self.colors['loss']),
            ("Return", f"{final_ret*100:+.2f}%", self.colors['profit'] if final_ret > 0 else self.colors['loss']),
            ("Max Drawdown", f"{df['max_drawdown'].max()*100:.2f}%", self.colors['drawdown']),
            ("Win Rate", f"{win_rate*100:.1f}%", self.colors['primary']),
            ("Sharpe", f"{self._calculate_sharpe(df['daily_return']):.2f}", self.colors['primary']),
            ("Trades", f"{len(trades)}", self.colors['primary'])
        ]
        
        y_pos = 0.9
        ax2.text(0.5, 1.0, "Key Metrics", ha='center', fontsize=18, weight='bold', color=self.colors['primary'])
        
        for label, value, color in metrics:
            ax2.text(0.1, y_pos, label, ha='left', fontsize=14, color=self.colors['neutral'])
            ax2.text(0.9, y_pos, value, ha='right', fontsize=15, weight='bold', color=color)
            ax2.plot([0.1, 0.9], [y_pos-0.05, y_pos-0.05], color='#ecf0f1', lw=1.5)
            y_pos -= 0.15

        # 3. Z-Score & Position
        ax3 = fig.add_subplot(gs[1, :])
        
        if 'current_spread' in df.columns:
            spread = df['current_spread']
            # Rolling Z-score for visualization
            zscore = (spread - spread.rolling(30).mean()) / (spread.rolling(30).std() + 1e-8)
            ax3.plot(steps, zscore, color=self.colors['zscore'], alpha=0.8, lw=1.5, label='Spread Z-Score')
            ax3.axhline(2.0, color=self.colors['neutral'], linestyle='--', alpha=0.5)
            ax3.axhline(-2.0, color=self.colors['neutral'], linestyle='--', alpha=0.5)
            ax3.axhline(0, color=self.colors['primary'], lw=1)
            ax3.set_ylabel('Z-Score')
        
        ax3b = ax3.twinx()
        ax3b.fill_between(steps, df['position'], color='black', alpha=0.1, step='post', label='Position Size')
        ax3b.set_ylabel('Position')
        ax3b.grid(False)
        ax3.set_title('Z-Score Signal vs. Position Execution', loc='left')

        # 4. Drawdown
        ax4 = fig.add_subplot(gs[2, :2])
        ax4.fill_between(steps, 0, -df['drawdown_pct'], color=self.colors['drawdown'], alpha=0.3)
        ax4.plot(steps, -df['drawdown_pct'], color=self.colors['drawdown'], lw=1.5)
        ax4.set_title('Underwater Plot (Drawdown %)', loc='left')
        ax4.set_ylabel('Drawdown %')
        ax4.grid(True, alpha=0.3)

        # 5. Daily Returns
        ax5 = fig.add_subplot(gs[2, 2:])
        if not df[df['daily_return'] != 0].empty:
            sns.histplot(df[df['daily_return'] != 0]['daily_return'], kde=True, ax=ax5, color=self.colors['primary'], alpha=0.6)
        ax5.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax5.set_title('Daily Returns Distribution', loc='left')
        ax5.set_xlabel('Return')

        # 6. Cumulative P&L
        ax6 = fig.add_subplot(gs[3, :])
        cum_pnl = df['realized_pnl']
        ax6.plot(steps, cum_pnl, color=self.colors['primary'], lw=2)
        ax6.fill_between(steps, 0, cum_pnl, where=(cum_pnl>=0), color=self.colors['fill_profit'], alpha=0.2)
        ax6.fill_between(steps, 0, cum_pnl, where=(cum_pnl<0), color=self.colors['fill_loss'], alpha=0.2)
        ax6.set_title('Cumulative Realized P&L ($)', loc='left')
        ax6.grid(True, alpha=0.3)

        # Save
        filename = f"{pair_name.replace('-', '_')}_analysis.png"
        filepath = os.path.join(self.output_dir, "pairs", filename)
        plt.savefig(filepath)
        plt.close()
        print(f"   📊 Saved pair analysis: {filepath}")

    def visualize_pair_behavior(self, traces: List[Dict], pair_name: str):
        """
        Creates a 'Behavior Analysis' report comparing Price Action vs Strategy Returns.
        Useful for debugging why a pair worked or failed.
        """
        if not traces: return
        df = pd.DataFrame(traces)
        
        # Check if we have price data
        if 'price_x' not in df.columns or 'price_y' not in df.columns:
            print(f"⚠️ Cannot generate behavior report for {pair_name}: Missing raw price data.")
            return

        # Normalize prices to start at 1.0 for comparison
        df['norm_x'] = df['price_x'] / df['price_x'].iloc[0]
        df['norm_y'] = df['price_y'] / df['price_y'].iloc[0]
        
        # Setup Figure
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.15)
        
        # 1. Normalized Price Action (Top Panel)
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df['local_step'], df['norm_x'], color=self.colors['asset_x'], lw=2, label=f"Asset X (Normalized)")
        ax1.plot(df['local_step'], df['norm_y'], color=self.colors['asset_y'], lw=2, label=f"Asset Y (Normalized)")
        
        # Fill the spread area to visualize divergence
        ax1.fill_between(df['local_step'], df['norm_x'], df['norm_y'], color='gray', alpha=0.1, label="Spread Divergence")
        
        ax1.set_ylabel("Normalized Price (Start=1.0)")
        ax1.set_title(f"{pair_name}: Price Co-movement Analysis", loc='left', fontsize=16, weight='bold', color=self.colors['primary'])
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        # Hide x-labels for top plot
        plt.setp(ax1.get_xticklabels(), visible=False)

        # 2. Strategy Cumulative Return (Bottom Panel)
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        cum_ret_pct = df['cum_return'] * 100
        ax2.plot(df['local_step'], cum_ret_pct, color=self.colors['primary'], lw=2, label="Strategy Return %")
        
        # Color fill based on profitability
        ax2.fill_between(df['local_step'], 0, cum_ret_pct, where=(cum_ret_pct >= 0), color=self.colors['fill_profit'], alpha=0.2)
        ax2.fill_between(df['local_step'], 0, cum_ret_pct, where=(cum_ret_pct < 0), color=self.colors['fill_loss'], alpha=0.2)
        
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_ylabel("Cumulative Return (%)")
        ax2.set_xlabel("Trading Steps")
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Save
        filename = f"behavior_{pair_name.replace('-', '_')}.png"
        filepath = os.path.join(self.output_dir, "behavior", filename)
        plt.savefig(filepath)
        plt.close()
        print(f"   📉 Saved behavior analysis: {filepath}")

    def visualize_portfolio(self, all_traces: List[Dict], skipped_pairs: List[Dict], final_summary: Dict):
        """
        Create aggregated portfolio dashboard with improved Correlation analysis.
        """
        metrics = final_summary['metrics']
        pair_summaries = metrics['pair_summaries']
        
        # --- Data Processing ---
        df_all = pd.DataFrame(all_traces)
        time_col = 'timestamp' if 'timestamp' in df_all.columns else 'local_step'

        # Total Capital Est
        if 'position_value' in df_all.columns:
            total_capital = df_all.groupby('pair')['position_value'].max().sum()
        else:
            total_capital = 10000 * len(df_all['pair'].unique())
        total_capital = max(total_capital, 1.0) 

        # P&L Matrix
        pnl_matrix = df_all.pivot_table(index=time_col, columns='pair', values='realized_pnl')
        pnl_matrix = pnl_matrix.ffill().fillna(0.0)
        
        # Portfolio Curve
        total_portfolio_pnl_dollars = pnl_matrix.sum(axis=1)
        portfolio_return_pct = (total_portfolio_pnl_dollars / total_capital) * 100
        portfolio_equity_curve = 100 + portfolio_return_pct

        # --- Figure Setup ---
        fig = plt.figure(figsize=(22, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.25)
        
        final_pnl = total_portfolio_pnl_dollars.iloc[-1]
        final_pct = portfolio_return_pct.iloc[-1]
        
        fig.suptitle(f"Portfolio Executive Dashboard | Net P&L: ${final_pnl:,.2f} | Return: {final_pct:+.2f}%", 
                     fontsize=24, weight='bold', color=self.colors['primary'], y=0.96)

        # 1. Main Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(portfolio_equity_curve.index, portfolio_equity_curve, color=self.colors['primary'], lw=3, label='Portfolio Value')
        ax1.fill_between(portfolio_equity_curve.index, 100, portfolio_equity_curve, 
                         color=self.colors['primary'], alpha=0.1)
        ax1.axhline(100, linestyle='--', color=self.colors['neutral'], alpha=0.8)
        
        # Right Axis (Return %)
        ax1_right = ax1.twinx()
        ax1_right.set_ylim(ax1.get_ylim())
        ax1_right.set_yticks(ax1.get_yticks())
        ax1_right.set_yticklabels([f"{y-100:.0f}%" for y in ax1.get_yticks()])
        ax1_right.set_ylabel("Total Return (%)", rotation=270, labelpad=20, weight='bold', color=self.colors['neutral'])
        ax1_right.grid(False)

        # Final Tag
        if not portfolio_equity_curve.empty:
            val = portfolio_equity_curve.iloc[-1]
            ret = val - 100
            color = self.colors['profit'] if ret >= 0 else self.colors['loss']
            ax1.annotate(f"{ret:+.2f}%", xy=(portfolio_equity_curve.index[-1], val),
                         xytext=(10, 0), textcoords='offset points', va='center', weight='bold', color='white',
                         bbox=dict(boxstyle="round,pad=0.4", fc=color, ec="none"))

        ax1.set_title(f"Aggregated Performance (Est. Capital: ${total_capital:,.0f})", loc='left')
        ax1.set_ylabel("Equity (Base 100)")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. Pair Returns Ranking
        pair_names = [p['pair'] for p in pair_summaries]
        pair_returns = [p['cum_return'] * 100 for p in pair_summaries]
        skipped_names = [s['pair'] for s in skipped_pairs]

        ax2 = fig.add_subplot(gs[1, :])
        colors = [self.colors['profit'] if r > 0 else self.colors['loss'] for r in pair_returns]
        bars = ax2.bar(pair_names, pair_returns, color=colors, alpha=0.8, width=0.6)
        ax2.axhline(0, color='black', lw=1)
        ax2.set_title("Individual Pair Contribution (%)", loc='left')
        ax2.set_ylabel("Return %")
        
        # Rotated labels if many pairs
        if len(pair_names) > 10:
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # 3. Correlation Analysis
        pnl_changes = pnl_matrix.diff().fillna(0.0)
        
        if pnl_changes.shape[1] > 1:
            corr_matrix = pnl_changes.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            
            # 3a. Correlation Distribution (Histogram)
            corr_values = corr_matrix.where(mask).stack().values
            ax3 = fig.add_subplot(gs[2, 0])
            sns.histplot(corr_values, kde=True, ax=ax3, color=self.colors['zscore'], bins=15, alpha=0.6)
            ax3.set_title("Diversification Health (Correlation Dist.)", loc='left')
            ax3.set_xlabel("Pairwise Correlation")
            ax3.set_ylabel("Frequency")
            if len(corr_values) > 0:
                ax3.axvline(np.mean(corr_values), color='black', linestyle='--', label=f'Avg: {np.mean(corr_values):.2f}')
                ax3.legend()

            # 3b. Top Concentration Risks (Table)
            corr_stacked = corr_matrix.where(mask).stack()
            corr_stacked.index.names = ['Pair A', 'Pair B'] 
            corr_pairs = corr_stacked.reset_index()
            corr_pairs.columns = ['Pair A', 'Pair B', 'Corr']
            
            top_corr = corr_pairs.sort_values('Corr', ascending=False).head(8)
            
            ax4 = fig.add_subplot(gs[2, 1])
            ax4.axis('off')
            ax4.set_title("⚠️ Top Concentration Risks", loc='center', color=self.colors['loss'])
            
            cell_text = []
            for _, row in top_corr.iterrows():
                val_color = self.colors['loss'] if row['Corr'] > 0.7 else self.colors['primary']
                cell_text.append([f"{row['Pair A']} vs {row['Pair B']}", f"{row['Corr']:.2f}"])
            
            if not cell_text:
                cell_text = [["No significant correlations", "-"]]

            table = ax4.table(cellText=cell_text, colLabels=["Pair Combination", "Correlation"], 
                              loc='center', cellLoc='center', bbox=[0, 0, 1, 0.9])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            
        else:
            # Fallback if 1 pair
            ax3 = fig.add_subplot(gs[2, :2])
            ax3.text(0.5, 0.5, "Insufficient Data for Correlation Analysis", ha='center', fontsize=14)
            ax3.axis('off')

        # 4. Max Drawdown Distribution
        ax5 = fig.add_subplot(gs[2, 2])
        pair_dds = [p['max_drawdown']*100 for p in pair_summaries]
        sns.histplot(pair_dds, bins=10, color=self.colors['drawdown'], kde=True, ax=ax5, alpha=0.6)
        ax5.set_title("Risk Profile (Max Drawdown Dist.)", loc='left')
        ax5.set_xlabel("Drawdown %")

        # 5. Global Stats Table
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        flat_metrics = [
            ("Total Net P&L", f"${final_pnl:,.2f}"),
            ("Total Return", f"{final_pct:+.2f}%"),
            ("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"),
            ("Win Rate", f"{metrics['win_rate']*100:.2f}%"),
            ("Max Portfolio DD", f"{metrics['max_drawdown']*100:.2f}%"),
            ("VaR (95%)", f"{metrics['var_95']:.4f}"),
            ("Active Pairs", f"{len(pair_names)}"),
            ("Skipped Pairs", f"{len(skipped_names)}")
        ]
        
        col_count = 4
        row_count = 2
        cell_text = [ [] for _ in range(row_count) ]
        
        for i, (label, val) in enumerate(flat_metrics):
            r = i % row_count
            cell_text[r].extend([label, val])

        table = ax6.table(cellText=cell_text, loc='center', cellLoc='center', bbox=[0.05, 0.2, 0.9, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('white')
            if col % 2 == 0:
                cell.set_facecolor('#f7f9f9')
                cell.set_text_props(weight='bold', color=self.colors['primary'])
            else:
                cell.set_facecolor('white')
                cell.set_text_props(color='black')

        # Save Main Dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"portfolio_dashboard_{timestamp}.png")
        plt.savefig(filepath)
        plt.close()
        print(f"\n📊 Saved portfolio dashboard: {filepath}")
    
    def visualize_executive_summary(self, explanation_text: str):
        """
        Creates a dedicated Executive Summary card with the Gemini explanation.
        """
        if not explanation_text:
            return

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor(self.colors['text_bg'])
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Wrapper to prevent text running off
        wrapper = textwrap.TextWrapper(width=90, replace_whitespace=False)
        formatted_text = ""
        
        # Formatting to ensure paragraphs are preserved
        paragraphs = explanation_text.split('\n')
        for p in paragraphs:
            if p.strip():
                formatted_text += "\n".join(wrapper.wrap(p)) + "\n\n"

        # Title
        ax.text(0.05, 0.95, "Risk Manager: Executive Summary", 
                fontsize=24, weight='bold', color=self.colors['primary'], va='top')
        
        # Body
        ax.text(0.05, 0.85, formatted_text, 
                fontsize=16, color='#2c3e50', va='top', ha='left', family='monospace')

        # Footer
        ax.text(0.5, 0.05, "Generated by AI Supervisor Agent", 
                fontsize=12, color=self.colors['neutral'], ha='center')

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"executive_summary_{timestamp}.png")
        plt.savefig(filepath, facecolor=self.colors['text_bg'])
        plt.close()
        print(f"📄 Saved executive summary: {filepath}")

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
        
        # 1. Standard Analysis
        visualizer.visualize_pair(traces, pair_name, was_skipped, skip_info)
        
        # 2. NEW Behavior Analysis (Price vs Return)
        visualizer.visualize_pair_behavior(traces, pair_name)
    
    # Generate portfolio aggregate
    print("\n📊 Generating portfolio aggregate report...")
    visualizer.visualize_portfolio(all_traces, skipped_pairs, final_summary)
    
    # Generate Executive Summary Text Card
    if 'explanation' in final_summary:
        print("\n📄 Generating executive summary card...")
        visualizer.visualize_executive_summary(final_summary['explanation'])
    
    print(f"\n✅ All reports saved to: {output_dir}/")
