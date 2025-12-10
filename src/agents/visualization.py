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

# Ensure config is importable
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
        os.makedirs(os.path.join(output_dir, "behavior"), exist_ok=True)
        
        # Institutional Color Palette
        self.colors = {
            'primary': '#2c3e50',       # Dark Slate (Prices/Equity)
            'profit': '#27ae60',        # Emerald Green
            'loss': '#c0392b',          # Pomegranate Red
            'drawdown': '#e74c3c',      # Soft Red
            'zscore': '#8e44ad',        # Purple
            'fill_profit': '#2ecc71',
            'fill_loss': '#e74c3c',
            'neutral': '#95a5a6',
            'accent': '#f39c12',        # Orange (Warnings)
            'text_bg': '#fdfefe',       # Very light grey for text card
            'asset_x': '#2980b9',       # Blue for Asset X
            'asset_y': '#7f8c8d'        # Grey for Asset Y
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
            'realized_pnl': 0.0, 'max_drawdown': 0.0, 'cum_return': 0.0, 'position': 0.0,
            'realized_pnl_this_step': 0.0, 'transaction_costs': 0.0, 'z_score': 0.0, 'current_spread': 0.0
        }
        for col, val in required_cols.items():
            if col not in df.columns: df[col] = val

        steps = df['local_step']
        # cum_return in traces is a decimal (e.g., 0.05), convert to percent for display
        df['cum_return_pct'] = df['cum_return'] * 100
        df['drawdown_pct'] = df['max_drawdown'] * 100
        
        trades_entry = df[df['trade_occurred'] == True] # Use trade_occurred to find entries/exits
        forced_exit = df[df['forced_close'] == True]

        # --- Win Rate Logic (Per Pair) ---
        # A realized PnL change (realized_pnl_this_step != 0) signifies a trade closure or flip.
        closed_trades_mask = df['realized_pnl_this_step'] != 0
        closed_trades_df = df[closed_trades_mask].copy()
        
        # Net PnL of the realized trade = Gross PnL - Transaction Cost
        closed_trades_df['net_trade_pnl'] = closed_trades_df['realized_pnl_this_step'] - closed_trades_df['transaction_costs']
        
        total_closed_trades = len(closed_trades_df)
        if total_closed_trades > 0:
            # Winning trades are those where the net PnL is positive
            winning_trades = (closed_trades_df['net_trade_pnl'] > 0).sum()
            win_rate = winning_trades / total_closed_trades
        else:
            win_rate = 0.0
            
        # Final metrics extraction
        total_pnl = df['realized_pnl'].iloc[-1]
        final_ret = df['cum_return'].iloc[-1]
        total_entries = len(trades_entry)

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
        # Equity is 100 * (1 + decimal return)
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
        
        # total_pnl is realized_pnl. cum_return is final_ret.
        metrics = [
            ("Total Realized P&L", f"${total_pnl:,.2f}", self.colors['profit'] if total_pnl > 0 else self.colors['loss']),
            ("Final Return", f"{final_ret*100:+.2f}%", self.colors['profit'] if final_ret > 0 else self.colors['loss']),
            ("Max Drawdown", f"{df['max_drawdown'].max()*100:.2f}%", self.colors['drawdown']),
            ("Trade Win Rate", f"{win_rate*100:.1f}%", self.colors['primary']), 
            ("Sharpe Ratio", f"{self._calculate_sharpe(df['daily_return']):.2f}", self.colors['primary']),
            ("Trades Executed", f"{total_entries}", self.colors['primary']) 
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
        
        # Use logged z_score if available
        if 'z_score' in df.columns:
            zscore = df['z_score']
            ax3.plot(steps, zscore, color=self.colors['zscore'], alpha=0.8, lw=1.5, label='Spread Z-Score')
        # Fallback to calculating Z-score from spread if logged z_score is missing
        elif 'current_spread' in df.columns:
            spread = df['current_spread']
            # Recompute Z-score using typical 30-day lookback for visualization
            zscore = (spread - spread.rolling(30).mean()) / (spread.rolling(30).std() + 1e-8)
            ax3.plot(steps, zscore, color=self.colors['zscore'], alpha=0.8, lw=1.5, label='Spread Z-Score (Est)')

        ax3.axhline(2.0, color=self.colors['loss'], linestyle='--', alpha=0.8)
        ax3.axhline(-2.0, color=self.colors['loss'], linestyle='--', alpha=0.8)
        ax3.axhline(0, color=self.colors['primary'], lw=1)
        ax3.set_ylabel('Z-Score')
        
        ax3b = ax3.twinx()
        # Position is already scaled (e.g., -100, 0, 100)
        ax3b.fill_between(steps, df['position'], color=self.colors['neutral'], alpha=0.1, step='post', label='Position Size')
        ax3b.set_ylabel('Position')
        ax3b.set_yticks([-100, 0, 100])
        ax3b.set_yticklabels(['Short', 'Flat', 'Long'])
        ax3b.grid(False)
        ax3.set_title('Z-Score Signal vs. Position Execution', loc='left')

        # 4. Drawdown
        ax4 = fig.add_subplot(gs[2, :2])
        # Drawdown is logged as a positive fraction/decimal, plot as negative for 'underwater' look
        ax4.fill_between(steps, 0, -df['drawdown_pct'], color=self.colors['drawdown'], alpha=0.3)
        ax4.plot(steps, -df['drawdown_pct'], color=self.colors['drawdown'], lw=1.5)
        ax4.set_title('Underwater Plot (Drawdown %)', loc='left')
        ax4.set_ylabel('Drawdown %')
        ax4.grid(True, alpha=0.3)

        # 5. Daily Returns
        ax5 = fig.add_subplot(gs[2, 2:])
        if not df[df['daily_return'] != 0].empty:
            # Daily returns are decimals (e.g., 0.01), scale to % for readability in plot
            sns.histplot(df[df['daily_return'] != 0]['daily_return'] * 100, kde=True, ax=ax5, color=self.colors['primary'], alpha=0.6)
        ax5.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax5.set_title('Daily Returns Distribution', loc='left')
        ax5.set_xlabel('Return %')
        ax5.grid(True, alpha=0.3)

        # 6. Cumulative Realized P&L
        ax6 = fig.add_subplot(gs[3, :])
        # realized_pnl is the cumulative P&L in dollars
        cum_pnl = df['realized_pnl']
        ax6.plot(steps, cum_pnl, color=self.colors['primary'], lw=2)
        ax6.fill_between(steps, 0, cum_pnl, where=(cum_pnl>=0), color=self.colors['fill_profit'], alpha=0.2)
        ax6.fill_between(steps, 0, cum_pnl, where=(cum_pnl<0), color=self.colors['fill_loss'], alpha=0.2)
        ax6.set_title('Cumulative Realized P&L ($)', loc='left')
        ax6.set_xlabel('Trading Steps')
        ax6.set_ylabel('P&L ($)')
        ax6.grid(True, alpha=0.3)

        # Save
        filename = f"{pair_name.replace('-', '_')}_analysis.png"
        filepath = os.path.join(self.output_dir, "pairs", filename)
        plt.savefig(filepath)
        plt.close()
        print(f"    📊 Saved pair analysis: {filepath}")

    def visualize_pair_behavior(self, traces: List[Dict], pair_name: str):
        """
        Creates a 'Behavior Analysis' report comparing Price Action vs Strategy Returns.
        """
        if not traces: return
        df = pd.DataFrame(traces)
        
        if 'price_x' not in df.columns or 'price_y' not in df.columns:
            print(f"⚠️ Cannot generate behavior report for {pair_name}: Missing raw price data.")
            return

        # Normalized Prices
        df['norm_x'] = df['price_x'] / df['price_x'].iloc[0]
        df['norm_y'] = df['price_y'] / df['price_y'].iloc[0]
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.15)
        
        # 1. Normalized Price Co-movement
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df['local_step'], df['norm_x'], color=self.colors['asset_x'], lw=2, label=f"Asset X (Normalized)")
        ax1.plot(df['local_step'], df['norm_y'], color=self.colors['asset_y'], lw=2, label=f"Asset Y (Normalized)")
        
        # Fill spread divergence area
        ax1.fill_between(df['local_step'], df['norm_x'], df['norm_y'], color='gray', alpha=0.1, label="Spread Divergence")
        
        ax1.set_ylabel("Normalized Price (Start=1.0)")
        ax1.set_title(f"{pair_name}: Price Co-movement Analysis", loc='left', fontsize=16, weight='bold', color=self.colors['primary'])
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # 2. Strategy Cumulative Return
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        # cum_return is a decimal (e.g., 0.05), plot as percent
        cum_ret_pct = df['cum_return'] * 100
        ax2.plot(df['local_step'], cum_ret_pct, color=self.colors['primary'], lw=2, label="Strategy Return %")
        
        ax2.fill_between(df['local_step'], 0, cum_ret_pct, where=(cum_ret_pct >= 0), color=self.colors['fill_profit'], alpha=0.2)
        ax2.fill_between(df['local_step'], 0, cum_ret_pct, where=(cum_ret_pct < 0), color=self.colors['fill_loss'], alpha=0.2)
        
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_ylabel("Cumulative Return (%)")
        ax2.set_xlabel("Trading Steps (Time)")
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        filename = f"behavior_{pair_name.replace('-', '_')}.png"
        filepath = os.path.join(self.output_dir, "behavior", filename)
        plt.savefig(filepath)
        plt.close()
        print(f"    📉 Saved behavior analysis: {filepath}")

    def visualize_portfolio(self, all_traces: List[Dict], skipped_pairs: List[Dict], final_summary: Dict):
        """
        Create aggregated portfolio dashboard with Correlation analysis and correct Portfolio-Wide Max Drawdown.
        """
        metrics = final_summary['metrics']
        pair_summaries = metrics['pair_summaries']
        
        # --- Data Processing ---
        df_all = pd.DataFrame(all_traces)
        if df_all.empty:
            print("⚠️ No traces to visualize.")
            return

        # Ensure necessary P&L columns exist for total trade calculation
        if 'realized_pnl_this_step' not in df_all.columns: df_all['realized_pnl_this_step'] = 0.0
        if 'transaction_costs' not in df_all.columns: df_all['transaction_costs'] = 0.0
        if 'cum_return' not in df_all.columns: df_all['cum_return'] = 0.0
        if 'realized_pnl' not in df_all.columns: df_all['realized_pnl'] = 0.0
        if 'local_step' not in df_all.columns: df_all['local_step'] = 0.0


        # --- Global Win Rate Logic ---
        trade_events = df_all[df_all['realized_pnl_this_step'] != 0].copy()
        trade_events['net_pnl'] = trade_events['realized_pnl_this_step'] - trade_events['transaction_costs']
        
        total_global_trades = len(trade_events)
        if total_global_trades > 0:
            global_wins = (trade_events['net_pnl'] > 0).sum()
            global_win_rate = global_wins / total_global_trades
        else:
            global_win_rate = 0.0

        # Use 'step' from the trace if available, otherwise 'local_step'
        time_col = 'step' if 'step' in df_all.columns else 'local_step'

        # Total Capital Est (Assume $10000 initial capital per pair)
        initial_capital_per_pair = 10000.0
        num_pairs = df_all['pair'].nunique()
        total_initial_capital = initial_capital_per_pair * num_pairs
        
        # P&L Matrix (Cumulative PnL per pair over the 'time_col' index)
        # Use 'realized_pnl' which is the cumulative PnL for that pair up to that step.
        pnl_matrix = df_all.pivot_table(index=time_col, columns='pair', values='realized_pnl')
        pnl_matrix = pnl_matrix.ffill().fillna(0.0) # Forward fill any missing time steps and start NaNs with 0
        
        # Portfolio Curve
        total_portfolio_pnl_dollars = pnl_matrix.sum(axis=1)
        portfolio_return_pct = (total_portfolio_pnl_dollars / total_initial_capital) * 100
        portfolio_equity_curve = 100 + portfolio_return_pct

        # --- AGGREGATE PORTFOLIO MAX DRAWDOWN ---
        running_max = portfolio_equity_curve.cummax()
        dd_series = (running_max - portfolio_equity_curve) / running_max
        portfolio_max_dd = abs(dd_series.max()) # Max drawdown is the largest value in the positive DD series
        
        print(f"    ℹ️  Calculated Global Portfolio Max Drawdown: {portfolio_max_dd*100:.2f}%")
        # -------------------------------------------------------

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
        
        ax1_right = ax1.twinx()
        ax1_right.set_ylim(ax1.get_ylim())
        ax1_right.set_yticks(ax1.get_yticks())
        ax1_right.set_yticklabels([f"{y-100:.0f}%" for y in ax1.get_yticks()])
        ax1_right.set_ylabel("Total Return (%)", rotation=270, labelpad=20, weight='bold', color=self.colors['neutral'])
        ax1_right.grid(False)

        if not portfolio_equity_curve.empty:
            val = portfolio_equity_curve.iloc[-1]
            ret = val - 100
            color = self.colors['profit'] if ret >= 0 else self.colors['loss']
            ax1.annotate(f"{ret:+.2f}%", xy=(portfolio_equity_curve.index[-1], val),
                          xytext=(10, 0), textcoords='offset points', va='center', weight='bold', color='white',
                          bbox=dict(boxstyle="round,pad=0.4", fc=color, ec="none"))

        ax1.set_title(f"Aggregated Performance (Est. Initial Capital: ${total_initial_capital:,.0f})", loc='left')
        ax1.set_ylabel("Equity (Base 100)")
        ax1.set_xlabel("Steps (Time)")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. Pair Returns Ranking
        pair_names = [p['pair'] for p in pair_summaries]
        pair_returns = [p['return'] for p in pair_summaries] # Use 'return' from summary (which is already percent)
        skipped_names = [s['pair'] for s in skipped_pairs]

        ax2 = fig.add_subplot(gs[1, :])
        colors = [self.colors['profit'] if r > 0 else self.colors['loss'] for r in pair_returns]
        bars = ax2.bar(pair_names, pair_returns, color=colors, alpha=0.8, width=0.6)
        ax2.axhline(0, color='black', lw=1)
        ax2.set_title("Individual Pair Contribution (%)", loc='left')
        ax2.set_ylabel("Return %")
        
        if len(pair_names) > 10:
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # 3. Correlation Analysis
        # Use PnL changes (daily PnL) to determine correlation
        pnl_changes = pnl_matrix.diff().fillna(0.0)
        
        if pnl_changes.shape[1] > 1:
            # Calculate daily correlation matrix
            corr_matrix = pnl_changes.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            
            # 3a. Histogram
            corr_values = corr_matrix.where(mask).stack().values
            ax3 = fig.add_subplot(gs[2, 0])
            sns.histplot(corr_values, kde=True, ax=ax3, color=self.colors['zscore'], bins=15, alpha=0.6)
            ax3.set_title("Diversification Health (Correlation Dist.)", loc='left')
            ax3.set_xlabel("Pairwise Correlation")
            ax3.set_ylabel("Frequency")
            if len(corr_values) > 0:
                ax3.axvline(np.mean(corr_values), color='black', linestyle='--', label=f'Avg: {np.mean(corr_values):.2f}')
                ax3.legend()

            # 3b. Top Risk Table
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
                cell_text.append([f"{row['Pair A']}-{row['Pair B']}", f"{row['Corr']:.2f}"])
            
            if not cell_text:
                cell_text = [["No significant correlations", "-"]]

            table = ax4.table(cellText=cell_text, colLabels=["Pair Combination", "Correlation"], 
                              loc='center', cellLoc='center', bbox=[0, 0, 1, 0.9])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            
        else:
            ax3 = fig.add_subplot(gs[2, :2])
            ax3.text(0.5, 0.5, "Insufficient Data for Correlation Analysis", ha='center', fontsize=14)
            ax3.axis('off')

        # 4. Max Drawdown Distribution
        ax5 = fig.add_subplot(gs[2, 2])
        # 'drawdown' in pair_summaries is already a decimal (e.g., 0.18), convert to percent
        pair_dds = [p['drawdown']*100 for p in pair_summaries]
        sns.histplot(pair_dds, bins=10, color=self.colors['drawdown'], kde=True, ax=ax5, alpha=0.6)
        ax5.set_title("Risk Profile (Max Drawdown Dist.)", loc='left')
        ax5.set_xlabel("Drawdown %")

        # 5. Global Stats Table
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # Calculate global Sharpe Ratio using all daily returns
        all_daily_returns = df_all['daily_return'].to_numpy()
        global_sharpe = self._calculate_sharpe(all_daily_returns)
        
        flat_metrics = [
            ("Total Net P&L", f"${final_pnl:,.2f}"),
            ("Total Return", f"{final_pct:+.2f}%"),
            ("Sharpe Ratio", f"{global_sharpe:.2f}"), # Use calculated global sharpe
            ("Win Rate", f"{global_win_rate*100:.2f}%"),
            ("Portfolio Max DD", f"{portfolio_max_dd*100:.2f}%"),
            ("VaR (95%)", f"{metrics.get('var_95', 0.0):.4f}"), # Use VaR from summary if present
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
        Optimized to dynamically resize height so text is never cut off.
        """
        if not explanation_text:
            return

        # 1. Formatting and Wrapping text first to calculate size
        # Increased width to 100 characters for better fit
        wrapper = textwrap.TextWrapper(width=100, replace_whitespace=False)
        formatted_text = ""
        
        paragraphs = explanation_text.split('\n')
        for p in paragraphs:
            if p.strip():
                formatted_text += "\n".join(wrapper.wrap(p)) + "\n\n"
        
        # 2. Calculate required height
        # Estimate: Base height + (Lines * Height per line)
        num_lines = formatted_text.count('\n')
        estimated_height = max(10, 4 + (num_lines * 0.4)) # 0.4 inch per line buffer

        # 3. Create Figure with Dynamic Height
        fig = plt.figure(figsize=(18, estimated_height))
        fig.patch.set_facecolor(self.colors['text_bg'])
        ax = fig.add_subplot(111)
        ax.axis('off')

        # 4. Render Text
        # Header
        ax.text(0.05, 0.98, "Risk Manager: Executive Summary", 
                 fontsize=24, weight='bold', color=self.colors['primary'], va='top', ha='left')
        
        # Body (Aligned Top-Left)
        # We start slightly lower (0.92) to leave room for header
        ax.text(0.05, 0.92, formatted_text, 
                 fontsize=16, color='#2c3e50', va='top', ha='left', family='monospace')

        # Footer
        ax.text(0.5, 0.02, "Generated by AI Supervisor Agent", 
                 fontsize=12, color=self.colors['neutral'], ha='center', va='bottom')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"executive_summary_{timestamp}.png")
        plt.savefig(filepath, facecolor=self.colors['text_bg'])
        plt.close()
        print(f"📄 Saved executive summary: {filepath}")

    def _calculate_sharpe(self, returns):
        """Calculates the Annualized Sharpe Ratio."""
        if len(returns) < 2: return 0.0
        # Use try/except or config check to get risk_free_rate safely
        try:
            risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
        except NameError:
            risk_free_rate = 0.04
            
        rf = risk_free_rate / 252 
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
        visualizer.visualize_pair_behavior(traces, pair_name)
    
    # Generate portfolio aggregate
    print("\n📊 Generating portfolio aggregate report...")
    visualizer.visualize_portfolio(all_traces, skipped_pairs, final_summary)
    
    # Generate Executive Summary Text Card
    if 'explanation' in final_summary:
        print("\n📄 Generating executive summary card...")
        visualizer.visualize_executive_summary(final_summary['explanation'])
    
    print(f"\n✅ All reports saved to: {output_dir}/")

def calculate_sharpe(traces, risk_free_rate=None):
    if risk_free_rate is None:
        risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
    returns = np.array([t.get('daily_return', 0.0) for t in traces])
    if len(returns) < 2: return 0.0
    rf_daily = risk_free_rate / 252.0
    excess_returns = returns - rf_daily
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    if std_excess < 1e-9: return 0.0
    return (mean_excess / std_excess) * np.sqrt(252)

def calculate_sortino(traces, risk_free_rate=None):
    if risk_free_rate is None:
        risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
    returns = np.array([t.get('daily_return', 0.0) for t in traces])
    if len(returns) < 2: return 0.0
    rf_daily = risk_free_rate / 252.0
    excess_returns = returns - rf_daily
    mean_excess = np.mean(excess_returns)
    
    # Sortino uses downside deviation of excess returns below 0
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return 0.0
        
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    if downside_deviation < 1e-9: return 0.0     
    return (mean_excess / downside_deviation) * np.sqrt(252)


class PairTradingEnv(gym.Env):

    def __init__(self, series_x: pd.Series, series_y: pd.Series, 
                 lookback: int = 30,
                 initial_capital: float = 10000,
                 position_scale: int = 100,
                 transaction_cost_rate: float = 0.0005,
                 test_mode: bool = False):
        
        super().__init__()
        
        # Align series
        self.data = pd.concat([series_x, series_y], axis=1).dropna()
        self.lookback = lookback
        self.test_mode = test_mode
        self.initial_capital = initial_capital
        self.position_scale = position_scale
        self.transaction_cost_rate = transaction_cost_rate
        
        # Action: 3 discrete actions (short, flat, long)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 14 features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        # Precompute spread and features
        self._precompute_features()
        
        self.reset()

    def _compute_rsi(self, series, period=14):
        """Helper to calculate RSI of the spread"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def _precompute_features(self):
        """Compute spread and advanced features"""
        x = self.data.iloc[:, 0]
        y = self.data.iloc[:, 1]
        
        # Raw spread
        self.spread = x - y
        
        # 1. Z-scores (Mean Reversion Signals)
        self.zscore_short = (
            (self.spread - self.spread.rolling(self.lookback).mean()) / 
            (self.spread.rolling(self.lookback).std() + 1e-8)
        )
        
        self.zscore_long = (
            (self.spread - self.spread.rolling(self.lookback * 2).mean()) / 
            (self.spread.rolling(self.lookback * 2).std() + 1e-8)
        )
        
        # 2. Volatility Features (Risk Detection)
        self.vol_short = self.spread.rolling(self.lookback).std()
        self.vol_long = self.spread.rolling(self.lookback * 3).std()
        
        # Volatility Ratio
        self.vol_ratio = self.vol_short / (self.vol_long + 1e-8)
        
        # 3. Momentum Features (Trend Detection)
        self.rsi = self._compute_rsi(self.spread, period=14)
        
        # Convert to numpy and fill NaNs
        self.spread_np = np.nan_to_num(self.spread.to_numpy(), nan=0.0)
        self.zscore_short_np = np.nan_to_num(self.zscore_short.to_numpy(), nan=0.0)
        self.zscore_long_np = np.nan_to_num(self.zscore_long.to_numpy(), nan=0.0)
        self.vol_np = np.nan_to_num(self.vol_short.to_numpy(), nan=1.0)
        self.vol_ratio_np = np.nan_to_num(self.vol_ratio.to_numpy(), nan=1.0)
        self.rsi_np = np.nan_to_num(self.rsi.to_numpy(), nan=50.0)
        
        # Store prices for logging
        self.price_x_np = x.to_numpy()
        self.price_y_np = y.to_numpy()

    def _get_observation(self, idx: int) -> np.ndarray:
        """Build NORMALIZED observation vector"""
        if idx < 0 or idx >= len(self.spread_np):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        norm_unrealized = self.unrealized_pnl / self.initial_capital
        norm_realized = self.realized_pnl / self.initial_capital
        
        obs = np.array([
            self.zscore_short_np[idx],
            self.zscore_long_np[idx],
            self.vol_np[idx],
            self.spread_np[idx],
            
            # NEW FEATURES
            self.rsi_np[idx] / 100.0,
            self.vol_ratio_np[idx],
            
            float(self.position / self.position_scale),  
            float(self.entry_spread) if self.position != 0 else 0.0,
            
            # NORMALIZED FINANCIALS
            float(norm_unrealized),
            float(norm_realized),
            
            float(self.cash / self.initial_capital - 1),  
            float(self.portfolio_value / self.initial_capital - 1),  
            
            float(self.days_in_position) / 252.0,
            float(self.num_trades) / 100.0,
        ], dtype=np.float32)
        
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.idx = self.lookback if not self.test_mode else 0
        self.position = 0
        self.entry_spread = 0.0 
        self.days_in_position = 0
        
        # Financial tracking
        self.cash = self.initial_capital
        self.realized_pnl = 0.0 
        self.unrealized_pnl = 0.0 
        self.portfolio_value = self.initial_capital
        
        # Performance tracking
        self.peak_value = self.initial_capital
        self.num_trades = 0
        self.trade_history = []
        
        # For return calculation
        self.prev_portfolio_value = self.initial_capital
        
        return self._get_observation(self.idx), {}

    def step(self, action: int):
        """
        Execute one trading step with IMPROVED Reward Calculation.
        """
        current_idx = self.idx
        
        # 1. Determine if this is the last available step
        is_last_step = (current_idx >= len(self.spread_np) - 1)
        
        # 2. Determine Action
        if is_last_step:
            target_position = 0 # FORCE EXIT
        else:
            base_position = int(action) - 1
            target_position = base_position * self.position_scale

        # 3. Setup Data
        current_spread = float(self.spread_np[current_idx])
        current_zscore = float(self.zscore_short_np[current_idx])
        
        if is_last_step:
            next_spread = current_spread 
            next_idx = current_idx 
        else:
            next_idx = current_idx + 1
            next_spread = float(self.spread_np[next_idx])
            
        # 4. Execute Trade & Update Financials
        position_change = target_position - self.position
        trade_occurred = (position_change != 0)
        
        realized_pnl_this_step = 0.0
        transaction_costs = 0.0
        
        if trade_occurred:
            # Calculate Realized P&L
            if self.position != 0:
                spread_change = current_spread - self.entry_spread
                
                # Check if we are closing or flipping
                if target_position == 0 or np.sign(target_position) != np.sign(self.position):
                    closed_size = abs(self.position)
                else:
                    closed_size = abs(position_change)
                    
                # Standard PnL Calculation
                realized_pnl_this_step = (self.position / abs(self.position)) * closed_size * spread_change

            # Transaction Costs
            trade_size = abs(position_change)
            notional = trade_size * abs(current_spread)
            transaction_costs = notional * self.transaction_cost_rate
            self.num_trades += 1
            
            # Reset/Update Entry Price
            if target_position != 0 and np.sign(target_position) != np.sign(self.position):
                # Flipping or Opening New
                self.entry_spread = current_spread
                self.days_in_position = 0
            elif target_position == 0:
                # Flat
                self.entry_spread = 0.0
                self.days_in_position = 0
                
            # Log history
            if self.position != 0:
                  self.trade_history.append({
                    'entry_spread': self.entry_spread,
                    'exit_spread': current_spread,
                    'position': self.position,
                    'pnl': realized_pnl_this_step,
                    'holding_days': self.days_in_position,
                    'forced_close': is_last_step
                })
        else:
            # Holding existing position
            self.days_in_position += 1
            
        # Update State
        self.position = target_position
        self.realized_pnl += realized_pnl_this_step - transaction_costs
        self.cash = self.initial_capital + self.realized_pnl
        
        if self.position != 0:
            self.unrealized_pnl = self.position * (next_spread - self.entry_spread)
        else:
            self.unrealized_pnl = 0.0
            
        self.portfolio_value = self.cash + self.unrealized_pnl
        
        # 5. Returns
        if not hasattr(self, 'prev_portfolio_value'):
            self.prev_portfolio_value = self.initial_capital

        # Calculate log returns for better stability, or standard % returns
        daily_return = (self.portfolio_value - self.prev_portfolio_value) / max(self.prev_portfolio_value, 1e-8)
        self.prev_portfolio_value = self.portfolio_value

        # 6. Metrics
        prev_peak = self.peak_value
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-8)
        
        # ==============================================================================
        # 7. IMPROVED REWARD CALCULATION
        # ==============================================================================
        
        reward = 0.0

        # A. PnL Reward (Risk-Adjusted)
        # -------------------------------------------------------------------------
        # Instead of raw return, we penalize volatility implicitly via the Sortino-style logic.
        # If return is positive, full reward. If negative, heavier penalty.
        if daily_return > 0:
            reward += daily_return * 100.0  # Scale up small % returns
        else:
            reward += daily_return * 120.0  # 1.2x penalty for losses (Loss Aversion)

        # B. Realized PnL Bonus (The "Cookie")
        # -------------------------------------------------------------------------
        # We give a significant one-time bonus for locking in a profit.
        # This encourages the agent to actually CLOSE trades rather than hold forever.
        if realized_pnl_this_step > 0:
            # Reward is proportional to the % gain on capital
            pnl_pct = realized_pnl_this_step / self.initial_capital
            reward += pnl_pct * 500.0 # Big spike for banking profit
        
        # C. Drawdown Delta Penalty (The "Stop Loss")
        # -------------------------------------------------------------------------
        # CRITICAL CHANGE: Only penalize if drawdown INCREASES.
        # This avoids the "Death Spiral" where an agent in drawdown gets punished 
        # even if it makes a good trade that recovers 1% of the loss.
        # We check if peak_value didn't update, meaning we are below high water mark.
        if self.portfolio_value < prev_peak:
            # Calculate how much deeper the drawdown got this step
            # If we recovered (daily_return > 0), this adds nothing or is positive.
            # We want to punish negative returns specifically when already in drawdown.
            if daily_return < 0:
                reward -= abs(daily_return) * 50.0 # Extra penalty for losing money while down

        # D. Holding Cost (Time Value of Money)
        # -------------------------------------------------------------------------
        # Non-linear penalty. Holding for 5 days is fine. Holding for 50 is bad.
        # Caps at a certain point to prevent explosion.
        if self.position != 0:
            holding_penalty = min(self.days_in_position, 50) * 0.005
            reward -= holding_penalty

        # E. Z-Score Alignment (Guidance / Shaping)
        # -------------------------------------------------------------------------
        # Only apply this if the agent is NOT in a trade (to guide entry) 
        # or if the position opposes the Z-score logic.
        norm_pos = self.position / self.position_scale
        
        # "Anti-alignment": If Z > 1 (Expensive) and we are Long (Pos > 0) -> Penalize!
        if (current_zscore > 1.0 and norm_pos > 0) or (current_zscore < -1.0 and norm_pos < 0):
            reward -= 0.1 # Small constant penalty for fighting the mean reversion
            
        # "Pro-alignment": If Z > 1 and we Short, or Z < -1 and we Long -> Small drip feed
        if (current_zscore > 1.0 and norm_pos < 0) or (current_zscore < -1.0 and norm_pos > 0):
            reward += 0.05 

        # Clip reward to maintain stability for PPO (prevents gradients exploding)
        reward = np.clip(reward, -10.0, 10.0)
        
        # 8. Index
        if not is_last_step:
            self.idx = next_idx
        
        # 9. Obs
        obs = self._get_observation(self.idx)
        
        # 10. Info
        info = {
            'portfolio_value': float(self.portfolio_value),
            'cash': float(self.cash),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'realized_pnl_this_step': float(realized_pnl_this_step),
            'transaction_costs': float(transaction_costs),
            'position': int(self.position),
            'entry_spread': float(self.entry_spread),
            'current_spread': float(current_spread),
            'z_score': float(current_zscore),  # Logged z_score
            'days_in_position': int(self.days_in_position),
            'daily_return': float(daily_return),
            'drawdown': float(drawdown),
            'num_trades': int(self.num_trades),
            'trade_occurred': bool(trade_occurred),
            'cum_return': float(self.portfolio_value / self.initial_capital - 1),
            'forced_close': is_last_step and trade_occurred,
            'price_x': float(self.price_x_np[current_idx]),
            'price_y': float(self.price_y_np[current_idx])
        }
        
        terminated = is_last_step
        
        return obs, float(reward), terminated, False, info
        
@dataclass
class OperatorAgent:
    
    logger: Optional[JSONLogger] = None
    storage_dir: str = "models/"

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        self.active = True
        self.transaction_cost = CONFIG.get("transaction_cost", 0.0005)
        self.current_step = 0
        self.traces_buffer = []
        self.max_buffer_size = 1000

    def get_current_step(self):
        return self.current_step

    def get_traces_since_step(self, start_step):
        return [t for t in self.traces_buffer if t.get('step', 0) >= start_step]

    def add_trace(self, trace):
        self.traces_buffer.append(trace)
        if len(self.traces_buffer) > self.max_buffer_size:
            self.traces_buffer = self.traces_buffer[-self.max_buffer_size:]

    def clear_traces_before_step(self, step):
        self.traces_buffer = [t for t in self.traces_buffer if t.get('step', 0) >= step]

    def apply_command(self, command):
        cmd_type = command.get("command")
        if cmd_type == "pause":
            self.active = False
            if self.logger:
                self.logger.log("operator", "paused", {})
        elif cmd_type == "resume":
            self.active = True
            if self.logger:
                self.logger.log("operator", "resumed", {})

    def load_model(self, model_path):
        return RecurrentPPO.load(model_path)

    def train_on_pair(self, prices: pd.DataFrame, x: str, y: str, 
                      lookback: int = None, timesteps: int = None, 
                      shock_prob: float = None, shock_scale: float = None,
                      use_curriculum: bool = False):

        # Get seed from CONFIG
        seed = CONFIG.get("random_seed", 42)
                            
        if not self.active:
            return None

        if lookback is None:
            lookback = CONFIG.get("rl_lookback", 30) 
            
        if timesteps is None:
            timesteps = CONFIG.get("rl_timesteps", 500000)

        series_x = prices[x]
        series_y = prices[y]

        print(f"\n{'='*70}")
        print(f"Training pair: {x} - {y} (LSTM POLICY)")
        print(f"  Data length: {len(series_x)} days")
        print(f"  Timesteps: {timesteps:,}")
        print(f"  Time Window (Lookback): {lookback} (Paper optimal: 30)")
        print(f"  LSTM Hidden Size: 512 (Paper optimal)")
        print(f"{'='*70}")

        print("\n🚀 Training with Recurrent PPO (LSTM)...")
        env = PairTradingEnv(
            series_x, series_y, lookback, position_scale=100, 
            transaction_cost_rate=0.0005, test_mode=False
        )
        
        # Seed the environment
        env.reset(seed=seed)

        policy_kwargs = dict(
            lstm_hidden_size=512,
            n_lstm_layers=1
        )

        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=0.001,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.04,
            verbose=1,
            device="auto",
            seed=seed,
            policy_kwargs=policy_kwargs 
        )

        model.learn(total_timesteps=timesteps)

        # Save model
        model_path = os.path.join(self.storage_dir, f"operator_model_{x}_{y}.zip")
        model.save(model_path)
        print(f"\n✅ Model saved to {model_path}")

        # Evaluate on training data
        print("\n📊 Evaluating on training data...")
        env_eval = PairTradingEnv(
              series_x, series_y, lookback, position_scale=100,
              transaction_cost_rate = 0.0005, test_mode=False
        )
        
        obs, _ = env_eval.reset()
        done = False
        daily_returns = []
        positions = []

        # Initialize LSTM states
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts,
                deterministic=True
            )
            obs, reward, done, _, info = env_eval.step(action)
            episode_starts = np.array([done])
            
            daily_returns.append(info.get('daily_return', 0))
            positions.append(info.get('position', 0))

        # Calculate metrics
        rets = np.array(daily_returns)
        rf_daily = CONFIG.get("risk_free_rate", 0.04) / 252
        excess_rets = rets - rf_daily

        sharpe = 0.0
        if len(excess_rets) > 1 and np.std(excess_rets, ddof=1) > 1e-8:
            sharpe = np.mean(excess_rets) / np.std(excess_rets, ddof=1) * np.sqrt(252)
        
        downside = excess_rets[excess_rets < 0]
        sortino = 0.0
        if len(downside) > 1 and np.std(downside, ddof=1) > 1e-8:
            sortino = np.mean(excess_rets) / np.std(downside, ddof=1) * np.sqrt(252)

        final_return = (env_eval.portfolio_value / env_eval.initial_capital - 1) * 100

        # Position analysis
        unique_positions = np.unique(positions)
        print(f"\n📈 Training Results:")
        print(f"  Final Return: {final_return:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.3f}")
        print(f"  Sortino Ratio: {sortino:.3f}")
        print(f"  Positions used: {unique_positions}")

        for pos in unique_positions:
            count = np.sum(np.array(positions) == pos)
            pct = count / len(positions) * 100
            print(f"    Position {int(pos)}: {pct:.1f}% of time")

        trace = {
            "pair": f"{x}-{y}", # Update pair format for consistency with traces
            "cum_return": final_return,
            "max_drawdown": (env_eval.peak_value - env_eval.portfolio_value) / env_eval.peak_value,
            "sharpe": sharpe,
            "sortino": sortino,
            "model_path": model_path,
            "positions_used": unique_positions.tolist()
        }

        if self.logger:
            self.logger.log("operator", "pair_trained", trace)

        return trace

    def save_detailed_trace(self, trace: Dict[str, Any], filepath: str = "traces/operator_detailed.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "a") as f:
            f.write(json.dumps(trace, default=str) + "\n")


def train_operator_on_pairs(operator: OperatorAgent, prices: pd.DataFrame, 
                        pairs: list, max_workers: int = None):

    if max_workers is None:
        max_workers = CONFIG.get("max_workers", 2)

    all_traces = []

    def train(pair):
        x, y = pair
        return operator.train_on_pair(prices, x, y)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train, pair) for pair in pairs]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Operator Training"):
            result = f.result()
            if result:
                all_traces.append(result)

    save_path = os.path.join(operator.storage_dir, "all_operator_traces.json")
    with open(save_path, "w") as f:
        json.dump(all_traces, f, indent=2, default=str)

    if operator.logger:
        operator.logger.log("operator", "batch_training_complete", {"n_pairs": len(all_traces)})
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for trace in all_traces:
        # trace['pair'] is tuple in training, string in holdout. Normalize for printing.
        pair_str = f"{trace['pair'][0]}-{trace['pair'][1]}" if isinstance(trace['pair'], tuple) else trace['pair']
        print(f"{pair_str:<15}: "
              f"Return={trace['cum_return']:.2f}%, Sharpe={trace['sharpe']:.2f}")
    print("="*70)
    
    return all_traces


def run_operator_holdout(operator, holdout_prices, pairs, supervisor, warmup_steps=90):
    """
    Run holdout testing with supervisor monitoring.
    Uses 'warmup_steps' at the start of holdout_prices to initialize LSTM
    and internal indicators without recording PnL.
    
    If stopped by supervisor, metrics are calculated on the data generated up to that point.
    """
    
    # Check supervisor config
    if "supervisor_rules" in CONFIG and "holdout" in CONFIG["supervisor_rules"]:
        check_interval = CONFIG["supervisor_rules"]["holdout"].get("check_interval", 20)
    else:
        check_interval = 20
        
    operator.traces_buffer = []
    operator.current_step = 0

    global_step = 0
    all_traces = []
    skipped_pairs = []
    
    # Storage for final summary
    pair_summaries = []

    # Ensure lookback matches training
    lookback = CONFIG.get("rl_lookback", 30)
    
    for pair in pairs:
        print(f"\n{'='*70}")
        print(f"Testing pair: {pair[0]} - {pair[1]}")
        print(f"{'='*70}")

        # 1. Data Validation
        if pair[0] not in holdout_prices.columns or pair[1] not in holdout_prices.columns:
            print(f"⚠️ Warning: Tickers {pair} not found in holdout data - skipping")
            skipped_pairs.append({"pair": f"{pair[0]}-{pair[1]}", "reason": "Data not found", "severity": "skip"})
            continue

        series_x = holdout_prices[pair[0]].dropna()
        series_y = holdout_prices[pair[1]].dropna()
        aligned = pd.concat([series_x, series_y], axis=1).dropna()

        # Ensure we have enough data for Lookback + Warmup + At least 1 trade step
        if len(aligned) < lookback + warmup_steps + 1:
            print(f"⚠️ Insufficient data ({len(aligned)} steps total). Needs {lookback + warmup_steps + 1} - skipping")
            skipped_pairs.append({"pair": f"{pair[0]}-{pair[1]}", "reason": "Insufficient data", "severity": "skip"})
            continue

        # 2. Model Loading
        model_path = os.path.join(operator.storage_dir, f"operator_model_{pair[0]}_{pair[1]}.zip")
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found - skipping")
            skipped_pairs.append({"pair": f"{pair[0]}-{pair[1]}", "reason": "Model not found", "severity": "skip"})
            continue

        model = operator.load_model(model_path)
        print(f"  ✓ Model loaded")

        # 3. Environment Setup (Test Mode)
        env = PairTradingEnv(
            series_x=aligned.iloc[:, 0], 
            series_y=aligned.iloc[:, 1], 
            lookback=lookback, 
            initial_capital=10000,
            transaction_cost_rate=0.0005, 
            test_mode=True
        )

        episode_traces = []
        local_step = 0
        obs, info = env.reset() 

        # ==============================================================================
        # WARM-UP PHASE
        # ==============================================================================
        print(f"  ⏳ Warming up model state on {warmup_steps} steps of history...")
        
        # Initialize LSTM states
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        warmup_completed = True
        
        # Run the model to update states, BUT ignore results/pnl
        for i in range(warmup_steps):
            if env.idx >= len(env.spread_np) - 1:
                print("  ⚠️ Data ended during warm-up. Skipping pair.")
                warmup_completed = False
                break 
                
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=True
            )
            
            obs, _, done, _, _ = env.step(action)
            episode_starts = np.array([done])
            
            if done:
                warmup_completed = False
                break
        
        if not warmup_completed:
            continue

        # ==============================================================================
        # FINANCIAL RESET (Prepare for Real Trading)
        # ==============================================================================
        env.cash = env.initial_capital
        env.portfolio_value = env.initial_capital
        env.realized_pnl = 0.0 
        env.unrealized_pnl = 0.0 
        env.num_trades = 0
        env.trade_history = []
        env.peak_value = env.initial_capital
        
        # Force Flat Position to ensure clean start
        if env.position != 0:
            env.position = 0
            env.entry_spread = 0.0
            env.days_in_position = 0

        print(f"  ✓ Warm-up complete. Financials reset. Trading starts at Index {env.idx}.")

        # ==============================================================================
        # MAIN TRADING LOOP
        # ==============================================================================
        terminated = False
        stop_triggered = False
        
        while not terminated:
            
            # Predict using the warmed-up lstm_states
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=True
            )
            
            obs, reward, terminated, _, info = env.step(action)
            episode_starts = np.array([terminated])

            # --- Trace Logging ---
            trace = {
                "pair": f"{pair[0]}-{pair[1]}",
                "step": global_step,
                "local_step": local_step,
                "reward": float(reward),
                "portfolio_value": float(info.get("portfolio_value", 0.0)),
                "cum_return": float(info.get("cum_return", 0.0)),
                "position": float(info.get("position", 0)),
                "max_drawdown": float(info.get("drawdown", 0)),
                "cash": float(info.get("cash", 0.0)),
                "realized_pnl": float(info.get("realized_pnl", 0.0)),
                "unrealized_pnl": float(info.get("unrealized_pnl", 0.0)),
                "realized_pnl_this_step": float(info.get("realized_pnl_this_step", 0.0)),
                "transaction_costs": float(info.get("transaction_costs", 0.0)),
                "entry_spread": float(info.get("entry_spread", 0.0)),
                "current_spread": float(info.get("current_spread", 0.0)),
                "z_score": float(info.get("z_score", 0.0)), 
                "days_in_position": int(info.get("days_in_position", 0)),
                "daily_return": float(info.get("daily_return", 0.0)),
                "num_trades": int(info.get("num_trades", 0)),
                "trade_occurred": bool(info.get("trade_occurred", False)),
                "risk_exit": bool(info.get("risk_exit", False)),
                "price_x": float(info.get("price_x", 0.0)),
                "price_y": float(info.get("price_y", 0.0))
            }

            episode_traces.append(trace)
            all_traces.append(trace)
            operator.add_trace(trace)
            
            if hasattr(operator, 'save_detailed_trace'):
                operator.save_detailed_trace(trace)
                
            if operator.logger:
                operator.logger.log("operator", "holdout_step", trace)

            # --- Supervisor Monitoring ---
            if local_step > 0 and local_step % check_interval == 0:
                decision = supervisor.check_operator_performance(
                    episode_traces, 
                    pair, 
                    phase="holdout"
                )
                
                if decision["action"] == "stop":
                    severity = decision.get("severity", "critical")
                    print(f"\n⛔ SUPERVISOR INTERVENTION [{severity.upper()}]: Stopping pair early")
                    print(f"    Reason: {decision['reason']}")
                    
                    # Store metrics reported by the supervisor for the skipped pair
                    skip_info = {
                        "pair": f"{pair[0]}-{pair[1]}",
                        "reason": decision['reason'],
                        "severity": severity,
                        "step_stopped": global_step,
                        "metrics": decision['metrics']
                    }
                    skipped_pairs.append(skip_info)
                    
                    if operator.logger:
                        operator.logger.log("supervisor", "intervention", skip_info)
                    
                    stop_triggered = True
                    break # Break out of the WHILE loop, proceed to metrics calculation below
                
                elif decision["action"] == "adjust":
                    print(f"\n⚠️  SUPERVISOR WARNING: {decision['reason']}")

            local_step += 1
            global_step += 1
            operator.current_step = global_step

        # End of Pair Loop - Reporting Phase
        if stop_triggered:
            print(f"⏭️  Pair stopped early at step {local_step} due to Supervisor Intervention.")
        else:
            print(f"  ✓ Complete: {len(episode_traces)} steps")
            
        # ==============================================================================
        # DETAILED METRICS REPORTING
        # ==============================================================================
        if len(episode_traces) > 0:
            # 1. Calculate Metrics (works for partial or full episodes)
            sharpe = calculate_sharpe(episode_traces)
            sortino = calculate_sortino(episode_traces)
            final_return = episode_traces[-1]['cum_return'] * 100
            
            # Recalculate max drawdown based on the actual episode traces
            returns_series = pd.Series([t['daily_return'] for t in episode_traces])
            equity_curve = (1 + returns_series).cumprod()
            running_max = equity_curve.cummax()
            dd_series = (running_max - equity_curve) / running_max
            max_dd = dd_series.max() # Max drawdown is the largest value in the DD series

            # 2. Position Analysis
            positions = [t['position'] for t in episode_traces]
            
            # 3. Print Results
            print(f"\n📊 Holdout Results for {pair[0]}-{pair[1]}:")
            print(f"  Final Return: {final_return:.2f}%")
            print(f"  Max Drawdown: {max_dd:.2%}")
            print(f"  Sharpe Ratio: {sharpe:.3f}")
            print(f"  Sortino Ratio: {sortino:.3f}")
            
            # Win Rate (Bonus Metric)
            # Filter steps where a PnL was actually realized (trade closed or flipped)
            pnl_events = [t['realized_pnl_this_step'] for t in episode_traces if abs(t['realized_pnl_this_step']) > 0]
            if len(pnl_events) > 0:
                # Need to use net PnL for true win rate
                net_pnl_events = [(t['realized_pnl_this_step'] - t['transaction_costs']) for t in episode_traces if abs(t['realized_pnl_this_step']) > 0]
                wins = len([p for p in net_pnl_events if p > 0])
                win_rate = (wins / len(net_pnl_events)) * 100
                print(f"  Win Rate: {win_rate:.1f}% ({len(net_pnl_events)} realized trades)")
            else:
                win_rate = 0.0
                print(f"  Win Rate: N/A (0 trades)")

            # Position Distribution
            print(f"  Position Distribution:")
            for pos in [-100, 0, 100]:
                count = np.sum(np.array(positions) == pos)
                pct = count / len(positions) * 100
                # Map value to readable name
                name = "Flat"
                if pos > 0: name = "Long"
                elif pos < 0: name = "Short"
                print(f"    {name} ({int(pos)}): {pct:.1f}% of time")

            # Store for final summary - Includes stopped pairs
            pair_summaries.append({
                "pair": f"{pair[0]}-{pair[1]}",
                "return": final_return,
                "sharpe": sharpe,
                "drawdown": max_dd,
                "trades": env.num_trades,
                "win_rate": win_rate if len(pnl_events) > 0 else 0.0,
                "status": "STOPPED" if stop_triggered else "COMPLETE"
            })

            # Logging
            if operator.logger:
                final_pnl = episode_traces[-1].get('realized_pnl', 0)
                operator.logger.log("operator", "episode_complete", {
                    "pair": f"{pair[0]}-{pair[1]}",
                    "total_steps": len(episode_traces),
                    "final_cum_return": final_return,
                    "total_pnl": final_pnl,
                    "sharpe": sharpe,
                    "sortino": sortino,
                    "was_stopped": stop_triggered
                })
            
    print("\n" + "="*80)
    print("HOLDOUT TESTING COMPLETE: SUMMARY")
    print("="*80)
    print(f"{'Pair':<15} | {'Status':<9} | {'Return':<8} | {'Sharpe':<6} | {'Max DD':<8} | {'Win Rate':<8}")
    print("-" * 80)
    
    total_ret = 0
    for s in pair_summaries:
        status_icon = "🛑" if s['status'] == "STOPPED" else "✅"
        print(f"{s['pair']:<15} | {status_icon} {s['status'][:3]}.. | {s['return']:>7.2f}% | {s['sharpe']:>6.2f} | {s['drawdown']:>7.1%} | {s['win_rate']:>7.1f}%")
        total_ret += s['return']
        
    avg_ret = total_ret / len(pair_summaries) if pair_summaries else 0.0
    print("-" * 80)
    print(f"Average Return: {avg_ret:.2f}% across {len(pair_summaries)} pairs")
    print("="*80)
    print(f"Total steps simulated: {global_step}")
    
    return all_traces, skipped_pairs

def calculate_sharpe(traces, risk_free_rate=None):
    if risk_free_rate is None:
        # Use try/except or config check to get risk_free_rate safely
        try:
            risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
        except NameError:
            risk_free_rate = 0.04
            
    returns = np.array([t.get('daily_return', 0.0) for t in traces])
    if len(returns) < 2: return 0.0
    rf_daily = risk_free_rate / 252.0
    excess_returns = returns - rf_daily
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    if std_excess < 1e-9: return 0.0
    return (mean_excess / std_excess) * np.sqrt(252)

def calculate_sortino(traces, risk_free_rate=None):
    if risk_free_rate is None:
        # Use try/except or config check to get risk_free_rate safely
        try:
            risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
        except NameError:
            risk_free_rate = 0.04
            
    returns = np.array([t.get('daily_return', 0.0) for t in traces])
    if len(returns) < 2: return 0.0
    rf_daily = risk_free_rate / 252.0
    excess_returns = returns - rf_daily
    mean_excess = np.mean(excess_returns)
    
    # Sortino uses downside deviation of excess returns below 0
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return 0.0
        
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    if downside_deviation < 1e-9: return 0.0     
    return (mean_excess / downside_deviation) * np.sqrt(252)
