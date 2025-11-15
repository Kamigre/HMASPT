"""
Selector Agent for identifying cointegrated stock pairs using Temporal Graph Neural Networks.
Requires: torch, torch-geometric (optional)
"""

import os
import json
import datetime
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. SelectorAgent will have limited functionality.")

try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Statistical validation will be limited.")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import CONFIG
from utils import half_life as compute_half_life, compute_spread
from agents.message_bus import MessageBus


if TORCH_AVAILABLE:
    class MemoryTGNN(nn.Module):
        """
        Temporal Graph Neural Network (TGNN) with memory.
        
        Purpose:
        - Learn dynamic node embeddings that evolve over time
        - Capture both structural relationships (edges) and temporal evolution
        - Provide embeddings for pair selection scoring
        
        Components:
        1. Input projection: stabilizes features
        2. GRUCell: updates hidden state ("memory") for each node
        3. Learnable gating to blend old and new memory
        4. Residual GAT layers with LayerNorm and dropout
        5. L2-normalized embeddings for scoring pairs
        6. Decoder: scores pairs of nodes using concatenated embeddings
        """
        
        def __init__(self, in_channels, hidden_channels=64, num_heads=1, num_layers=1, 
                     dropout=0.1, blend_factor=0.3, device=None):
            super().__init__()
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.hidden_channels = hidden_channels
            self.num_heads = num_heads
            self.num_layers = max(1, num_layers)
            self.dropout_p = dropout
            self.blend_factor = blend_factor

            self.input_proj = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.LayerNorm(hidden_channels)
            )

            self.msg_gru = nn.GRUCell(hidden_channels, hidden_channels)

            self.update_gate = nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.Sigmoid()
            )

            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(self.num_layers):
                self.convs.append(GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads))
                self.norms.append(nn.LayerNorm(hidden_channels))

            self.dropout = nn.Dropout(p=self.dropout_p)

            self.proj_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.LayerNorm(hidden_channels)
            )

            self.decoder = nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(hidden_channels, 1)
            )

            self.node_memory = None
            self._reset_parameters()
            self.to(self.device)

        def _reset_parameters(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, x, edge_index, pair_index=None, memory=None, timestamps=None):
            """
            Forward pass for TGNN.
            
            Args:
                x: Node features (num_nodes x in_channels)
                edge_index: Graph edges (2 x num_edges)
                pair_index: Optional pairs to score (2 x num_pairs)
                memory: Optional previous node memory
                timestamps: Optional temporal encoding
                
            Returns:
                - Embeddings for all nodes and updated memory, or
                - Pair scores if pair_index is provided
            """
            x = x.to(self.device)
            N = x.size(0)
            edge_index = edge_index.to(self.device) if edge_index is not None else None

            if memory is None:
                memory = torch.zeros(N, self.hidden_channels, device=self.device)
            else:
                memory = memory.to(self.device)

            x_proj = self.input_proj(x)

            if timestamps is not None:
                tproj = torch.tanh(nn.Linear(timestamps.shape[-1], self.hidden_channels).to(self.device)(timestamps.to(self.device)))
                x_proj = x_proj + tproj

            new_mem = self.msg_gru(x_proj, memory)

            gate_in = torch.cat([memory, new_mem], dim=1)
            gate = self.update_gate(gate_in)
            memory = (1.0 - gate) * memory + gate * new_mem

            h = memory
            for conv, norm in zip(self.convs, self.norms):
                if edge_index is not None and edge_index.numel() > 0:
                    h_conv = conv(h, edge_index)
                else:
                    h_conv = h
                h = norm(h + self.dropout(F.relu(h_conv)))

            self.node_memory = memory

            z = self.proj_head(h)
            z = F.normalize(z, p=2, dim=1)

            if pair_index is not None:
                src_idx = pair_index[0].long().to(self.device)
                dst_idx = pair_index[1].long().to(self.device)
                pair_embed = torch.cat([z[src_idx], z[dst_idx]], dim=1)
                logits = self.decoder(pair_embed).view(-1)
                return logits

            return z, memory

        def model_memory(self):
            return self.node_memory


@dataclass
class SelectorAgent:
    """
    SelectorAgent with temporal holdout for simulating future performance.
    
    Key Features:
    - Monthly temporal graphs
    - Memory-based TGNN for embeddings
    - Temporal holdout: train on first 4 years, score pairs on last year
    - GPU support if available
    
    Improvements:
    - Integrates with MessageBus for supervisor commands
    - Traces important events to JSONL for audit
    - Periodically checks message bus during long operations
    """
    
    df: pd.DataFrame
    fundamentals: Optional[pd.DataFrame] = None
    model: Any = None
    scaler: Optional[MinMaxScaler] = None
    edge_index: Optional[Any] = None
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    message_bus: Optional[MessageBus] = None
    trace_path: str = "traces/selector_trace.jsonl"
    corr_threshold: float = 0.8
    holdout_years: int = 1
    node_features: Optional[pd.DataFrame] = None
    temporal_graphs: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    val_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    test_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    holdout_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
        if self.message_bus is None:
            self.message_bus = MessageBus()
        self._log_event("init", {"device": self.device, "corr_threshold": self.corr_threshold})

    def _log_event(self, event: str, details: Dict[str, Any]):
        """Append a structured event to the JSONL trace file and publish to the bus."""
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": "selector",
            "event": event,
            "details": details,
        }
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        if self.message_bus:
            try:
                self.message_bus.publish(entry)
            except Exception:
                pass

    def _check_for_commands(self):
        """Non-blocking check for commands addressed to 'selector' and apply them."""
        if self.message_bus is None:
            return
        while True:
            cmd = self.message_bus.get_command_for("selector", timeout=0.0)
            if cmd is None:
                break
            try:
                self._apply_command(cmd)
                self._log_event("command_applied", {"command": cmd})
            except Exception as e:
                self._log_event("command_failed", {"command": cmd, "error": str(e)})

    def _apply_command(self, cmd: Dict[str, Any]):
        """
        Supported runtime commands:
        - {"command": "adjust_threshold", "value": 0.8}
        - {"command": "set_holdout_years", "value": 2}
        - {"command": "rebuild_node_features", "windows": [5,15,30]}
        - {"command": "dump_state", "path": "state.json"}
        """
        c = cmd.get("command")
        if c == "adjust_threshold":
            self.corr_threshold = float(cmd.get("value", self.corr_threshold))
            self._log_event("threshold_changed", {"new_threshold": self.corr_threshold})
        elif c == "set_holdout_years":
            self.holdout_years = int(cmd.get("value", self.holdout_years))
            self._log_event("holdout_changed", {"holdout_years": self.holdout_years})
        elif c == "rebuild_node_features":
            windows = cmd.get("windows", [5, 15, 30])
            self.build_node_features(windows=windows)
            self._log_event("node_features_rebuilt", {"windows": windows})
        elif c == "dump_state":
            path = cmd.get("path", "traces/selector_state.json")
            state = {
                "corr_threshold": self.corr_threshold,
                "holdout_years": self.holdout_years,
                "num_temporal_graphs": len(self.temporal_graphs) if self.temporal_graphs else 0
            }
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(state, f, default=str, indent=2)
            self._log_event("state_dumped", {"path": path})
        else:
            self._log_event("unknown_command", {"command": cmd})

    def build_node_features(self, windows=[5, 15, 30]) -> pd.DataFrame:
        """
        Build rolling window features for each stock and encode categorical features.
        
        Rolling features capture temporal behavior:
        - mean: trend over window
        - std: volatility over window
        - cum_return: cumulative log returns
        
        One-hot encodes sectors and standardizes all numeric features.
        """
        df = self.df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

        for window in windows:
            df[f"mean_{window}"] = df.groupby("ticker")["adj_close"].transform(
                lambda x: x.rolling(window).mean()
            )
            df[f"std_{window}"] = df.groupby("ticker")["adj_close"].transform(
                lambda x: x.rolling(window).std()
            )
            df[f"cum_return_{window}"] = df.groupby("ticker")["adj_close"].transform(
                lambda x: np.log(x / x.shift(1)).rolling(window).sum()
            )

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df["eps_yoy_growth"] = df.get("eps_yoy_growth", 0.0).fillna(0.0)
        df["peg_adj"] = df.get("peg_adj", 0.0).fillna(0.0)

        all_sectors = sorted(self.df["sector"].unique())
        encoder = OneHotEncoder(sparse_output=False, dtype=int, categories=[all_sectors])
        sector_encoded = encoder.fit_transform(df[["sector"]])
        sector_df = pd.DataFrame(sector_encoded, columns=encoder.get_feature_names_out())
        df = pd.concat([df.reset_index(drop=True), sector_df.reset_index(drop=True)], axis=1)
        df.drop(columns=["sector"], inplace=True, errors="ignore")

        df.fillna(0.0, inplace=True)

        exclude_cols = ["date", "ticker"]
        numeric_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

        if self.scaler is None:
            self.scaler = MinMaxScaler()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        self.node_features = df
        self._log_event("node_features_built", {"n_rows": len(df), "windows": windows})
        return df

    def build_temporal_graphs(self, corr_threshold: float = None, holdout_years: int = None):
        """
        Build weekly temporal graphs based on correlations between tickers.
        Uses all data except the last "holdout_years" for training.
        """
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not installed. Cannot build temporal graphs.")
            return None, None, None

        if corr_threshold is None:
            corr_threshold = self.corr_threshold
        else:
            self.corr_threshold = corr_threshold

        if holdout_years is None:
            holdout_years = self.holdout_years
        else:
            self.holdout_years = holdout_years

        self.temporal_graphs = []
        df = self.node_features.copy()
        df["date"] = pd.to_datetime(df["date"])
        tickers = df["ticker"].unique().tolist()

        exclude_cols = ["date", "ticker", "close", "adj_factor", "split_factor", "div_amount", "volume", "sector"]
        feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

        last_date = df["date"].max()
        holdout_start = last_date - pd.DateOffset(years=holdout_years)
        mid_point = holdout_start + pd.DateOffset(months=6)
        val_start = holdout_start
        val_end = mid_point - pd.DateOffset(days=1)
        test_start = mid_point
        test_end = last_date

        train_df = df[df["date"] < val_start]
        val_df = df[(df["date"] >= val_start) & (df["date"] <= val_end)]
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

        weeks = sorted(train_df["date"].dt.to_period("W").unique())

        for i in range(len(weeks) - 1):
            self._check_for_commands()

            start_week = weeks[i]
            end_week = weeks[i + 1]

            mask = (train_df["date"].dt.to_period("W") >= start_week) & \
                  (train_df["date"].dt.to_period("W") <= end_week)
            interval = train_df.loc[mask]

            if interval.empty:
                continue

            weekly_features = interval.groupby("ticker")[feature_cols].mean().fillna(0.0)
            weekly_features = weekly_features.reindex(tickers, fill_value=0.0)

            corr_matrix = np.corrcoef(weekly_features.values)
            corr_matrix = np.nan_to_num(corr_matrix)

            edges = np.argwhere(np.abs(corr_matrix) >= corr_threshold)
            edges = edges[edges[:, 0] < edges[:, 1]]

            num_nodes = len(tickers)
            edges = edges[(edges[:, 0] < num_nodes) & (edges[:, 1] < num_nodes)]

            if len(edges) > 0:
                edge_index = torch.tensor(edges.T, dtype=torch.long, device=self.device)
                edge_attr = torch.tensor([[corr_matrix[i, j]] for i, j in edges], dtype=torch.float, device=self.device)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                edge_attr = torch.zeros((0, 1), dtype=torch.float, device=self.device)

            self.temporal_graphs.append({
                "start": str(start_week),
                "end": str(end_week),
                "edge_index": edge_index,
                "edge_attr": edge_attr
            })

        self.val_period = (val_start, val_end)
        self.test_period = (test_start, test_end)
        self.holdout_period = (val_start, test_end)

        self._log_event("temporal_graphs_built", {
            "n_graphs": len(self.temporal_graphs),
            "val_period": (str(self.val_period[0]), str(self.val_period[1])),
            "test_period": (str(self.test_period[0]), str(self.test_period[1])),
            "corr_threshold": corr_threshold
        })

        print(f"✅ Temporal graphs built: {len(self.temporal_graphs)}")
        print(f"   Validation: {self.val_period[0].date()} → {self.val_period[1].date()}")
        print(f"   Test: {self.test_period[0].date()} → {self.test_period[1].date()}")

        return self.temporal_graphs, val_df, test_df

    def score_all_pairs_holdout(self):
        """
        Score all ticker pairs using memory propagation.
        Ensures all nodes have embeddings even if not in edges.
        """
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not installed. Cannot score pairs.")
            return pd.DataFrame()

        if self.model is None:
            print("Warning: No model attached. Cannot score pairs.")
            return pd.DataFrame()

        tickers = self.node_features["ticker"].unique().tolist()
        num_nodes = len(tickers)

        x_full = torch.tensor(
            self.node_features.drop(columns=["date", "ticker"], errors="ignore").values,
            dtype=torch.float,
            device=self.device
        )

        self.model.eval()
        memory = None

        with torch.no_grad():
            for g in self.temporal_graphs:
                self._check_for_commands()

                out = self.model(x_full, g["edge_index"], memory=memory)

                if isinstance(out, tuple) and len(out) == 2:
                    _, memory = out
                else:
                    memory = out

                memory = memory.detach()

            h = memory

            src_idx, dst_idx = np.triu_indices(num_nodes, k=1)
            pair_embed = torch.cat([h[src_idx], h[dst_idx]], dim=1)
            scores = self.model.decoder(pair_embed)

        results = pd.DataFrame({
            "x": np.array(tickers)[src_idx],
            "y": np.array(tickers)[dst_idx],
            "score": scores.cpu().numpy().flatten()
        }).sort_values("score", ascending=False).reset_index(drop=True)

        topk = results.head(10).to_dict(orient="records")
        self._log_event("pairs_scored", {"n_pairs": len(results), "top_10": topk})

        return results

    def validate_pairs(self, df_pairs: pd.DataFrame, validation_window: Tuple[pd.Timestamp, pd.Timestamp], 
                      half_life_max: float = 60, min_crossings_per_year: int = 24):
        """
        Validate scored pairs in the holdout period.
        
        Checks:
        - Mean reversion (ADF test)
        - Half-life < half_life_max
        - Sufficient crossings per year
        """
        if not STATSMODELS_AVAILABLE:
            print("Warning: statsmodels not installed. Validation will be limited.")
            return pd.DataFrame()

        start, end = validation_window
        validated = []
        prices = self.df.pivot(index="date", columns="ticker", values="adj_close").sort_index()

        for _, row in df_pairs.iterrows():
            self._check_for_commands()

            x, y = row["x"], row["y"]
            if x not in prices.columns or y not in prices.columns:
                continue

            series_x = prices[x].loc[start:end].dropna()
            series_y = prices[y].loc[start:end].dropna()
            if min(len(series_x), len(series_y)) < 60:
                continue

            spread = compute_spread(series_x, series_y)
            if spread.empty:
                continue

            crossings = ((spread - spread.mean()).shift(1) * (spread - spread.mean()) < 0).sum()
            days = (series_x.index[-1] - series_x.index[0]).days
            crossings_per_year = crossings / (days / 252)
            if crossings_per_year < min_crossings_per_year:
                continue

            try:
                adf_stat, adf_p, *_ = adfuller(spread.dropna())
            except Exception:
                adf_p = 1.0

            hl = compute_half_life(spread.values)

            pass_criteria = (adf_p < 0.05) and (hl < half_life_max)

            validated.append({
                "x": x, "y": y, "score": row["score"],
                "adf_p": float(adf_p), "half_life": float(hl),
                "mean_crossings": int(crossings),
                "pass": pass_criteria
            })

        self._log_event("pairs_validated", {"n_validated": len(validated)})
        return pd.DataFrame(validated)
