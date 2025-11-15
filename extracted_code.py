
# ========== CODE CELL 0 ==========
# ==========================================
# Package Installation
# ==========================================
!pip install numpy pandas scikit-learn statsmodels scipy torch torchvision torchaudio \
faiss-cpu gymnasium stable-baselines3 langchain huggingface_hub torch_geometric

# ==========================================
# Standard Libraries
# ==========================================
import os
import sys
import json
import math
import time
import sqlite3
import subprocess
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Dict, Optional

# ==========================================
# Data Manipulation
# ==========================================
import numpy as np
import pandas as pd

# ==========================================
# Statistical Tools
# ==========================================
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from scipy.stats import zscore

# ==========================================
# Machine Learning Tools
# ==========================================
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

# ==========================================
# PyTorch & Graph Neural Networks
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import Data

# ==========================================
# FAISS
# ==========================================
import faiss

# ==========================================
# Reinforcement Learning
# ==========================================
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ==========================================
# LangChain & HuggingFace
# ==========================================
from langchain import LLMChain, PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==========================================
# Visualization
# ==========================================
import matplotlib.pyplot as plt


!pip install "shimmy>=2.0"
import os
import json
import time
import queue
import threading
import datetime
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.stattools import adfuller
from torch_geometric.nn import GATConv

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
import gym
from gym import spaces
from stable_baselines3 import PPO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import datetime
import time

# ========== CODE CELL 1 ==========
with open("/content/drive/MyDrive/Colab Notebooks/TFM/all_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

records = []
for ticker, content in data.items():
    sector = content.get("sector", None)
    for rec in content.get("data", []):
        rec["ticker"] = ticker
        rec["sector"] = sector
        # Reorder keys so 'date' appears first
        rec = {"date": rec["date"], **{k: v for k, v in rec.items() if k != "date"}}
        records.append(rec)

df = pd.json_normalize(records)

# Convert 'date' to datetime
df["date"] = pd.to_datetime(df["date"])

# Ensure column order explicitly (in case JSON order changes)
cols = ["date"] + [c for c in df.columns if c != "date"]
df = df[cols]

# ========== CODE CELL 2 ==========
# --------------------------
# Configuration and utilities
# --------------------------

CONFIG = {
    "cointegration_pvalue_threshold": 0.05,
    "half_life_min": 1,
    "transaction_cost": 0.0005,
    "rl_policy": "MlpPolicy",
    "windows": [60],
    "rl_timesteps": 50000,
    "half_life_max": 60
}


def half_life(spread: np.ndarray) -> float:

    """
    Compute half-life of mean reversion for a spread series following
    the approach: fit AR(1): ds_t = a + b * s_{t-1} + eps, half-life = -ln(2) / ln(b)
    If b >= 1 or the regression fails, return a large number.
    """

    spread = np.asarray(spread)
    spread = spread[~np.isnan(spread)]
    if len(spread) < 10:
        return np.inf
    y = spread[1:]
    x = spread[:-1]
    x = sm.add_constant(x)

    try:
        res = sm.OLS(y, x).fit()
        b = res.params[1]
        if b >= 1:
            return np.inf
        halflife = -math.log(2) / math.log(abs(b))
        if halflife < 0 or np.isinf(halflife) or np.isnan(halflife):
            return np.inf
        return halflife
    except Exception:
        return np.inf


def compute_spread(series_x: pd.Series, series_y: pd.Series) -> pd.Series:

    """
    Compute residual spread from OLS regression of y ~ x (Engle-Granger residual).
    """

    aligned = pd.concat([series_x, series_y], axis=1).dropna()
    if aligned.shape[0] < 10:
        return pd.Series(dtype=float)
    x = sm.add_constant(aligned.iloc[:, 0])
    try:
        res = sm.OLS(aligned.iloc[:, 1], x).fit()
        residuals = res.resid
        return residuals
    except Exception:
        return pd.Series(dtype=float)


def save_json(obj: Any, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x), indent=2)

# --------------------------
# Simple message bus for inter-agent communication
# --------------------------
class MessageBus:
    """
    Minimal thread-safe message bus.
    - publish(event): put an event into the global event queue
    - send_command(target, command): put a command into the target-specific queue
    - get_events(): non-blocking generator returning available events
    - get_command_for(target): non-blocking retrieval of one command for target (or None)
    """

    def __init__(self):
        self.global_queue = queue.Queue()
        self.command_queues: Dict[str, queue.Queue] = {}
        self.lock = threading.Lock()

    def publish(self, event: Dict[str, Any]):
        """Publish a generic event (informational) to the global queue."""
        self.global_queue.put(event)

    def send_command(self, target: str, command: Dict[str, Any]):
        """Send a command dict to a specific agent (creates queue lazily)."""
        with self.lock:
            if target not in self.command_queues:
                self.command_queues[target] = queue.Queue()
            self.command_queues[target].put(command)

        # also publish command as an event for traceability
        cmd_event = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": "message_bus",
            "type": "command_sent",
            "target": target,
            "command": command,
        }
        self.publish(cmd_event)

    def get_command_for(self, target: str, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """Try to get a command for target — non-blocking by default (timeout=0)."""
        with self.lock:
            q = self.command_queues.get(target)
        if q is None:
            return None
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_events(self) -> List[Dict[str, Any]]:
        """Return all events currently in the global queue (non-blocking)."""
        events = []
        while True:
            try:
                events.append(self.global_queue.get_nowait())
            except queue.Empty:
                break
        return events


# Lightweight SwarmOrchestrator (keeps same merging behaviour)
class SwarmOrchestrator:
    """
    Lightweight swarm system to merge decisions from multiple agents.
    Strategy 'consensus': keep all unique actions.
    """
    def __init__(self, agents: List[str] = ["rules", "llm"], strategy: str = "consensus"):
        self.agents = agents
        self.strategy = strategy

    def merge_actions(self, actions_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        merged = []
        for a in actions_list:
            key = (a.get("action"), a.get("reason", ""))
            if key not in seen:
                merged.append(a)
                seen.add(key)
        return merged

# Lightweight Graph recorder (compatible with your earlier Graph)
class Graph:
    def __init__(self, name: str):
        self.name = name
        self.nodes = []
        self.edges = []

    def add_node(self, node_name: str, node_type: str = "action"):
        self.nodes.append({"name": node_name, "type": node_type})

    def add_edge(self, from_node: str, to_node: str):
        self.edges.append({"from": from_node, "to": to_node})

    def export(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"nodes": self.nodes, "edges": self.edges}, f, indent=2)

# JSONL trace logger for Supervisor events (keeps format consistent)
class JSONLogger:
    def __init__(self, path: str = "traces/supervisor_trace.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def log(self, agent: str, event: str, details: Dict[str, Any]):
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": agent,
            "event": event,
            "details": details
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

# ========== CODE CELL 3 ==========
# NOTE: This cell provides an improved SelectorAgent with:
# - a simple thread-safe MessageBus for inter-agent communication,
# - JSONL tracing of events/commands,
# - runtime command handling so Supervisor can adjust parameters while the Selector runs.
# The comment and docstring formats from your original code are preserved.

# --------------------------
# Selector Agent
# --------------------------

@dataclass
class SelectorAgent:

    """
    SelectorAgent with temporal holdout for simulating future performance.

    Key Features:
    - Monthly temporal graphs.
    - Memory-based TGNN for embeddings.
    - Temporal holdout: train on first 4 years, score pairs on last year.
    - GPU support if available.

    Improvements included:
    - Integrates with a MessageBus so Supervisor can send runtime commands.
    - Traces important events and commands to a JSONL trace file for audit.
    - Periodically checks the message bus during long-running operations (training / scoring).
    """

    df: pd.DataFrame  # Historical stock data with columns: date, ticker, adj_close, sector, etc.
    fundamentals: Optional[pd.DataFrame] = None  # Optional financial features like EPS, PEG
    model: Any = None  # Memory-based TGNN model instance
    scaler: Optional[MinMaxScaler] = None  # Optional feature scaler
    edge_index: Optional[torch.Tensor] = None  # Tensor storing graph edges
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

    # New instrumentation/runtime fields
    message_bus: Optional[MessageBus] = None
    trace_path: str = "traces/selector_trace.jsonl"
    corr_threshold: float = 0.8  # default correlation threshold (can be adjusted at runtime)
    holdout_years: int = 1
    node_features: Optional[pd.DataFrame] = None
    temporal_graphs: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    val_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    test_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    holdout_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
        # default message bus if none provided (shared instance should be set by orchestrator)
        if self.message_bus is None:
            self.message_bus = MessageBus()
        # quick trace of initialization
        self._log_event("init", {"device": self.device, "corr_threshold": self.corr_threshold})

    # --------------------------
    # Internal tracing utilities
    # --------------------------
    def _log_event(self, event: str, details: Dict[str, Any]):
        """Append a structured event to the JSONL trace file and publish to the bus."""
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": "selector",
            "event": event,
            "details": details,
        }
        # write to jsonl
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        # also publish to message bus for Supervisor consumption
        if self.message_bus:
            try:
                self.message_bus.publish(entry)
            except Exception:
                # ensure tracing does not crash execution
                pass

    # --------------------------
    # Command handling (runtime)
    # --------------------------
    def _check_for_commands(self):
        """Non-blocking check for commands addressed to 'selector' and apply them."""
        if self.message_bus is None:
            return
        # poll for commands until none are present (non-blocking)
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
        Supported runtime commands (examples):
         - {"command": "adjust_threshold", "value": 0.8}
         - {"command": "set_holdout_years", "value": 2}
         - {"command": "rebuild_node_features", "windows": [5,15,30]}
         - {"command": "retrain_tgn", "epochs": 2}
         - {"command": "dump_state", "path": "state.json"}
        """
        c = cmd.get("command")
        if c == "adjust_threshold":
            new_v = float(cmd.get("value", self.corr_threshold))
            self.corr_threshold = new_v
            self._log_event("threshold_changed", {"new_threshold": self.corr_threshold})
        elif c == "set_holdout_years":
            new_v = int(cmd.get("value", self.holdout_years))
            self.holdout_years = new_v
            self._log_event("holdout_changed", {"holdout_years": self.holdout_years})
        elif c == "rebuild_node_features":
            windows = cmd.get("windows", [5, 15, 30])
            self.build_node_features(windows=windows)
            self._log_event("node_features_rebuilt", {"windows": windows})
        elif c == "retrain_tgn":
            # lightweight wrapper: call train_tgn_temporal_batches if model + optimizer present
            epochs = int(cmd.get("epochs", 1))
            opt = cmd.get("optimizer", None)
            # If no optimizer provided, Supervisor might ask Operator to coordinate training; here we log and skip
            if self.model is not None and opt is not None:
                # if optimizer was serialized as state or path, user needs to attach actual optimizer object
                self.train_tgn_temporal_batches(opt, epochs=epochs)
                self._log_event("retrained", {"epochs": epochs})
            else:
                self._log_event("retrain_requested", {"epochs": epochs, "note": "no_optimizer_attached"})
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
            # Unknown command: just log
            self._log_event("unknown_command", {"command": cmd})

    # --------------------------
    # Node features
    # --------------------------
    def build_node_features(self, windows=[5, 15, 30]) -> pd.DataFrame:

        """
        Build rolling window features for each stock and encode categorical features.

        Rolling features capture temporal behavior:
        - mean: trend over window
        - std: volatility over window
        - cum_return: cumulative log returns (profitability proxy)

        One-hot encodes sectors consistently across the dataset to avoid mismatched feature sizes.
        Standardizes all numeric features to prevent scale bias in the TGNN.
        """

        df = self.df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

        # Create rolling features for each window
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

        # Fill missing fundamental features
        df["eps_yoy_growth"] = df.get("eps_yoy_growth", 0.0).fillna(0.0)
        df["peg_adj"] = df.get("peg_adj", 0.0).fillna(0.0)

        # One-hot encode sector consistently
        all_sectors = sorted(self.df["sector"].unique())  # fixed order of categories
        encoder = OneHotEncoder(sparse_output=False, dtype=int, categories=[all_sectors])
        sector_encoded = encoder.fit_transform(df[["sector"]])
        sector_df = pd.DataFrame(sector_encoded, columns=encoder.get_feature_names_out())
        df = pd.concat([df.reset_index(drop=True), sector_df.reset_index(drop=True)], axis=1)
        df.drop(columns=["sector"], inplace=True, errors="ignore")

        # Fill remaining NaNs with 0
        df.fillna(0.0, inplace=True)

        # Standarization
        exclude_cols = ["date", "ticker"]
        numeric_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

        if self.scaler is None:
            self.scaler = MinMaxScaler()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        # Store standardized node features
        self.node_features = df
        self._log_event("node_features_built", {"n_rows": len(df), "windows": windows})
        return df

    # --------------------------
    # Weekly emporal graph construction with holdout
    # --------------------------
    def build_temporal_graphs(self, corr_threshold: float = None, holdout_years: int = None):

        """
        Build weekly temporal graphs based on correlations between tickers.
        Uses all data except the last "holdout_years" for training.
        Within the last year, splits into:
          - First 6 months: validation
          - Last 6 months: test
        """

        # allow runtime override via either argument or current attribute
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

        # Numeric feature columns
        exclude_cols = ["date", "ticker", "close", "adj_factor", "split_factor", "div_amount", "volume", "sector"]
        feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

        # Define dynamic temporal boundaries
        last_date = df["date"].max()
        holdout_start = last_date - pd.DateOffset(years=holdout_years)
        mid_point = holdout_start + pd.DateOffset(months=6)
        val_start = holdout_start
        val_end = mid_point - pd.DateOffset(days=1)
        test_start = mid_point
        test_end = last_date

        # Split data
        train_df = df[df["date"] < val_start]
        val_df = df[(df["date"] >= val_start) & (df["date"] <= val_end)]
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

        # Build temporal graphs for training data only
        weeks = sorted(train_df["date"].dt.to_period("W").unique())

        for i in range(len(weeks) - 1):
            # check for runtime commands periodically
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
                "start": weekly_features.index.name or str(start_week),
                "end": str(end_week),
                "edge_index": edge_index,
                "edge_attr": edge_attr
            })

        # Store holdout info
        self.val_period = (val_start, val_end)
        self.test_period = (test_start, test_end)
        self.holdout_period = (val_start, test_end)

        self._log_event("temporal_graphs_built", {
            "n_graphs": len(self.temporal_graphs),
            "val_period": (str(self.val_period[0]), str(self.val_period[1])),
            "test_period": (str(self.test_period[0]), str(self.test_period[1])),
            "corr_threshold": corr_threshold
        })

        print(f"Temporal graphs built: {len(self.temporal_graphs)}")
        print(f"Validation period: {self.val_period[0].date()} → {self.val_period[1].date()}")
        print(f"Test period: {self.test_period[0].date()} → {self.test_period[1].date()}")
        print(f"Holdout period: {self.holdout_period[0].date()} → {self.holdout_period[1].date()}")

        return self.temporal_graphs, val_df, test_df

    # --------------------------
    # Memory-based TGNN
    # --------------------------
    class MemoryTGNN(nn.Module):

        """
        Temporal Graph Neural Network (TGNN) with memory.

        Purpose:
        - Learn dynamic node embeddings that evolve over time.
        - Capture both structural relationships (edges) and temporal evolution.
        - Provide embeddings that can be scored for pair selection.

        Components:
        1. Input projection: stabilizes features.
        2. GRUCell: updates a hidden state ("memory") for each node.
        3. Learnable gating to blend old and new memory.
        4. Residual GAT layers with LayerNorm and dropout.
        5. L2-normalized embeddings for scoring pairs.
        6. Decoder: scores pairs of nodes (tickers) using concatenated embeddings.
        """

        def __init__(self, in_channels, hidden_channels=64, num_heads=1, num_layers=1, dropout=0.1, blend_factor=0.3, device=None):
            super().__init__()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.hidden_channels = hidden_channels
            self.num_heads = num_heads
            self.num_layers = max(1, num_layers)
            self.dropout_p = dropout
            self.blend_factor = blend_factor

            # Input projection for stable embeddings
            self.input_proj = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.LayerNorm(hidden_channels)
            )

            # GRU for memory update
            self.msg_gru = nn.GRUCell(hidden_channels, hidden_channels)

            # Learnable gating to blend new and old memory
            self.update_gate = nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.Sigmoid()
            )

            # Residual GAT layers
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(self.num_layers):
                self.convs.append(GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads))
                self.norms.append(nn.LayerNorm(hidden_channels))

            self.dropout = nn.Dropout(p=self.dropout_p)

            # Projection head for L2-normalized embeddings
            self.proj_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.LayerNorm(hidden_channels)
            )

            # Decoder for pair scoring
            self.decoder = nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(hidden_channels, 1)
            )

            self.node_memory = None
            self._reset_parameters()
            self.to(self.device)

        # Initialize parameters with Xavier uniform for better training.
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

            # Initialize memory
            if memory is None:
                memory = torch.zeros(N, self.hidden_channels, device=self.device)
            else:
                memory = memory.to(self.device)

            # Project input
            x_proj = self.input_proj(x)

            # Optional: add time encoding if provided
            if timestamps is not None:
                tproj = torch.tanh(nn.Linear(timestamps.shape[-1], self.hidden_channels).to(self.device)(timestamps.to(self.device)))
                x_proj = x_proj + tproj

            # GRU memory update
            new_mem = self.msg_gru(x_proj, memory)

            # Gated blend between old and new memory
            gate_in = torch.cat([memory, new_mem], dim=1)
            gate = self.update_gate(gate_in)
            memory = (1.0 - gate) * memory + gate * new_mem

            # Residual GAT layers
            h = memory
            for conv, norm in zip(self.convs, self.norms):
                if edge_index is not None and edge_index.numel() > 0:
                    h_conv = conv(h, edge_index)
                else:
                    h_conv = h
                h = norm(h + self.dropout(F.relu(h_conv)))

            # Save updated memory
            self.node_memory = memory

            # Projection head + L2 normalization
            z = self.proj_head(h)
            z = F.normalize(z, p=2, dim=1)

            # Pair scoring if requested
            if pair_index is not None:
                src_idx = pair_index[0].long().to(self.device)
                dst_idx = pair_index[1].long().to(self.device)
                pair_embed = torch.cat([z[src_idx], z[dst_idx]], dim=1)
                logits = self.decoder(pair_embed).view(-1)
                return logits

            return z, memory

        def model_memory(self):
            return self.node_memory

    # --------------------------
    # Training
    # --------------------------
    def train_tgn_temporal_batches(self, optimizer, batch_size=16, epochs=3, neg_sample_ratio=1):

        """
        Train the TGNN model across temporal graph snapshots using
        binary cross entropy + contrastive negative sampling.

        If self.decoder exists it will be used to compute logits from concatenated node embeddings.
        Otherwise a small two-layer decoder is created and attached to self as self._train_decoder
        and its params are added to the optimizer.
        """

        numeric_features = self.node_features.drop(columns=["date", "ticker"], errors="ignore").select_dtypes(include=[np.number])
        if numeric_features.empty:
            raise ValueError("No numeric columns found in node_features after filtering.")

        # Convert to tensor safely
        x = torch.from_numpy(numeric_features.values).float().to(self.device)

        tickers = self.node_features["ticker"].unique().tolist()
        num_nodes = len(tickers)
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        # Prepare decoder: prefer existing self.decoder, else create one and attach it to self
        decoder = getattr(self, "decoder", None)
        if decoder is None:
            # create a small two-layer MLP that maps concatenated embeddings -> logit
            # we will attach it to self so it can be saved/inspected later
            # embedding dim (d) will be inferred on first forward
            if not hasattr(self, "_train_decoder"):
                self._train_decoder = None  # placeholder; will be initialized lazily
            decoder = getattr(self, "_train_decoder", None)

        bce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

        self._log_event("tgn_training_started", {"n_snapshots": len(self.temporal_graphs), "epochs": epochs})
        print(f"Starting TGNN training on {len(self.temporal_graphs)} temporal snapshots.")
        memory = None  # lag-one memory state (Si-1)
        for epoch in range(epochs):
            if self.model is None:
                raise RuntimeError("No model attached to SelectorAgent. Set self.model before training.")
            self.model.train()
            total_loss = 0.0
            total_events = 0

            for i, graph in enumerate(self.temporal_graphs):
                # periodically allow Supervisor to intervene between snapshots
                self._check_for_commands()

                optimizer.zero_grad()

                # detach/clones to avoid inplace warnings
                edge_index = graph["edge_index"].detach().clone()  # shape [2, E]
                # edge_attr may not be needed for link pred; keep if model requires it
                edge_attr = graph.get("edge_attr", None)
                if edge_attr is not None:
                    edge_attr = edge_attr.detach().clone()

                # Forward pass: expect embeddings (num_nodes x d), optionally memory_out
                model_out = self.model(x, edge_index=edge_index, memory=memory)  # pass previous memory (lag-one)
                # handle different return shapes
                if isinstance(model_out, tuple) and len(model_out) >= 2:
                    z, memory = model_out[0], model_out[1]
                else:
                    z = model_out
                    # memory remains whatever model may have updated internally (if it doesn't return memory we keep previous memory)

                # Detach memory to avoid backward-through-graph errors
                if memory is not None:
                    memory = memory.detach()

                if z is None:
                    raise RuntimeError("Model returned None embeddings. Expected node embeddings tensor.")

                if z.dim() == 1:
                    # make sure z is (num_nodes, d)
                    raise RuntimeError("Embeddings have wrong shape; expected (num_nodes, d).")

                # lazy initialize decoder if needed (we now know embedding dim)
                if getattr(self, "_train_decoder", None) is None and getattr(self, "decoder", None) is None:
                    d = z.size(1)
                    self._train_decoder = nn.Sequential(
                        nn.Linear(2 * d, d),
                        nn.ReLU(),
                        nn.Linear(d, 1)
                    ).to(self.device)
                    decoder = self._train_decoder
                    # add new params to optimizer so they're updated
                    optimizer.add_param_group({"params": self._train_decoder.parameters()})
                elif getattr(self, "decoder", None) is None:
                    decoder = self._train_decoder

                # Prepare positive pairs from edge_index
                if edge_index.numel() == 0:
                    # no events in this snapshot
                    continue

                src = edge_index[0]  # shape (E,)
                dst = edge_index[1]  # shape (E,)
                E = src.size(0)
                total_events += E

                # gather node embeddings for pairs
                z_src = z[src]  # (E, d)
                z_dst = z[dst]  # (E, d)
                pos_cat = torch.cat([z_src, z_dst], dim=1)  # (E, 2d)

                # compute logits for positive pairs
                logits_pos = decoder(pos_cat).view(-1)  # (E,)

                # sample negatives: for each positive edge sample `neg_sample_ratio` negative destinations
                # ensure negatives are not the true destination (simple rejection sampling)
                neg_samples = []
                num_neg = int(E * neg_sample_ratio)
                if num_neg > 0:
                    # sample random destination node indices (allow duplicates for speed)
                    # perform in-device sampling
                    rand_idx = torch.randint(0, num_nodes, (num_neg,), device=self.device)
                    # ensure rand_idx != src[:num_neg] by simple resample loop (cheap unless tiny graph)
                    # align size by repeating src if needed
                    src_for_neg = src.repeat((neg_sample_ratio,))[:num_neg]
                    # avoid exact matches
                    mask_equal = rand_idx == src_for_neg
                    while mask_equal.any():
                        rand_idx[mask_equal] = torch.randint(0, num_nodes, (mask_equal.sum().item(),), device=self.device)
                        mask_equal = rand_idx == src_for_neg
                    neg_src = z[src_for_neg]  # (num_neg, d)
                    neg_dst = z[rand_idx]      # (num_neg, d)
                    neg_cat = torch.cat([neg_src, neg_dst], dim=1)  # (num_neg, 2d)
                    logits_neg = decoder(neg_cat).view(-1)  # (num_neg,)
                else:
                    logits_neg = torch.empty(0, device=self.device)

                # Labels: positives = 1, negatives = 0
                pos_labels = torch.ones_like(logits_pos, device=self.device)
                if logits_neg.numel() > 0:
                    neg_labels = torch.zeros_like(logits_neg, device=self.device)
                    logits = torch.cat([logits_pos, logits_neg], dim=0)
                    labels = torch.cat([pos_labels, neg_labels], dim=0)
                else:
                    logits = logits_pos
                    labels = pos_labels

                # compute loss (BCE with logits)
                loss = bce_loss_fn(logits, labels)

                # backward + step
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * (E if logits_neg.numel() == 0 else (E + logits_neg.numel()))
                # Note: total_loss accumulates scaled sum to report an aggregate; you can change granularity as needed.

            avg_loss = total_loss / max(total_events, 1)
            self._log_event("tgn_epoch_complete", {"epoch": epoch + 1, "avg_loss": avg_loss})
            print(f"Epoch {epoch+1}/{epochs} complete. Avg (scaled) loss: {avg_loss:.6f}")

        self._log_event("tgn_training_complete", {"epochs": epochs})
        print("✅ TGNN training complete.")

    # --------------------------
    # Score pairs
    # --------------------------
    def score_all_pairs_holdout(self):

        """
        Score all ticker pairs using memory propagation.
        Ensures all nodes have embeddings even if they were not in edges.
        """

        tickers = self.node_features["ticker"].unique().tolist()
        num_nodes = len(tickers)

        # Convert node features to tensor
        x_full = torch.tensor(
            self.node_features.drop(columns=["date", "ticker"]).values,
            dtype=torch.float,
            device=self.device
        )

        self.model.eval()
        memory = None

        with torch.no_grad():
            # Propagate memory through all temporal graphs
            for g in self.temporal_graphs:
                # allow supervisor to intervene between graph steps
                self._check_for_commands()

                out = self.model(x_full, g["edge_index"], memory=memory)

                # Ensure memory is always extracted
                if isinstance(out, tuple) and len(out) == 2:
                    _, memory = out
                else:
                    memory = out

                # Detach memory to avoid retaining computation graph
                memory = memory.detach()

            # memory now contains final node embeddings
            h = memory

            # Generate all unique pairs (upper triangle)
            src_idx, dst_idx = np.triu_indices(num_nodes, k=1)
            pair_index = torch.tensor([src_idx, dst_idx], dtype=torch.long, device=self.device)

            # Score pairs using decoder
            pair_embed = torch.cat([h[src_idx], h[dst_idx]], dim=1)
            scores = self.model.decoder(pair_embed)

        # Build results DataFrame
        results = pd.DataFrame({
            "x": np.array(tickers)[src_idx],
            "y": np.array(tickers)[dst_idx],
            "score": scores.cpu().numpy().flatten()
        }).sort_values("score", ascending=False).reset_index(drop=True)

        # log scoring summary
        topk = results.head(10).to_dict(orient="records")
        self._log_event("pairs_scored", {"n_pairs": len(results), "top_10": topk})

        return results

    # --------------------------
    # Pair validation
    # --------------------------
    def validate_pairs(self, df_pairs: pd.DataFrame, validation_window: Tuple[pd.Timestamp, pd.Timestamp], half_life_max: float = 60, min_crossings_per_year: int = 24):

        """
        Validate scored pairs in the holdout period.

        Checks:
        - Mean reversion (ADF test)
        - Half-life < half_life_max
        - Sufficient crossings per year
        - Sharpe ratio
        - Sortino ratio
        - Maximum drawdown
        """

        def compute_spread(x: pd.Series, y: pd.Series) -> pd.Series:
            x, y = x.align(y, join="inner")
            if len(x) < 2 or len(y) < 2:
                return pd.Series(dtype=float)
            beta = np.polyfit(y.values, x.values, 1)[0]
            return x - beta * y

        def half_life(spread: np.ndarray) -> float:
            if len(spread) < 2:
                return np.inf
            spread = np.array(spread)
            spread_lag = np.roll(spread, 1)[1:]
            delta_spread = np.diff(spread)
            beta = np.polyfit(spread_lag, delta_spread, 1)[0]
            if beta >= 0:
                return np.inf
            return -np.log(2) / beta

        start, end = validation_window
        validated = []
        prices = self.df.pivot(index="date", columns="ticker", values="adj_close").sort_index()

        for _, row in df_pairs.iterrows():
            # allow Supervisor to interrupt during potentially long validation loops
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

            # Count mean crossings
            crossings = ((spread - spread.mean()).shift(1) * (spread - spread.mean()) < 0).sum()
            days = (series_x.index[-1] - series_x.index[0]).days
            crossings_per_year = crossings / (days / 252)
            if crossings_per_year < min_crossings_per_year:
                continue

            # ADF test for stationarity
            try:
                adf_stat, adf_p, *_ = adfuller(spread.dropna())
            except Exception:
                adf_p = 1.0

            hl = half_life(spread.values)

            # Pass criteria
            pass_criteria = (adf_p < 0.05) and (hl < half_life_max)

            validated.append({
                "x": x, "y": y, "score": row["score"],
                "adf_p": float(adf_p), "half_life": float(hl),
                "mean_crossings": int(crossings),
                "pass": pass_criteria
            })

        self._log_event("pairs_validated", {"n_validated": len(validated)})
        return pd.DataFrame(validated)

# ========== CODE CELL 4 ==========
# --------------------------
# Pair Trading Environment
# --------------------------
class PairTradingEnv(gym.Env):
    metadata = {"render.modes": ["human", "plot"]}

    def __init__(self, series_x: pd.Series, series_y: pd.Series, lookback: int = 500,
                 shock_prob: float = 0.01, shock_scale: float = 0.1,
                 initial_capital: float = 1000):

        super().__init__()

        # Align and clean
        self.align = pd.concat([series_x, series_y], axis=1).dropna()
        self.lookback = lookback
        self.ptr = lookback

        # Action/Observation spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # Portfolio state
        self.position = 0
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.cum_returns = 0.0
        self.peak = initial_capital
        self.max_drawdown = 0.0
        self.trades = []

        self.shock_prob = shock_prob
        self.shock_scale = shock_scale

        # Precompute spread features
        self.spread = compute_spread(self.align.iloc[:, 0], self.align.iloc[:, 1])
        n = len(self.spread)
        shock_mask = np.random.rand(n) < self.shock_prob
        self.shocks = np.random.randn(n) * self.shock_scale * self.spread.std() * shock_mask
        self.spread_shocked = self.spread + self.shocks

        self.zscores = (self.spread_shocked - self.spread_shocked.rolling(self.lookback).mean()) / self.spread_shocked.rolling(self.lookback).std()
        self.vols = self.spread_shocked.rolling(21).std()
        self.rx = self.align.iloc[:, 0].pct_change()
        self.ry = self.align.iloc[:, 1].pct_change()
        self.corrs = self.rx.rolling(21).corr(self.ry)

        # Convert to NumPy arrays
        self.zscores_np = np.nan_to_num(self.zscores.to_numpy())
        self.vols_np = np.nan_to_num(self.vols.to_numpy())
        self.corrs_np = np.nan_to_num(self.corrs.to_numpy())
        self.spread_np = self.spread_shocked.to_numpy()

    def _compute_features(self, idx: int):
        z = self.zscores_np[idx]
        vol = self.vols_np[idx]
        corr = self.corrs_np[idx]
        start = max(0, idx - self.lookback)
        hl = half_life(self.spread_np[start:idx]) if idx > start else CONFIG["half_life_max"]
        return np.array([z, vol, hl, corr], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.ptr = self.lookback
        self.position = 0
        self.portfolio_value = self.initial_capital
        self.cum_returns = 0.0
        self.peak = self.initial_capital
        self.max_drawdown = 0.0
        self.trades = []
        obs = self._compute_features(self.ptr)
        return obs, {}

    def step(self, action: int):
        target_pos = action - 1
        next_ptr = self.ptr + 1
        terminated = next_ptr >= len(self.spread_np)
        truncated = False

        obs_next = self._compute_features(self.ptr)

        if terminated:
            return obs_next, 0.0, terminated, truncated, {
                "cum_returns": self.cum_returns,
                "max_drawdown": self.max_drawdown
            }

        # Compute PnL based on spread change
        ret = self.spread_np[next_ptr] - self.spread_np[self.ptr]
        reward = -ret * target_pos / (self.align.iloc[self.lookback, 0])

        # Transaction cost
        if target_pos != self.position:
            reward -= CONFIG["transaction_cost"] + 0.002

        # Update portfolio
        daily_return = reward / max(1e-8, self.portfolio_value)
        self.portfolio_value *= (1 + daily_return)
        self.cum_returns = self.portfolio_value - self.initial_capital
        self.peak = max(self.peak, self.portfolio_value)
        self.max_drawdown = max(self.max_drawdown, (self.peak - self.portfolio_value) / self.peak)

        self.position = target_pos
        self.ptr = next_ptr
        self.trades.append(daily_return)

        return obs_next, float(daily_return), terminated, truncated, {
            "pnl": daily_return,
            "cum_returns": self.cum_returns,
            "max_drawdown": self.max_drawdown
        }

# --------------------------
# Operator Agent (Enhanced)
# --------------------------
@dataclass
class OperatorAgent:
    message_bus: MessageBus
    logger: JSONLogger
    storage_dir: str = "models/"

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        self.active = True
        self.transaction_cost = CONFIG["transaction_cost"]

    def apply_command(self, command):
        cmd_type = command.get("command")
        if cmd_type == "adjust_transaction_cost":
            old = self.transaction_cost
            self.transaction_cost = command.get("new_value", old)
            CONFIG["transaction_cost"] = self.transaction_cost
            self.logger.log("operator", "adjust_transaction_cost", {
                "old_value": old, "new_value": self.transaction_cost
            })
        elif cmd_type == "pause":
            self.active = False
            self.logger.log("operator", "paused", {})
        elif cmd_type == "resume":
            self.active = True
            self.logger.log("operator", "resumed", {})

    def train_on_pair(self, prices: pd.DataFrame, x: str, y: str,
                      lookback: int = 252, timesteps: int = CONFIG["rl_timesteps"]):
        if not self.active:
            return None

        series_x = prices[x]
        series_y = prices[y]
        env = PairTradingEnv(series_x, series_y, lookback)
        model = PPO(CONFIG["rl_policy"], env, verbose=0, device="cpu")
        model.learn(total_timesteps=timesteps)

        model_path = os.path.join(self.storage_dir, f"operator_model_{x}_{y}.zip")
        model.save(model_path)

        obs, _ = env.reset()
        done = False
        daily_returns = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            daily_returns.append(reward)

        rets = np.array(daily_returns)
        rf_daily = 0.02 / 252
        excess_rets = rets - rf_daily

        sharpe = np.mean(excess_rets) / (np.std(excess_rets, ddof=1) + 1e-8) * np.sqrt(252)
        downside = excess_rets[excess_rets < 0]
        sortino = (np.mean(excess_rets) / (np.std(downside, ddof=1) + 1e-8) * np.sqrt(252)) if len(downside) else np.inf

        trace = {
            "pair": (x, y),
            "cum_reward": (np.prod(1 + rets) - 1) * 100,
            "max_drawdown": env.max_drawdown,
            "sharpe": sharpe,
            "sortino": sortino,
            "model_path": model_path
        }

        self.logger.log("operator", "pair_trained", trace)
        self.message_bus.send_command({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": "operator",
            "event": "pair_trained",
            "details": trace
        })

        return trace

# --------------------------
# Parallel Training Function
# --------------------------
def train_operator_on_pairs(operator: OperatorAgent, prices: pd.DataFrame, pairs: list, max_workers: int = 2):
    all_traces = []
    def train(pair):
        x, y = pair
        print(f"\n🔹 Training Operator on pair ({x}, {y})")
        return operator.train_on_pair(prices, x, y)

    print(pairs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train, pair) for pair in pairs]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Operator Training"):
            result = f.result()
            if result:
                all_traces.append(result)

    save_json(all_traces, os.path.join(operator.storage_dir, "all_operator_traces.json"))
    operator.logger.log("operator", "batch_training_complete", {"n_pairs": len(all_traces)})
    print("\n✅ All traces saved successfully.")
    return all_traces


# ========== CODE CELL 5 ==========
# --------------------------
# Supervisor Agent
# --------------------------

try:
    from langchain import LLMChain, PromptTemplate
    from langchain.chat_models import HuggingFaceHub
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python
!pip3 install huggingface-hub
!pip3 install sentence-transformers langchain langchain-experimental
!huggingface-hf download TheBloke/Llama-2-13B-chat-GGUF llama-2-13b-chat.Q4_K_M.gguf --local-dir /content --local-dir-use-symlinks False

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

@dataclass
class SupervisorAgent:
    """
    SupervisorAgent that monitors operator traces and selector/operator events,
    suggests and issues actions via the MessageBus, keeps a JSONL trace,
    and can optionally use a local LLM + LangChain prompt for richer analysis.
    """
    message_bus: MessageBus
    logger: JSONLogger
    max_total_drawdown: float = 0.20
    storage_dir: str = "./storage"
    llm_model_path: str = "/content/llama-2-13b-chat.Q4_K_M.gguf"
    temperature: float = 0.1
    n_gpu_layers: int = 80  # adjust according to your GPU VRAM
    n_batch: int = 512
    llm: Optional[LlamaCpp] = None
    explainer_chain: Optional[LLMChain] = None

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)

        # Initialize callback manager
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        try:
            # Initialize LlamaCpp model
            self.llm = LlamaCpp(
                model_path=self.llm_model_path,
                temperature=self.temperature,
                n_gpu_layers=self.n_gpu_layers,
                n_batch=self.n_batch,
                callback_manager=callback_manager,
                verbose=True,
            )
            # Initialize explainer chain if LangChain is available
            if HAS_LANGCHAIN:
                self.explainer_chain = LLMChain(
                    llm=self.llm,
                    prompt=PromptTemplate(
                        input_variables=["metrics", "actions"],
                        template=(
                            "You are an expert quantitative supervisor overseeing algorithmic trading agents.\n"
                            "Given the portfolio metrics below, explain in natural language why the following actions are being taken.\n\n"
                            "Metrics:\n{metrics}\n\n"
                            "Actions:\n{actions}\n\n"
                            "Explain clearly and concisely the reasoning behind each action, including risk management implications."
                        )
                    ),
                )
            print(f"LLaMA LLM loaded: {self.llm_model_path}")
        except Exception as e:
            self.llm = None
            self.explainer_chain = None
            print("Warning: could not load LLaMA 13B model; continuing without LLM:", str(e))

        # initial log
        self.logger.log("supervisor", "init", {
            "max_total_drawdown": self.max_total_drawdown,
            "storage_dir": self.storage_dir
        })

    # --------------------------
    # Map high-level actions to concrete commands for agents
    # --------------------------
    def _action_to_commands(self, action: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert a high-level action (from rules or LLM) into concrete message-bus commands.
        Example mappings:
          - retrain_selector -> {"target":"selector", "command":"retrain_tgn", "epochs": 2}
          - reduce_risk -> {"target":"operator", "command":"adjust_transaction_cost", "new_value": 0.01}
          - freeze_agents -> pause both selector and operator
        """
        act = action.get("action", "")
        reason = action.get("reason", "")
        commands = []

        if act == "retrain_selector":
            commands.append({"target": "selector", "command": "retrain_tgn", "epochs": action.get("epochs", 1)})
        elif act == "reduce_risk":
            # increase transaction cost slightly and ask operators to reduce position size
            new_tc = action.get("new_transaction_cost", CONFIG.get("transaction_cost", 0.005) + 0.002)
            commands.append({"target": "operator", "command": "adjust_transaction_cost", "new_value": new_tc})
            commands.append({"target": "operator", "command": "adjust_position_size", "new_value": action.get("new_position_size", 0.5)})
        elif act == "freeze_agents" or act == "pause_agents":
            commands.append({"target": "selector", "command": "pause"})
            commands.append({"target": "operator", "command": "pause"})
        elif act == "resume_agents":
            commands.append({"target": "selector", "command": "resume"})
            commands.append({"target": "operator", "command": "resume"})
        elif act == "review_required":
            # no automatic command; just log and alert human operator
            pass
        elif act == "increase_capital_allocation":
            # Example: alert operator to increase position size (conservative mapping)
            commands.append({"target": "operator", "command": "adjust_position_size", "new_value": action.get("new_value", 1.0)})
        else:
            # Generic passthrough if LLM returns a command-like dict
            if "target" in action and "command" in action:
                commands.append(action)

        return commands

    # --------------------------
    # LLM helper (attempt to get a single JSON action)
    # --------------------------
    def llm_based_analysis(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Query the LLM for a single high-level recommended action in JSON.
        If LLM is unavailable or outputs invalid JSON, return a fallback action dict.
        """
        if self.llm is None:
            # simple heuristic fallback using rules
            # If many negatives or large drawdown, ask to reduce risk
            if metrics.get("negatives", 0) > metrics.get("n_traces", 0) * 0.5:
                return {"action": "retrain_selector", "reason": "fallback_rule"}
            if metrics.get("max_drawdown", 0) > self.max_total_drawdown:
                return {"action": "reduce_risk", "reason": "fallback_rule", "new_transaction_cost": CONFIG.get("transaction_cost", 0.005) + 0.002}
            return {"action": "no_op", "reason": "fallback_rule"}

        # Compose a prompt/query and call LLM (expect JSON string)
        prompt = (
            f"Given these portfolio metrics: {json.dumps(metrics, default=str)}, "
            "suggest one high-level action (in JSON format) to improve portfolio stability or returns. "
            "Examples: {\"action\":\"rebalance\"}, {\"action\":\"freeze_agents\"}, {\"action\":\"retrain_selector\"}."
        )
        try:
            # LangChain-compatible call if explainer_chain exists
            if self.explainer_chain is not None:
                # Use explainer_chain to generate natural-language explanation AND action (we expect LLM to return JSON)
                response = self.explainer_chain.run(metrics=json.dumps(metrics, default=str), actions="[]")
                # try to parse JSON blob inside response
                parsed = json.loads(response)
                return parsed
            else:
                raw = self.llm(prompt)
                # raw may be a string or list-like; coerce to string then parse
                text = raw if isinstance(raw, str) else str(raw)
                # try to find first JSON-like substring
                try:
                    parsed = json.loads(text)
                    return parsed
                except json.JSONDecodeError:
                    # try to extract JSON object between first {...} pair
                    import re
                    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
                    if m:
                        try:
                            return json.loads(m.group(0))
                        except Exception:
                            pass
                    return {"action": "review_required", "reason": "LLM_unstructured_output", "raw": text}
        except Exception as e:
            # on any LLM error, fall back to rule-based suggestion
            self.logger.log("supervisor", "llm_error", {"error": str(e)})
            if metrics.get("max_drawdown", 0) > self.max_total_drawdown:
                return {"action": "reduce_risk", "reason": "llm_error_fallback"}
            return {"action": "no_op", "reason": "llm_error_fallback"}

    # --------------------------
    # Evaluate portfolio traces and issue commands (main sync method)
    # --------------------------
    def evaluate_portfolio(self, operator_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Portfolio metrics
        total_return = sum(t.get("cum_reward", 0.0) for t in operator_traces)
        max_dd = max([t.get("max_drawdown", 0.0) for t in operator_traces] + [0.0])
        returns = [t.get("cum_reward", 0.0) for t in operator_traces if "cum_reward" in t]

        var_95, cvar_95 = 0.0, 0.0
        if returns:
            var_95 = float(np.percentile(returns, 5))
            cvar_95 = float(np.mean([r for r in returns if r <= var_95]) if any(r <= var_95 for r in returns) else var_95)

        metrics_summary = {
            "total_return": float(total_return),
            "max_drawdown": float(max_dd),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "negatives": int(sum(1 for t in operator_traces if t.get("cum_reward", 0.0) < 0)),
            "n_traces": len(operator_traces)
        }

        # Rule-based actions
        actions_rules = []
        if metrics_summary["negatives"] > len(operator_traces) * 0.6:
            actions_rules.append({"action": "retrain_selector", "reason": "systemic_underperformance"})
        if max_dd > self.max_total_drawdown:
            actions_rules.append({"action": "reduce_risk", "reason": "portfolio_drawdown_limit_exceeded"})

        # LLM-based action
        llm_action = self.llm_based_analysis(metrics_summary)
        actions_llm = [llm_action] if llm_action else []

        # Merge via swarm
        actions = self.swarm.merge_actions(actions_rules + actions_llm)

        # Graph recording
        self.graph.add_node("PortfolioMetrics", "metrics")
        for act in actions:
            node_name = f"{act.get('action')} ({act.get('reason','')})"
            self.graph.add_node(node_name)
            self.graph.add_edge("PortfolioMetrics", node_name)

        # Explain actions (try LangChain explainer if available, else simple template)
        explanation = ""
        try:
            if self.explainer_chain is not None:
                explanation = self.explainer_chain.run(metrics=json.dumps(metrics_summary), actions=json.dumps(actions))
            else:
                explanation = f"Actions chosen: {actions}. Metrics: {metrics_summary}."
        except Exception as e:
            explanation = f"Explanation generation failed: {str(e)}"

        # Build supervisor record
        supervisor_record = {
            "total_return": metrics_summary["total_return"],
            "max_drawdown": metrics_summary["max_drawdown"],
            "var_95": metrics_summary["var_95"],
            "cvar_95": metrics_summary["cvar_95"],
            "actions": actions,
            "explanation": explanation,
            "timestamp": time.time()
        }

        # Persist and log
        os.makedirs(self.storage_dir, exist_ok=True)
        save_path = os.path.join(self.storage_dir, f"supervisor_record_{int(time.time())}.json")
        with open(save_path, "w") as f:
            json.dump(supervisor_record, f, indent=2, default=str)

        self.graph.export(os.path.join(self.storage_dir, f"supervisor_graph_{int(time.time())}.json"))
        self.logger.log("supervisor", "evaluation", supervisor_record)

        # Convert actions to concrete commands and send via message bus
        all_commands = []
        for act in actions:
            cmd_list = self._action_to_commands(act)
            for c in cmd_list:
                # attach metadata
                c_meta = dict(c)
                c_meta["issued_by"] = "supervisor"
                c_meta["issued_at"] = datetime.datetime.utcnow().isoformat()
                self.message_bus.send_command(c_meta["target"], c_meta)
                all_commands.append(c_meta)
                # log each command sent
                self.logger.log("supervisor", "command_sent", c_meta)

        # include issued commands in supervisor_record for traceability
        supervisor_record["issued_commands"] = all_commands
        # update persisted copy
        with open(save_path, "w") as f:
            json.dump(supervisor_record, f, indent=2, default=str)

        return supervisor_record

    # --------------------------
    # Continuous run loop (non-blocking if run in a thread)
    # --------------------------
    def run(self, poll_interval: Optional[float] = None):
        """
        Continuously monitor operator traces/events and periodically evaluate portfolio.
        Intended to be run in its own thread or as a background task.
        """
        if poll_interval is None:
            poll_interval = self.poll_interval

        self.logger.log("supervisor", "started", {"poll_interval": poll_interval})
        # Buffer for operator traces collected from events
        operator_traces: List[Dict[str, Any]] = []

        try:
            while True:
                # Drain events from bus (non-blocking)
                events = self.message_bus.drain_events()
                for ev in events:
                    # capture operator trace events of interest
                    if ev.get("agent") == "operator" and ev.get("type") in ("pair_trained", "trade_executed", "episode_finished"):
                        operator_traces.append(ev.get("details", {}))
                        # keep trace log also for Supervisor
                        self.logger.log("supervisor", "event_received", ev)

                # Periodic full evaluation
                if operator_traces:
                    # Evaluate and possibly issue commands
                    record = self.evaluate_portfolio(operator_traces)
                    # clear traces after evaluation to avoid repeated triggers (policy choice)
                    operator_traces = []

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            self.logger.log("supervisor", "stopped", {"reason": "keyboard_interrupt"})
            return
        except Exception as e:
            self.logger.log("supervisor", "error", {"error": str(e)})
            raise

# ========== CODE CELL 6 ==========
# --- Imports and dependencies ---
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor
import threading

# --- Utility: MessageBus and Logger from your notebook ---
message_bus = MessageBus()
logger = JSONLogger(path="traces/supervisortrace.jsonl")

# --- 1. Load historical dataframe ---
# Replace this with your actual data loading code
# df = pd.read_csv('your_data.csv')
# For mockup: assuming df is loaded from earlier notebook context

# --- 2. Initialize SelectorAgent (NO messagebus argument!) ---
selector = SelectorAgent(df=df)
selector.message_bus = message_bus
selector.trace_path = "traces/selectortrace.jsonl"

# --- 3. Build node features, graphs, and model ---
selector.build_node_features(windows=[5, 15, 30])
selector.build_temporal_graphs(corr_threshold=0.7, holdout_years=1)
selector.model = SelectorAgent.MemoryTGNN(
    in_channels=len(selector.node_features.drop(columns=["date", "ticker"]).columns),
    hidden_channels=32,
    num_heads=1,
    num_layers=2,
    device=selector.device
)
optimizer = torch.optim.Adam(selector.model.parameters(), lr=0.001)

# --- 4. Train selector TGNN model ---
selector.train_tgn_temporal_batches(
    optimizer=optimizer, batch_size=16, epochs=5, neg_sample_ratio=1
)

# --- 5. Score and validate pairs ---
df_scores = selector.score_all_pairs_holdout()
top_pairs = df_scores.head(25)
validation_results = selector.validate_pairs(
    df_pairs=top_pairs,
    validation_window=(selector.holdout_period),
    half_life_max=60,
    min_crossings_per_year=12
)
print("Validation results:\n", validation_results)

# --- 6. Prepare price dataframe for operator ---
prices = df.pivot(index="date", columns="ticker", values="adj_close").sort_index().dropna(how="any", axis=0)

# --- 7. Initialize OperatorAgent with required positional arguments ---
operator = OperatorAgent(message_bus, logger, storage_dir="models")

# --- 8. Train OperatorAgent on pairs (parallel execution) ---
tickers = list(prices.columns)
pairs = [("KLAC", "RTX"), ("EXR", "INCY")]  # Replace ... with your real top pairs
def trainoperatoronpairs(operator, prices, pairs, maxworkers=2):
    alltraces = []
    def trainpair(pair):
        return operator.train_on_pair(prices, pair[0], pair[1])
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(trainpair, pairs))
        alltraces.extend(results)
    return alltraces

results = train_operator_on_pairs(operator, prices, pairs, max_workers=2)

# --- 9. Start SupervisorAgent in a background thread to monitor agents ---
supervisor = SupervisorAgent(
    message_bus=message_bus,
    logger=logger,
    max_total_drawdown=0.20,
    storage_dir=".storage",
    model_name="AI4Finance-Foundation/FinGPT-Forecaster-dow30",
    poll_interval=5.0
)
supervisor_thread = threading.Thread(target=supervisor.run)
supervisor_thread.start()

# ========== CODE CELL 7 ==========
# --- 9. Start SupervisorAgent in a background thread to monitor agents ---
supervisor = SupervisorAgent(
    message_bus=message_bus,
    logger=logger,
    max_total_drawdown=0.20,
    storage_dir=".storage"
)
supervisor_thread = threading.Thread(target=supervisor.run)
supervisor_thread.start()

# ========== CODE CELL 8 ==========


# ========== CODE CELL 9 ==========


# ========== CODE CELL 10 ==========


# ========== CODE CELL 11 ==========
# --------------------------
# SelectorAgent run
# --------------------------

# Initialize SelectorAgent with your full dataframe
selector = SelectorAgent(df=df)

# Step 1: Build node features
node_features = selector.build_node_features(windows=[5, 15, 30])
print("✅ Node features built")

# Step 2: Build temporal graphs with a holdout period (e.g., 1 year)
temporal_graphs = selector.build_temporal_graphs(corr_threshold=0.7, holdout_years=1)
print(f"✅ Built {len(temporal_graphs)} temporal graphs.")
selector.holdout_start, selector.holdout_end = selector.holdout_period
print("Holdout period:", selector.holdout_start.date(), "→", selector.holdout_end.date())

# Step 3: Initialize memory-based TGNN model
in_channels = len(selector.node_features.drop(columns=["date", "ticker"]).columns)
selector.model = SelectorAgent.MemoryTGNN(
    in_channels=in_channels,
    hidden_channels=32,
    num_heads=1,
    num_layers=2,
    device=selector.device
).to(selector.device)
print("✅ TGNN model initialized")

# Step 4: Define optimizer
optimizer = torch.optim.Adam(selector.model.parameters(), lr=0.001)

# Step 5: Train TGNN on temporal graphs (excluding holdout period)
selector.train_tgn_temporal_batches(
    optimizer=optimizer,
    batch_size=16,
    epochs=5,
    neg_sample_ratio=1
)
print("✅ TGNN training complete")

# Step 6: Score all ticker pairs in the holdout period
df_scores = selector.score_all_pairs_holdout()
print("✅ Holdout pair scores (sample):")
print(df_scores.head())

# Step 7: Select top-N pairs for validation
top_pairs = df_scores.head(25)
print(f"✅ Selected {len(top_pairs)} top-scoring pairs for validation")

# Step 8: Validate top pairs on the holdout window
validation_results = selector.validate_pairs(
    df_pairs=top_pairs,
    validation_window=(selector.holdout_start, selector.holdout_end),
    half_life_max=60,
    min_crossings_per_year=12
)

print("✅ Validation complete\n")

# Step 9: Display full validation metrics
print("🔍 Full Validation Results:")
display_cols = [
    "x", "y", "score", "adf_p", "half_life", "mean_crossings", "pass"
]
print(validation_results[validation_results["pass"] == True][display_cols]
      .sort_values(by="score", ascending=False)
      .to_string(index=False))



# ========== CODE CELL 12 ==========

# --------------------------
# Run OperatorAgent
# --------------------------

prices = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
prices = prices.dropna(how="any", axis=0)

operator = OperatorAgent()

# --------------------------
# Example Usage
# --------------------------
tickers = list(prices.columns)
pairs = [("KLAC", "RTX"), ("EXR", "INCY")]
operator = OperatorAgent()
results = train_operator_on_pairs(operator, prices, pairs)

results
