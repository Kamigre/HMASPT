"""
Selector Agent for identifying cointegrated stock pairs using Temporal Graph Neural Networks.
Implementation following the paper's methodology exactly.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from statsmodels.tsa.stattools import adfuller

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import CONFIG

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.message_bus import MessageBus, JSONLogger


class TimeEncoder(nn.Module):
    """
    Time encoding function ψ(.) from the paper.
    Maps time differences to d-dimensional vectors.
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.w = nn.Linear(1, d)
        
    def forward(self, t):
        """
        Args:
            t: time differences, shape (batch_size,) or (batch_size, 1)
        Returns:
            d-dimensional time encoding
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        # Use harmonic encoding as in the paper reference [22]
        return torch.cos(self.w(t))


class MessageFunction(nn.Module):
    """
    Message function msg(.) from Equation 1.
    Computes messages for node memory updates.
    Projects concatenated features to memory dimension.
    """
    def __init__(self, memory_dim, edge_dim, time_dim):
        super().__init__()
        # Identity message function concatenates inputs, then projects
        self.memory_dim = memory_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        
        # Project concatenated features to memory dimension
        concat_dim = memory_dim * 2 + edge_dim + time_dim
        self.projection = nn.Linear(concat_dim, memory_dim)
        
    def forward(self, s_i, s_j, e_ij, time_enc):
        """
        Args:
            s_i: memory state of source node
            s_j: memory state of target node
            e_ij: edge features
            time_enc: time encoding ψ(t - t'_i)
        Returns:
            message vector (projected to memory_dim)
        """
        # Concatenate all inputs (identity message function)
        concatenated = torch.cat([s_i, s_j, e_ij, time_enc], dim=-1)
        # Project to memory dimension
        return self.projection(concatenated)


class MemoryModule(nn.Module):
    """
    Memory module mem(.) from Equation 2.
    Updates node memory using GRU.
    """
    def __init__(self, memory_dim):
        super().__init__()
        # Message is now projected to memory_dim, so input/hidden dims match
        self.gru = nn.GRUCell(memory_dim, memory_dim)
        
    def forward(self, message, memory):
        """
        Args:
            message: computed message m_i(t) (already projected to memory_dim)
            memory: previous memory state s_i(t-)
        Returns:
            updated memory s_i(t)
        """
        return self.gru(message, memory)


class TemporalGraphAttention(nn.Module):
    """
    Temporal graph attention from Equation 3.
    L-layer temporal graph attention with time encodings.
    """
    def __init__(self, d, num_heads, num_layers, dropout=0.2):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.time_encoder = TimeEncoder(d)
        
        # MLP and multi-head attention for each layer
        self.mlps = nn.ModuleList([
            nn.Linear(d * 2, d) for _ in range(num_layers)
        ])
        
        self.mhas = nn.ModuleList([
            nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.dropout = dropout
        
    def forward(self, z, edge_index, edge_times, current_time, N=10):
        """
        Args:
            z: initial embeddings z^(0) = s_i(t) + X_i(t)
            edge_index: edge indices
            edge_times: timestamps of edges
            current_time: current timestamp t
            N: number of most recent neighbors to consider
        Returns:
            final embeddings z^(L)
        """
        num_nodes = z.size(0)
        
        # Build temporal neighborhood for each node
        # π_i = {e_π_i(1)(t_π_i(1)), ..., e_π_i(N)(t_π_i(N))}
        neighbors = self._build_temporal_neighbors(
            edge_index, edge_times, num_nodes, N
        )
        
        # L layers of attention
        h = z
        for l in range(self.num_layers):
            h_next = []
            
            for i in range(num_nodes):
                # Query: q_i^(l)(t) = z_i^(l-1) || ψ(0)
                time_zero = self.time_encoder(torch.zeros(1, device=z.device))
                q = torch.cat([h[i:i+1], time_zero], dim=-1)
                
                # Keys and Values: neighbors with time encodings
                if i in neighbors and len(neighbors[i]) > 0:
                    neighbor_indices, neighbor_times = neighbors[i]
                    
                    # Time encodings for neighbors
                    time_diffs = current_time - neighbor_times
                    time_encs = self.time_encoder(time_diffs)
                    
                    # K_i^(l)(t) = V_i^(l)(t) = [z_i^(l-1) || e_π_i(k) || ψ(t - t_π_i(k))]
                    # Simplified: use neighbor embeddings with time encoding
                    kv = torch.cat([
                        h[neighbor_indices],
                        time_encs
                    ], dim=-1)
                    
                    # Add self-connection
                    kv = torch.cat([
                        torch.cat([h[i:i+1], time_zero], dim=-1),
                        kv
                    ], dim=0).unsqueeze(0)
                    
                    # Multi-head attention
                    q_input = q.unsqueeze(0)  # (1, 1, d*2)
                    attn_out, _ = self.mhas[l](q_input, kv, kv)
                    z_tilde = attn_out.squeeze(0)
                else:
                    # No neighbors: use self
                    z_tilde = q
                
                # MLP: z_i^(l) = mlp^(l)(z^(l-1) || z_tilde^(l))
                combined = torch.cat([h[i:i+1], z_tilde], dim=-1)
                h_i = self.mlps[l](combined)
                h_i = F.relu(h_i)
                h_i = F.dropout(h_i, p=self.dropout, training=self.training)
                h_next.append(h_i)
            
            h = torch.cat(h_next, dim=0)
        
        return h
    
    def _build_temporal_neighbors(self, edge_index, edge_times, num_nodes, N):
        """
        Build temporal neighborhood π_i for each node.
        Returns dict mapping node_id -> (neighbor_indices, neighbor_times)
        """
        neighbors = {i: ([], []) for i in range(num_nodes)}
        
        if edge_index.numel() == 0:
            return neighbors
        
        # For each edge, add to both nodes' neighborhoods
        for idx in range(edge_index.size(1)):
            src = edge_index[0, idx].item()
            dst = edge_index[1, idx].item()
            t = edge_times[idx] if edge_times is not None else 0.0
            
            neighbors[src][0].append(dst)
            neighbors[src][1].append(t)
            
            neighbors[dst][0].append(src)
            neighbors[dst][1].append(t)
        
        # Keep only N most recent neighbors for each node
        for i in range(num_nodes):
            if len(neighbors[i][0]) > 0:
                indices = neighbors[i][0]
                times = neighbors[i][1]
                
                # Sort by time (most recent first)
                sorted_pairs = sorted(zip(times, indices), reverse=True)[:N]
                
                if sorted_pairs:
                    times, indices = zip(*sorted_pairs)
                    neighbors[i] = (
                        torch.tensor(indices, device=edge_index.device),
                        torch.tensor(times, device=edge_index.device, dtype=torch.float)
                    )
                else:
                    neighbors[i] = (torch.tensor([], device=edge_index.device), 
                                  torch.tensor([], device=edge_index.device))
            else:
                neighbors[i] = (torch.tensor([], device=edge_index.device), 
                              torch.tensor([], device=edge_index.device))
        
        return neighbors


class MemoryTGNN(nn.Module):
    """
    Memory-based Temporal Graph Neural Network following the paper exactly.
    Encoder-decoder architecture with message, memory, and embedding modules.
    """
    
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=48, 
                 num_heads=2, num_layers=2, dropout=0.2, device=None):
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Time encoder
        self.time_encoder = TimeEncoder(hidden_dim)
        
        # Message function (projects concatenation to hidden_dim)
        self.message_function = MessageFunction(hidden_dim, edge_feature_dim, hidden_dim)
        
        # Memory module (GRU with matching dimensions)
        self.memory_module = MemoryModule(hidden_dim)
        
        # Embedding module (temporal graph attention)
        self.embedding_module = TemporalGraphAttention(
            hidden_dim, num_heads, num_layers, dropout
        )
        
        # Decoder (two-layer MLP + sigmoid)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Node memories (initialized to zero)
        self.node_memory = None
        
        # Track last event time for each node
        self.last_event_time = None
        
        self._reset_parameters()
        self.to(self.device)
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def reset_memory(self, num_nodes):
        """Initialize/reset node memories to zero."""
        self.node_memory = torch.zeros(
            num_nodes, self.hidden_dim, device=self.device
        )
        self.last_event_time = torch.zeros(num_nodes, device=self.device)
    
    def process_event_batch(self, node_features, edge_index, edge_attr, 
                           edge_times, current_time):
        """
        Process a batch of events (temporal batch from paper).
        
        Args:
            node_features: X_i(t) for all nodes
            edge_index: edge indices in batch
            edge_attr: edge features
            edge_times: timestamp for each edge
            current_time: current timestamp
            
        Returns:
            updated embeddings and memory
        """
        num_nodes = node_features.size(0)
        
        # Initialize memory if needed
        if self.node_memory is None:
            self.reset_memory(num_nodes)
        
        # Encode node features
        x_encoded = self.node_encoder(node_features)
        
        # MESSAGE MODULE: Compute messages for all events
        messages = torch.zeros(num_nodes, self.node_memory.size(1), device=self.device)
        message_counts = torch.zeros(num_nodes, device=self.device)
        
        if edge_index.numel() > 0:
            for idx in range(edge_index.size(1)):
                i = edge_index[0, idx].item()
                j = edge_index[1, idx].item()
                
                # Get edge features
                e_ij = edge_attr[idx:idx+1] if edge_attr is not None else torch.zeros(1, 1, device=self.device)
                
                # Time encoding ψ(t - t'_i) and ψ(t - t'_j)
                time_diff_i = current_time - self.last_event_time[i]
                time_diff_j = current_time - self.last_event_time[j]
                time_enc_i = self.time_encoder(time_diff_i.unsqueeze(0))
                time_enc_j = self.time_encoder(time_diff_j.unsqueeze(0))
                
                # Compute messages for both nodes (Equation 1)
                m_i = self.message_function(
                    self.node_memory[i:i+1],
                    self.node_memory[j:j+1],
                    e_ij,
                    time_enc_i
                )
                
                m_j = self.message_function(
                    self.node_memory[j:j+1],
                    self.node_memory[i:i+1],
                    e_ij,
                    time_enc_j
                )
                
                # Recent message aggregator: keep most recent message
                messages[i] = m_i.squeeze(0)
                messages[j] = m_j.squeeze(0)
                message_counts[i] = 1
                message_counts[j] = 1
                
                # Update last event time
                self.last_event_time[i] = current_time
                self.last_event_time[j] = current_time
        
        # MEMORY MODULE: Update memory for nodes with messages (Equation 2)
        updated_memory = self.node_memory.clone()
        for i in range(num_nodes):
            if message_counts[i] > 0:
                updated_memory[i] = self.memory_module(
                    messages[i:i+1],
                    self.node_memory[i:i+1]
                ).squeeze(0)
        
        self.node_memory = updated_memory
        
        # EMBEDDING MODULE: Generate embeddings (Equation 3)
        # z_i^(0)(t) = s_i(t) + X_i(t)
        z_0 = self.node_memory + x_encoded
        
        # L-layer temporal graph attention
        z = self.embedding_module(
            z_0, edge_index, edge_times, current_time
        )
        
        # Normalize embeddings
        z = F.normalize(z, p=2, dim=1)
        
        return z, self.node_memory
    
    def forward(self, node_features, edge_index, edge_attr=None, 
                edge_times=None, current_time=0.0, pair_index=None, memory=None):
        """
        Forward pass following the paper's architecture.
        
        Args:
            node_features: node features X_i(t)
            edge_index: edge connectivity
            edge_attr: edge features
            edge_times: timestamp for each edge
            current_time: current timestamp
            pair_index: optional pairs to score
            memory: optional external memory state
            
        Returns:
            embeddings (and optionally pair scores)
        """
        node_features = node_features.to(self.device)
        num_nodes = node_features.size(0)
        
        # Use external memory if provided
        if memory is not None:
            self.node_memory = memory.to(self.device)
        elif self.node_memory is None:
            self.reset_memory(num_nodes)
        
        # Handle edge_times
        if edge_times is None and edge_index is not None and edge_index.numel() > 0:
            edge_times = torch.ones(edge_index.size(1), device=self.device) * current_time
        
        # Process event batch
        z, memory = self.process_event_batch(
            node_features, edge_index, edge_attr, edge_times, current_time
        )
        
        # DECODER: Score pairs if requested (Equation 4)
        if pair_index is not None:
            src_idx = pair_index[0].long().to(self.device)
            dst_idx = pair_index[1].long().to(self.device)
            
            # Concatenate embeddings
            pair_features = torch.cat([z[src_idx], z[dst_idx]], dim=1)
            
            # MLP decoder with sigmoid
            scores = self.decoder(pair_features)
            return torch.sigmoid(scores.view(-1)), memory
        
        return z, memory


@dataclass
class SelectorAgent:
    """
    Selector Agent using Memory-based TGNN for pairs trading.
    Follows Algorithm 1 and 2 from the paper.
    """
    
    df: pd.DataFrame
    logger: JSONLogger = None
    message_bus: MessageBus = None
    fundamentals: Optional[pd.DataFrame] = None
    model: Any = None
    scaler: Optional[MinMaxScaler] = None
    edge_index: Optional[Any] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    trace_path: str = None
    corr_threshold: float = 0.8
    holdout_years: int = 1
    node_features: Optional[pd.DataFrame] = None
    temporal_graphs: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    val_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    test_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    holdout_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    train_end_date: Optional[pd.Timestamp] = None

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
        if self.message_bus is None:
            self.message_bus = MessageBus()
        self._log_event("init", {"device": self.device, "corr_threshold": self.corr_threshold})

    def _log_event(self, event: str, details: Dict[str, Any]):
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": "selector",
            "event": event,
            "details": details,
        }
        if self.logger:
            self.logger.log("selector", event, details)
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        if self.message_bus:
            try:
                self.message_bus.publish(entry)
            except Exception:
                pass

    def _check_for_commands(self):
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

    def _save_checkpoint(self, epoch, loss, path=None):
        if path is None:
            directory = os.path.dirname(self.trace_path) or "."
            path = os.path.join(directory, "selector_checkpoint.pt")
        
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "model_state": self.model.state_dict() if self.model else None,
            "scaler": self.scaler
        }
        
        torch.save(checkpoint, path)
        self._log_event("checkpoint_saved", {
            "epoch": epoch,
            "loss": float(loss),
            "path": path
        })

    def build_node_features(self, windows=[5, 15, 30], train_end_date=None) -> pd.DataFrame:
        df = self.df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        
        if train_end_date is not None:
            self.train_end_date = pd.to_datetime(train_end_date)
        
        # Engineer features
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
        
        # One-hot encode sectors
        all_sectors = sorted(self.df["sector"].unique())
        encoder = OneHotEncoder(sparse_output=False, dtype=int, categories=[all_sectors])
        sector_encoded = encoder.fit_transform(df[["sector"]])
        sector_df = pd.DataFrame(sector_encoded, columns=encoder.get_feature_names_out())
        df = pd.concat([df.reset_index(drop=True), sector_df.reset_index(drop=True)], axis=1)
        df.drop(columns=["sector"], inplace=True, errors="ignore")
        
        df.fillna(0.0, inplace=True)
        
        exclude_cols = ["date", "ticker"]
        numeric_cols = [c for c in df.columns 
                       if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
        
        # Fit scaler only on training data
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            
            if train_end_date is not None:
                train_mask = df["date"] < self.train_end_date
                train_data = df.loc[train_mask, numeric_cols]
                self.scaler.fit(train_data)
                self._log_event("scaler_fit", {
                    "train_end_date": str(self.train_end_date),
                    "train_samples": len(train_data),
                    "total_samples": len(df)
                })
            else:
                self.scaler.fit(df[numeric_cols])
                self._log_event("scaler_fit_warning", {
                    "message": "Scaler fit on entire dataset - potential data leakage!",
                    "total_samples": len(df)
                })
        
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        self.node_features = df
        self._log_event("node_features_built", {
            "n_rows": len(df),
            "windows": windows,
            "train_end_date": str(self.train_end_date) if self.train_end_date else None
        })
        return df

    def build_temporal_graphs(self, corr_threshold: float = None, holdout_years: int = None):
        """
        Build temporal graphs following Algorithm 2 from the paper.
        """
        if corr_threshold is None:
            corr_threshold = self.corr_threshold
        if holdout_years is None:
            holdout_years = self.holdout_years
        
        # Determine split dates
        df_dates = self.df.copy()
        df_dates["date"] = pd.to_datetime(df_dates["date"])
        
        last_date = df_dates["date"].max()
        holdout_start = last_date - pd.DateOffset(years=holdout_years)
        mid_point = holdout_start + pd.DateOffset(months=6)
        
        val_start = holdout_start
        val_end = mid_point - pd.DateOffset(days=1)
        test_start = mid_point
        test_end = last_date
        train_end = val_start - pd.DateOffset(days=1)
        
        self.val_period = (val_start, val_end)
        self.test_period = (test_start, test_end)
        self.holdout_period = (val_start, test_end)
        
        # Build features with proper train_end_date
        self.build_node_features(train_end_date=train_end)
        
        self.temporal_graphs = []
        df = self.node_features.copy()
        df["date"] = pd.to_datetime(df["date"])
        tickers = sorted(df["ticker"].unique().tolist())
        
        exclude_cols = ["date", "ticker", "close", "adj_factor", "split_factor", 
                       "div_amount", "volume"]
        feature_cols = [c for c in df.columns 
                       if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
        
        # Split data
        train_df = df[df["date"] < val_start].copy()
        val_df = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        
        # Add month column to training data
        train_df["month"] = train_df["date"].dt.to_period("M")
        unique_months = sorted(train_df["month"].unique())
        
        print(f"Building {len(unique_months)} monthly temporal graphs...")
        
        # Process each month (Algorithm 2)
        for month_idx, month_period in enumerate(unique_months):
            self._check_for_commands()
            
            month_data = train_df[train_df["month"] == month_period]
            
            if month_data.empty or len(month_data) < 10:
                continue
            
            # Aggregate features for the month
            monthly_features = month_data.groupby("ticker")[feature_cols].mean().fillna(0.0)
            monthly_features = monthly_features.reindex(tickers, fill_value=0.0)
            
            if len(monthly_features) < 2:
                continue
            
            # Compute correlation matrix S(P_i(Δ_k), P_j(Δ_k))
            corr_matrix = np.corrcoef(monthly_features.values)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Create edges where S >= γ (correlation threshold)
            edges = np.argwhere(np.abs(corr_matrix) >= corr_threshold)
            edges = edges[edges[:, 0] < edges[:, 1]]
            
            num_nodes = len(tickers)
            edges = edges[(edges[:, 0] < num_nodes) & (edges[:, 1] < num_nodes)]
            
            edge_index = torch.tensor(edges.T, dtype=torch.long, device=self.device)
            edge_attr = torch.tensor(
                [[corr_matrix[i, j]] for i, j in edges],
                dtype=torch.float,
                device=self.device
            )
            
            # Assign t = (k - 0.5) * δ as in the paper
            edge_times = torch.ones(len(edges), device=self.device) * month_idx
            
            self.temporal_graphs.append({
                "month": str(month_period),
                "start": str(month_period.start_time.date()),
                "end": str(month_period.end_time.date()),
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "edge_times": edge_times,
                "time": month_idx,
                "num_edges": len(edges)
            })
        
        train_df.drop(columns=["month"], inplace=True, errors="ignore")
        
        self._log_event("temporal_graphs_built", {
            "n_graphs": len(self.temporal_graphs),
            "frequency": "monthly",
            "train_end": str(train_end.date()),
            "val_period": (str(self.val_period[0].date()), str(self.val_period[1].date())),
            "test_period": (str(self.test_period[0].date()), str(self.test_period[1].date())),
            "corr_threshold": corr_threshold
        })
        
        avg_edges = np.mean([g["num_edges"] for g in self.temporal_graphs]) if self.temporal_graphs else 0
        
        print(f"✅ Temporal graphs built: {len(self.temporal_graphs)} monthly graphs")
        print(f"   Average edges per graph: {avg_edges:.1f}")
        print(f"   Training period: up to {train_end.date()}")
        print(f"   Validation: {self.val_period[0].date()} → {self.val_period[1].date()}")
        print(f"   Test: {self.test_period[0].date()} → {self.test_period[1].date()}")
        
        return self.temporal_graphs, val_df, test_df

    def train_tgn_temporal_batches(self, optimizer, batch_size=32, epochs=3, neg_sample_ratio=1):
        """
        Training procedure following Algorithm 1 from the paper.
        Uses lag-one temporal batching.
        """
        numeric_features = self.node_features.drop(
            columns=["date", "ticker"], errors="ignore"
        ).select_dtypes(include=[np.number])
        
        x = torch.from_numpy(numeric_features.values).float().to(self.device)
        
        tickers = sorted(self.node_features["ticker"].unique().tolist())
        num_nodes = len(tickers)
        
        bce_loss_fn = nn.BCELoss(reduction="mean")
        
        self._log_event("tgn_training_started", {
            "n_snapshots": len(self.temporal_graphs),
            "epochs": epochs
        })
        print(f"Starting TGNN training on {len(self.temporal_graphs)} temporal snapshots.")
        
        # Initialize memory
        self.model.reset_memory(num_nodes)
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        patience = 5
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_events = 0
            
            # Lag-one scheme: use B_{i-1} to predict B_i
            prev_graph = None
            
            for i, graph in enumerate(self.temporal_graphs):
                self._check_for_commands()
                optimizer.zero_grad()
                
                # Use previous batch to update memory and embeddings
                if prev_graph is not None:
                    with torch.no_grad():
                        _, memory = self.model(
                            x,
                            prev_graph["edge_index"],
                            prev_graph.get("edge_attr", None),
                            prev_graph.get("edge_times", None),
                            prev_graph.get("time", i - 1),
                            pair_index=None,
                            memory=self.model.node_memory
                        )
                        self.model.node_memory = memory.detach()
                
                # Current batch for prediction
                edge_index = graph["edge_index"]
                edge_attr = graph.get("edge_attr", None)
                edge_times = graph.get("edge_times", None)
                current_time = graph.get("time", i)
                
                if edge_index.numel() == 0:
                    prev_graph = graph
                    continue
                
                # Positive samples
                src = edge_index[0]
                dst = edge_index[1]
                E = src.size(0)
                total_events += E
                
                pos_pairs = torch.stack([src, dst], dim=0)
                
                # Get embeddings and score positive pairs
                pos_scores, memory = self.model(
                    x,
                    edge_index,
                    edge_attr,
                    edge_times,
                    current_time,
                    pair_index=pos_pairs,
                    memory=self.model.node_memory
                )
                
                pos_labels = torch.ones_like(pos_scores)
                
                # Negative sampling
                num_neg = int(E * neg_sample_ratio)
                if num_neg > 0:
                    # Random negative sampling
                    rand_idx = torch.randint(0, num_nodes, (num_neg,), device=self.device)
                    src_for_neg = src.repeat((neg_sample_ratio,))[:num_neg]
                    
                    # Ensure negative samples are different from source
                    mask_equal = rand_idx == src_for_neg
                    while mask_equal.any():
                        rand_idx[mask_equal] = torch.randint(
                            0, num_nodes, (mask_equal.sum().item(),), device=self.device
                        )
                        mask_equal = rand_idx == src_for_neg
                    
                    neg_pairs = torch.stack([src_for_neg, rand_idx], dim=0)
                    
                    # Score negative pairs
                    neg_scores, _ = self.model(
                        x,
                        edge_index,
                        edge_attr,
                        edge_times,
                        current_time,
                        pair_index=neg_pairs,
                        memory=memory.detach()
                    )
                    
                    neg_labels = torch.zeros_like(neg_scores)
                    
                    # Combine positive and negative
                    scores = torch.cat([pos_scores, neg_scores])
                    labels = torch.cat([pos_labels, neg_labels])
                else:
                    scores = pos_scores
                    labels = pos_labels
                
                # Loss (Equation 5)
                loss = bce_loss_fn(scores, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item() * (E + num_neg)
                
                # Update memory for next iteration
                self.model.node_memory = memory.detach()
                
                # Store as previous graph for lag-one scheme
                prev_graph = graph
            
            avg_loss = total_loss / max(total_events, 1)
            
            self._log_event("tgn_epoch_complete", {"epoch": epoch + 1, "avg_loss": avg_loss})
            print(f"Epoch {epoch+1}/{epochs} complete. Avg loss: {avg_loss:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self._save_checkpoint(epoch, avg_loss)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self._log_event("early_stopping", {"epoch": epoch + 1})
                    print(f"⛔ Early stopping triggered at epoch {epoch+1}")
                    break
        
        self._log_event("tgn_training_complete", {"epochs": epoch + 1})
        print("✅ TGNN training complete.")

    def score_all_pairs_holdout(self, use_validation=False):
        """
        Score all pairs in the holdout period (validation or test).
        
        Args:
            use_validation: If True, score validation period. If False, score test period.
            
        Returns:
            DataFrame with columns: x, y, score (sorted by score descending)
        """
        if self.node_features is None:
            raise ValueError("Node features not built. Call build_temporal_graphs() first.")
        
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        # Determine which period to score
        if use_validation:
            if self.val_period is None:
                raise ValueError("Validation period not set.")
            period_start, period_end = self.val_period
            period_name = "validation"
        else:
            if self.test_period is None:
                raise ValueError("Test period not set.")
            period_start, period_end = self.test_period
            period_name = "test"
        
        print(f"Scoring pairs on {period_name} period: {period_start.date()} → {period_end.date()}")
        
        # Get tickers
        tickers = sorted(self.node_features["ticker"].unique().tolist())
        num_nodes = len(tickers)
        
        # Filter data for holdout period
        df_holdout = self.node_features[
            (self.node_features["date"] >= period_start) &
            (self.node_features["date"] <= period_end)
        ].copy()
        
        # Aggregate features for the holdout period
        exclude_cols = ["date", "ticker"]
        feature_cols = [c for c in df_holdout.columns 
                       if c not in exclude_cols and np.issubdtype(df_holdout[c].dtype, np.number)]
        
        holdout_features = df_holdout.groupby("ticker")[feature_cols].mean().fillna(0.0)
        holdout_features = holdout_features.reindex(tickers, fill_value=0.0)
        
        x_holdout = torch.from_numpy(holdout_features.values).float().to(self.device)
        
        # Set model to eval mode
        self.model.eval()
        
        # Reset memory to state after training
        # (In practice, you'd want to replay training graphs to get final memory state)
        self.model.reset_memory(num_nodes)
        
        # Replay all training graphs to get proper memory state
        print("Replaying training graphs to initialize memory...")
        numeric_features = self.node_features.drop(
            columns=["date", "ticker"], errors="ignore"
        ).select_dtypes(include=[np.number])
        x_train = torch.from_numpy(numeric_features.values).float().to(self.device)
        
        with torch.no_grad():
            for i, graph in enumerate(self.temporal_graphs):
                _, memory = self.model(
                    x_train,
                    graph["edge_index"],
                    graph.get("edge_attr", None),
                    graph.get("edge_times", None),
                    graph.get("time", i),
                    pair_index=None,
                    memory=self.model.node_memory
                )
                self.model.node_memory = memory.detach()
        
        print("Memory state initialized. Scoring all pairs...")
        
        # Now score all pairs using holdout features
        with torch.no_grad():
            # Generate all possible pairs
            src_idx, dst_idx = np.triu_indices(num_nodes, k=1)
            
            # Process in batches to avoid memory issues
            batch_size = 10000
            all_scores = []
            
            for start in range(0, len(src_idx), batch_size):
                end = min(start + batch_size, len(src_idx))
                
                batch_src = torch.tensor(src_idx[start:end], device=self.device)
                batch_dst = torch.tensor(dst_idx[start:end], device=self.device)
                pair_index = torch.stack([batch_src, batch_dst], dim=0)
                
                # Score this batch of pairs
                # Use no edges (just final embeddings from memory)
                scores, _ = self.model(
                    x_holdout,
                    edge_index=None,  # No edges in holdout
                    edge_attr=None,
                    edge_times=None,
                    current_time=len(self.temporal_graphs),  # Time after training
                    pair_index=pair_index,
                    memory=self.model.node_memory
                )
                
                all_scores.append(scores.cpu().numpy())
        
        # Combine all scores
        all_scores = np.concatenate(all_scores)
        
        # Create results DataFrame
        results = pd.DataFrame({
            "x": np.array(tickers)[src_idx],
            "y": np.array(tickers)[dst_idx],
            "score": all_scores
        }).sort_values("score", ascending=False).reset_index(drop=True)
        
        topk = results.head(10).to_dict(orient="records")
        self._log_event("pairs_scored_holdout", {
            "period": period_name,
            "n_pairs": len(results),
            "top_10": topk
        })
        
        print(f"✅ Scored {len(results)} pairs on {period_name} period")
        print(f"   Top score: {results.iloc[0]['score']:.4f} ({results.iloc[0]['x']}-{results.iloc[0]['y']})")
        print(f"   Mean score: {results['score'].mean():.4f}")
        
        return results
            "
