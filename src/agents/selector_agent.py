"""
Selector Agent for identifying cointegrated stock pairs using Temporal Graph Neural Networks.
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

class MemoryTGNN(nn.Module):
    """
    Efficient TGNN with simplified but preserved edge memory.
    
    Key optimizations:
    - Single GAT layer
    - Simplified edge memory update (no complex projections)
    - Batch edge memory updates
    - Smaller hidden dimensions
    """
    
    def __init__(self, in_channels, hidden_channels=48, num_heads=2, 
                 dropout=0.2, device=None):

        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout_p = dropout

        # Simplified input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Node memory (tracks each stock's temporal evolution)
        self.node_gru = nn.GRUCell(hidden_channels, hidden_channels)
        
        # Simplified edge memory projection
        # Instead of complex projection, just average the two node embeddings
        self.edge_transform = nn.Linear(hidden_channels, hidden_channels)
        self.edge_gru = nn.GRUCell(hidden_channels, hidden_channels)

        # Single GAT layer (down from 3)
        out_per_head = max(1, hidden_channels // num_heads)
        self.gat = GATConv(
            hidden_channels, 
            out_per_head, 
            heads=num_heads, 
            concat=True,
            dropout=dropout
        )
        
        # Simplified decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

        # Edge memory storage (simplified)
        self.node_memory = None
        self.edge_memory_dict = {}
        
        self._reset_parameters()
        self.to(self.device)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_edge_features(self, edge_index, node_embeddings):
        """
        Fast edge feature creation by averaging node embeddings.
        Much simpler than original concatenate + project approach.
        """
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        # Simple average of connected nodes (faster than concat + project)
        edge_features = (node_embeddings[src_idx] + node_embeddings[dst_idx]) / 2
        edge_features = self.edge_transform(edge_features)
        
        return edge_features

    def update_edge_memory(self, edge_index, edge_features):
        """
        Batched edge memory update - much faster than loop.
        
        Key optimization: Process all edges at once, then store individually.
        """
        num_edges = edge_index.size(1)
        
        # Collect previous memories in batch
        prev_memories = []
        edge_keys = []
        
        for i in range(num_edges):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_key = (min(src, dst), max(src, dst))
            edge_keys.append(edge_key)
            
            if edge_key in self.edge_memory_dict:
                prev_memories.append(self.edge_memory_dict[edge_key])
            else:
                prev_memories.append(torch.zeros(self.hidden_channels, device=self.device))
        
        # Batch GRU update (much faster than loop)
        prev_memories_batch = torch.stack(prev_memories)
        new_memories_batch = self.edge_gru(edge_features, prev_memories_batch)
        
        # Store updated memories
        for i, edge_key in enumerate(edge_keys):
            self.edge_memory_dict[edge_key] = new_memories_batch[i]
        
        return new_memories_batch

    def forward(self, x, edge_index, edge_attr=None, pair_index=None, memory=None):
        """
        Forward pass with efficient edge memory.
        """
        x = x.to(self.device)
        N = x.size(0)
        
        # Encode input
        h = self.input_encoder(x)

        # Update node memory
        if memory is None:
            memory = torch.zeros(N, self.hidden_channels, device=self.device)
        else:
            memory = memory.to(self.device)
        
        memory = self.node_gru(h, memory)
        h = memory

        # Process edges with memory (if edges exist)
        if edge_index is not None and edge_index.numel() > 0:
            edge_index = edge_index.to(self.device)
            
            # Create edge features (simplified approach)
            edge_features = self.create_edge_features_fast(edge_index, h)
            
            # Update edge memory (batched for speed)
            edge_features = self.update_edge_memory_batch(edge_index, edge_features)
            
            # Single GAT layer with edge features
            h = self.gat(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)

        # Normalize embeddings
        z = F.normalize(h, p=2, dim=1)
        
        self.node_memory = memory

        # Score pairs if requested
        if pair_index is not None:
            src_idx = pair_index[0].long().to(self.device)
            dst_idx = pair_index[1].long().to(self.device)
            
            # Use edge memory for scoring if available
            pair_scores = []
            for i in range(len(src_idx)):
                src, dst = src_idx[i].item(), dst_idx[i].item()
                edge_key = (min(src, dst), max(src, dst))
                
                # Concatenate node embeddings
                pair_features = torch.cat([z[src_idx[i:i+1]], z[dst_idx[i:i+1]]], dim=1)
                
                # If we have edge memory for this pair, incorporate it
                if edge_key in self.edge_memory_dict:
                    edge_mem = self.edge_memory_dict[edge_key]
                    # Blend edge memory with pair features (simple attention-like mechanism)
                    pair_features = pair_features + torch.cat([edge_mem, edge_mem], dim=0).unsqueeze(0)
                
                score = self.decoder(pair_features)
                pair_scores.append(score)
            
            scores = torch.cat(pair_scores, dim=0)
            return torch.sigmoid(scores.view(-1))

        return z, memory

    def reset_edge_memory(self):
        """Reset edge memory (call between epochs or training phases)"""
        self.edge_memory_dict = {}
    
    def prune_edge_memory(self, keep_top_k=1000):
        """
        Prune edge memory to keep only most important edges.
        Call this periodically if memory grows too large.
        """
        if len(self.edge_memory_dict) <= keep_top_k:
            return
        
        # Keep edges with highest memory norm (most active pairs)
        edge_norms = {k: v.norm().item() for k, v in self.edge_memory_dict.items()}
        top_edges = sorted(edge_norms.items(), key=lambda x: x[1], reverse=True)[:keep_top_k]
        
        self.edge_memory_dict = {k: self.edge_memory_dict[k] for k, _ in top_edges}
        
        print(f"Pruned edge memory: {len(edge_norms)} → {keep_top_k} edges")

@dataclass
class SelectorAgent:
    """
    SelectorAgent with temporal holdout for simulating future performance.
    
    Key Features:
    - Monthly temporal graphs
    - Memory-based TGNN for embeddings
    - Temporal holdout: train on first 4 years, score pairs on last year
    - GPU support if available
    - No data leakage: Scaler fit only on training data
    
    Improvements:
    - Integrates with MessageBus for supervisor commands
    - Traces important events to JSONL for audit
    - Periodically checks message bus during long operations
    - Fixed data leakage by splitting data BEFORE scaling
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
    train_end_date: Optional[pd.Timestamp] = None  # Track where training data ends

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
        """
        Saves a checkpoint with model weights and metadata.
        Called automatically by early stopping in training.
        """
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
        """
        Build node features WITHOUT data leakage.
        
        CRITICAL: Scaler is fit ONLY on training data (before train_end_date).
        If train_end_date is None, scaler is fit on entire dataset (use only for initial exploration).
        
        Args:
            windows: Rolling window sizes for feature engineering
            train_end_date: End date of training data. Scaler will be fit only on data before this date.
        """
        df = self.df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        
        # Store train end date for reference
        if train_end_date is not None:
            self.train_end_date = pd.to_datetime(train_end_date)
        
        # Engineer features
        for window in windows:
            df[f"mean_{window}"] = df.groupby("ticker")["adj_close"].transform(lambda x: x.rolling(window).mean())
            df[f"std_{window}"] = df.groupby("ticker")["adj_close"].transform(lambda x: x.rolling(window).std())
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
        numeric_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

        # FIT SCALER ONLY ON TRAINING DATA TO PREVENT DATA LEAKAGE
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            
            if train_end_date is not None:
                # Fit only on training data
                train_mask = df["date"] < self.train_end_date
                train_data = df.loc[train_mask, numeric_cols]
                self.scaler.fit(train_data)
                self._log_event("scaler_fit", {
                    "train_end_date": str(self.train_end_date),
                    "train_samples": len(train_data),
                    "total_samples": len(df)
                })
            else:
                # Fit on all data (only for initial exploration - not recommended)
                self.scaler.fit(df[numeric_cols])
                self._log_event("scaler_fit_warning", {
                    "message": "Scaler fit on entire dataset - potential data leakage!",
                    "total_samples": len(df)
                })
        
        # Transform all data using the fitted scaler
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
        Build temporal graphs with proper train/val/test split BEFORE feature scaling.
        Uses MONTHLY frequency to avoid period comparison errors.
        """
        if corr_threshold is None:
            corr_threshold = self.corr_threshold
        else:
            self.corr_threshold = corr_threshold

        if holdout_years is None:
            holdout_years = self.holdout_years
        else:
            self.holdout_years = holdout_years

        # Determine split dates FIRST
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
        
        # Store periods
        self.val_period = (val_start, val_end)
        self.test_period = (test_start, test_end)
        self.holdout_period = (val_start, test_end)
        
        # NOW build features with proper train_end_date to prevent leakage
        self.build_node_features(train_end_date=train_end)
        
        self.temporal_graphs = []
        df = self.node_features.copy()
        df["date"] = pd.to_datetime(df["date"])
        tickers = df["ticker"].unique().tolist()

        exclude_cols = ["date", "ticker", "close", "adj_factor", "split_factor", "div_amount", "volume"]
        feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

        # Split data using datetime comparison (NOT period comparison!)
        train_df = df[df["date"] < val_start].copy()
        val_df = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

        # Add month column to training data AFTER splitting
        train_df["month"] = train_df["date"].dt.to_period("Y")
        
        # Get unique months sorted
        unique_months = sorted(train_df["month"].unique())
        
        print(f"Building {len(unique_months)} monthly temporal graphs from training data...")

        # Process each month
        for month_period in unique_months:
            self._check_for_commands()

            # Filter data for this specific month
            # Use the period column directly - no comparisons needed
            month_data = train_df[train_df["month"] == month_period]

            if month_data.empty or len(month_data) < 10:  # Skip months with too little data
                continue

            # Aggregate features for the month
            monthly_features = month_data.groupby("ticker")[feature_cols].mean().fillna(0.0)
            monthly_features = monthly_features.reindex(tickers, fill_value=0.0)

            # Skip if we don't have enough tickers
            if len(monthly_features) < 2:
                continue

            # Compute correlation matrix
            corr_matrix = np.corrcoef(monthly_features.values)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            # Create edges based on correlation threshold
            edges = np.argwhere(np.abs(corr_matrix) >= corr_threshold)
            edges = edges[edges[:, 0] < edges[:, 1]]  # Keep upper triangle only

            num_nodes = len(tickers)
            edges = edges[(edges[:, 0] < num_nodes) & (edges[:, 1] < num_nodes)]

            if len(edges) > 0:
                edge_index = torch.tensor(edges.T, dtype=torch.long, device=self.device)
                edge_attr = torch.tensor(
                    [[corr_matrix[i, j]] for i, j in edges], 
                    dtype=torch.float, 
                    device=self.device
                )
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                edge_attr = torch.zeros((0, 1), dtype=torch.float, device=self.device)

            # Store graph snapshot
            self.temporal_graphs.append({
                "month": str(month_period),
                "start": str(month_period.start_time.date()),
                "end": str(month_period.end_time.date()),
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "num_edges": len(edges)
            })

        # Clean up temporary column
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

            numeric_features = self.node_features.drop(columns=["date", "ticker"], errors="ignore").select_dtypes(include=[np.number])
            if numeric_features.empty:
                raise ValueError("No numeric columns found in node_features after filtering.")

            x = torch.from_numpy(numeric_features.values).float().to(self.device)

            tickers = self.node_features["ticker"].unique().tolist()
            num_nodes = len(tickers)
            ticker_to_idx = {t: i for i, t in enumerate(tickers)}

            decoder = getattr(self, "decoder", None)
            if decoder is None:
                if not hasattr(self, "_train_decoder"):
                    self._train_decoder = None
                decoder = getattr(self, "_train_decoder", None)

            bce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

            self._log_event("tgn_training_started", {"n_snapshots": len(self.temporal_graphs), "epochs": epochs})
            print(f"Starting TGNN training on {len(self.temporal_graphs)} temporal snapshots.")

            memory = None

            # Early stopping
            best_loss = float('inf')
            patience_counter = 0
            patience = 5

            for epoch in range(epochs):

                self.model.train()
                total_loss = 0.0
                total_events = 0

                for i, graph in enumerate(self.temporal_graphs):

                    self._check_for_commands()
                    optimizer.zero_grad()

                    edge_index = graph["edge_index"].detach().clone()
                    edge_attr = graph.get("edge_attr", None)
                    if edge_attr is not None:
                        edge_attr = edge_attr.detach().clone()

                    # Fix: Pass edge_attr as positional argument, not skipping it with keyword args
                    model_out = self.model(x, edge_index, edge_attr, pair_index=None, memory=memory)

                    if isinstance(model_out, tuple) and len(model_out) >= 2:
                        z, memory = model_out[0], model_out[1]
                    else:
                        z = model_out

                    if memory is not None:
                        memory = memory.detach()

                    if z is None:
                        raise RuntimeError("Model returned None embeddings.")
                    if z.dim() == 1:
                        raise RuntimeError("Embeddings must have shape (num_nodes, d).")

                    decoder = self.model.decoder

                    # Positive samples
                    if edge_index.numel() == 0:
                        continue

                    src = edge_index[0]
                    dst = edge_index[1]
                    E = src.size(0)
                    total_events += E

                    pos_cat = torch.cat([z[src], z[dst]], dim=1)
                    logits_pos = decoder(pos_cat).view(-1)
                    pos_labels = torch.ones_like(logits_pos)

                    # Negative sampling
                    num_neg = int(E * neg_sample_ratio)
                    if num_neg > 0:
                        rand_idx = torch.randint(0, num_nodes, (num_neg,), device=self.device)
                        src_for_neg = src.repeat((neg_sample_ratio,))[:num_neg]
                        mask_equal = rand_idx == src_for_neg
                        while mask_equal.any():
                            rand_idx[mask_equal] = torch.randint(0, num_nodes, (mask_equal.sum().item(),), device=self.device)
                            mask_equal = rand_idx == src_for_neg

                        neg_cat = torch.cat([z[src_for_neg], z[rand_idx]], dim=1)
                        logits_neg = decoder(neg_cat).view(-1)
                        neg_labels = torch.zeros_like(logits_neg)

                        logits = torch.cat([logits_pos, logits_neg])
                        labels = torch.cat([pos_labels, neg_labels])
                    else:
                        logits = logits_pos
                        labels = pos_labels

                    loss = bce_loss_fn(logits, labels)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item() * (E + (num_neg if num_neg > 0 else 0))

                avg_loss = total_loss / max(total_events, 1)

                self._log_event("tgn_epoch_complete", {"epoch": epoch + 1, "avg_loss": avg_loss})
                print(f"Epoch {epoch+1}/{epochs} complete. Avg (scaled) loss: {avg_loss:.6f}")

                # Early stopping check
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    self._save_checkpoint(epoch, avg_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self._log_event("early_stopping", {"epoch": epoch})
                        print(f"⛔ Early stopping triggered at epoch {epoch+1}")
                        break

            self._log_event("tgn_training_complete", {"epochs": epoch + 1})
            print("✅ TGNN training complete.")

    def score_all_pairs_holdout(self):
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

                out = self.model(x_full, g["edge_index"], g.get("edge_attr", None), pair_index=None, memory=memory)

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
