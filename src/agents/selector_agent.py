"""
Selector Agent for identifying cointegrated stock pairs using Temporal Graph Neural Networks.

- Uses a TGNN (Temporal Graph Neural Network) that combines message passing on graphs (spatial)
  with a recurrent memory mechanism (temporal).
- Supports multiple GAT (Graph Attention) layers and multi-head attention.
- Produces candidate stock pairs based on learned relationships in time-evolving correlation graphs.
"""

import os
import json
import datetime
import sys
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

# Import internal utilities (adjust path to your project layout)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CONFIG  # may be unused, kept for compatibility
from utils import half_life as compute_half_life, compute_spread
from agents.message_bus import MessageBus, JSONLogger


# ============================================================
# TGNN MODEL: Spatial-Temporal Graph Representation Learner
# ============================================================
class MemoryTGNN(nn.Module):
    """
    A Temporal Graph Neural Network with learnable memory per node.

    Core idea:
    - Each node (stock) has a hidden state (memory) that evolves over time via GRU updates.
    - At each time step, we run Graph Attention layers to propagate information across correlated stocks.
    - Multiple heads/layers allow richer local and multi-hop dependencies to be captured.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_heads: int = 3,
        num_layers: int = 3,
        dropout: float = 0.1,
        blend_factor: float = 0.3,
        device: Optional[str] = None,
        time_dim: int = 1,
        concat_heads: bool = True,
    ):
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_layers = max(1, int(num_layers))
        self.dropout_p = float(dropout)
        self.blend_factor = blend_factor
        self.concat_heads = concat_heads  # If False → average attention heads

        # ---------------- Input projection ----------------
        # Maps input features to a stable latent dimension for graph operations
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_channels),
        )

        # ---------------- Temporal (memory) component ----------------
        # Each node keeps a temporal memory, updated by a GRUCell.
        self.msg_gru = nn.GRUCell(hidden_channels, hidden_channels)
        # Update gate controls blending of old and new memory states.
        self.update_gate = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.Sigmoid(),
        )
        # Optional projection of timestamp embeddings into latent space.
        self.time_proj = nn.Linear(time_dim, hidden_channels)

        # ---------------- Spatial (graph attention) component ----------------
        # Stacks multiple GATConv layers to propagate messages between correlated nodes.
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_layers):
            if self.concat_heads:
                # When concatenating heads, set per-head output so heads*head_out = hidden_channels
                out_per_head = max(1, hidden_channels // self.num_heads)
                self.convs.append(
                    GATConv(hidden_channels, out_per_head, heads=self.num_heads, concat=True)
                )
                self.norms.append(nn.LayerNorm(out_per_head * self.num_heads))
            else:
                self.convs.append(
                    GATConv(hidden_channels, hidden_channels, heads=self.num_heads, concat=False)
                )
                self.norms.append(nn.LayerNorm(hidden_channels))

        self.dropout = nn.Dropout(p=self.dropout_p)

        # ---------------- Output projection ----------------
        # Normalizes and stabilizes node embeddings before scoring.
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_channels),
        )

        # Decoder scores pairs of nodes (e.g. how “related” or cointegrated they are)
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(hidden_channels, 1),
        )

        self.node_memory = None
        self._reset_parameters()
        self.to(self.device)

    # Initialize weights with Xavier uniform (standard for GNNs)
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ---------------- Forward pass ----------------
    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor], pair_index: Optional[torch.Tensor] = None, memory: Optional[torch.Tensor] = None, timestamps: Optional[torch.Tensor] = None):
        """
        Executes one temporal graph step.

        Args:
            x: (N × F) node feature matrix
            edge_index: (2 × E) tensor defining graph connectivity (or None / empty)
            pair_index: optional, if scoring specific node pairs; shape (2, n_pairs)
            memory: previous node memories (N × hidden)
            timestamps: optional temporal encoding (N × time_dim)

        Returns:
            If pair_index provided -> logits for pairs (1D tensor)
            Else -> (z, memory): node embeddings and updated memory
        """

        x = x.to(self.device)
        N = x.size(0)
        edge_index = edge_index.to(self.device) if (edge_index is not None and isinstance(edge_index, torch.Tensor)) else None

        # Initialize or fix memory shape
        if memory is None or memory.shape != (N, self.hidden_channels):
            memory = torch.zeros(N, self.hidden_channels, device=self.device)

        # Input + time projection
        x_proj = self.input_proj(x)
        if timestamps is not None:
            tproj = torch.tanh(self.time_proj(timestamps.to(self.device)))
            # Broadcast if needed
            if tproj.shape == x_proj.shape:
                x_proj = x_proj + tproj
            else:
                x_proj = x_proj + tproj

        # Temporal update using GRU and learnable gate
        new_mem = self.msg_gru(x_proj, memory)
        gate_in = torch.cat([memory, new_mem], dim=1)
        gate = self.update_gate(gate_in)
        memory = (1.0 - gate) * memory + gate * new_mem

        # Graph message passing across layers (spatial component)
        h = memory
        for conv, norm in zip(self.convs, self.norms):
            if edge_index is not None and edge_index.numel() > 0:
                h_conv = conv(h, edge_index)
            else:
                h_conv = h
            # Residual connection + normalization
            h = norm(h_conv + self.dropout(F.relu(h_conv)))

        self.node_memory = memory

        # Normalize embeddings
        z = F.normalize(self.proj_head(h), p=2, dim=1)

        # Optional: if pair indices provided, directly score relationships
        if pair_index is not None:
            src_idx = pair_index[0].long().to(self.device)
            dst_idx = pair_index[1].long().to(self.device)
            pair_embed = torch.cat([z[src_idx], z[dst_idx]], dim=1)
            logits = self.decoder(pair_embed).view(-1)
            return logits

        return z, memory

    def model_memory(self):
        """Returns latest node memory."""
        return self.node_memory


# ============================================================
# SELECTOR AGENT: Controls training, graph building, and validation
# ============================================================
@dataclass
class SelectorAgent:
    """
    High-level agent that:
    - Builds node features and temporal graphs from raw price data.
    - Trains the TGNN model to learn structural patterns.
    - Scores and validates stock pairs for potential cointegration.
    - Logs all actions and responds to commands from a message bus.
    """

    df: pd.DataFrame
    trace_path: Optional[str] = "trace.log"
    logger: Optional[JSONLogger] = None
    message_bus: Optional[MessageBus] = None
    fundamentals: Optional[pd.DataFrame] = None
    model: Optional[Any] = None
    scaler: Optional[MinMaxScaler] = None
    edge_index: Optional[Any] = None
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    corr_threshold: float = 0.6
    holdout_years: int = 1
    node_features: Optional[pd.DataFrame] = None
    temporal_graphs: List[Dict[str, Any]] = field(default_factory=list)
    val_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    test_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    holdout_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

    def __post_init__(self):
        dirpath = os.path.dirname(self.trace_path) if self.trace_path else "."
        os.makedirs(dirpath or ".", exist_ok=True)
        if self.message_bus is None:
            try:
                self.message_bus = MessageBus()
            except Exception:
                self.message_bus = None
        self._log_event("init", {"device": self.device, "corr_threshold": self.corr_threshold})

    def _log_event(self, event: str, details: Dict[str, Any]):
        """Writes structured JSON logs to file and message bus."""
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": "selector",
            "event": event,
            "details": details,
        }
        if self.logger:
            try:
                self.logger.log("selector", event, details)
            except Exception:
                pass
        if self.trace_path:
            try:
                with open(self.trace_path, "a") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            except Exception:
                # ignore logging file errors
                pass
        if self.message_bus:
            try:
                self.message_bus.publish(entry)
            except Exception:
                pass

    def _check_for_commands(self):
        """Placeholder to consume commands from message bus. Extend as needed."""
        # If you have a real message bus with commands, poll it here.
        return

    # ---------------- Data preparation: node feature builder ----------------
    def build_node_features(self, windows=[5, 15, 30]) -> pd.DataFrame:
        """
        Generates rolling statistical features for each ticker.
        - Creates mean, std, and cumulative log returns for given windows.
        - Adds sector encodings and scales numerics.
        - Performs time-based train/val/test split (avoiding leakage).
        """
        df = self.df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

        # Rolling-window features
        for window in windows:
            df[f"mean_{window}"] = df.groupby("ticker")["adj_close"].transform(lambda x: x.rolling(window).mean())
            df[f"std_{window}"] = df.groupby("ticker")["adj_close"].transform(lambda x: x.rolling(window).std())
            df[f"cum_return_{window}"] = df.groupby("ticker")["adj_close"].transform(
                lambda x: np.log(x / x.shift(1)).rolling(window).sum()
            )

        # Fill and encode
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df["eps_yoy_growth"] = df.get("eps_yoy_growth", 0.0).fillna(0.0)
        df["peg_adj"] = df.get("peg_adj", 0.0).fillna(0.0)

        # Sector encoding (OneHot)
        if "sector" in df.columns:
            all_sectors = sorted(df["sector"].fillna("OTHER").unique())
            encoder = OneHotEncoder(sparse_output=False, dtype=int, categories=[all_sectors])
            sector_df = pd.DataFrame(encoder.fit_transform(df[["sector"]].fillna("OTHER")),
                                     columns=encoder.get_feature_names_out(["sector"]))
            df = pd.concat([df.reset_index(drop=True), sector_df.reset_index(drop=True)], axis=1)
            df.drop(columns=["sector"], inplace=True, errors="ignore")

        df.fillna(0.0, inplace=True)

        # Scaling (fit only on train segment to avoid leakage)
        exclude_cols = ["date", "ticker"]
        numeric_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

        df["date"] = pd.to_datetime(df["date"])
        last_date = df["date"].max()
        holdout_start = last_date - pd.DateOffset(years=self.holdout_years)
        mid_point = holdout_start + pd.DateOffset(months=6)
        val_start, val_end = holdout_start, mid_point - pd.DateOffset(days=1)
        test_start, test_end = mid_point, last_date

        train_df = df[df["date"] < val_start]
        val_df = df[(df["date"] >= val_start) & (df["date"] <= val_end)]
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

        if len(train_df) == 0:
            # fallback: use entire df to fit scaler if no train split
            self.scaler = MinMaxScaler().fit(df[numeric_cols])
        else:
            self.scaler = MinMaxScaler().fit(train_df[numeric_cols])

        for part in (train_df, val_df, test_df):
            if not part.empty:
                part.loc[:, numeric_cols] = self.scaler.transform(part[numeric_cols])

        # Merge and save
        self.node_features = pd.concat([train_df, val_df, test_df], ignore_index=True)
        self.val_period, self.test_period = (val_start, val_end), (test_start, test_end)
        self.holdout_period = (val_start, test_end)
        self._log_event("node_features_built", {"n_rows": len(df), "windows": windows})
        return self.node_features

    # ---------------- Graph builder ----------------
    def build_temporal_graphs(self, corr_threshold: Optional[float] = None, holdout_years: Optional[int] = None):
        """
        Builds a sequence of weekly correlation graphs.
        - Each node = stock (ticker)
        - Edge = correlation ≥ threshold between their feature vectors
        """
        corr_threshold = corr_threshold or self.corr_threshold
        holdout_years = holdout_years or self.holdout_years

        self.temporal_graphs = []
        df = self.node_features.copy()
        df["date"] = pd.to_datetime(df["date"])
        tickers = df["ticker"].unique().tolist()

        exclude_cols = ["date", "ticker", "close", "adj_factor", "split_factor", "div_amount", "volume"]
        feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
        last_date = df["date"].max()
        holdout_start = last_date - pd.DateOffset(years=holdout_years)
        train_df = df[df["date"] < holdout_start]

        if train_df.empty:
            self._log_event("temporal_graphs_built", {"n_graphs": 0})
            print("⚠️ No training data before holdout start; temporal graphs empty.")
            return self.temporal_graphs

        weeks = sorted(train_df["date"].dt.to_period("W").unique())

        # Build graph for each week interval (week i to week i+1)
        for i in range(len(weeks) - 1):
            self._check_for_commands()
            start_week, end_week = weeks[i], weeks[i + 1]
            mask = (train_df["date"].dt.to_period("W") >= start_week) & (train_df["date"].dt.to_period("W") <= end_week)
            interval = train_df.loc[mask]
            if interval.empty:
                continue

            weekly_features = interval.groupby("ticker")[feature_cols].mean().fillna(0.0)
            weekly_features = weekly_features.reindex(tickers, fill_value=0.0)
            x_t = torch.from_numpy(weekly_features.values).float().to(self.device)

            corr_matrix = np.corrcoef(weekly_features.values)
            corr_matrix = np.nan_to_num(corr_matrix)
            # edges are pairs (i, j) where abs(corr) >= threshold
            edges = np.argwhere(np.abs(corr_matrix) >= corr_threshold)
            # keep upper triangle pairs (i < j)
            edges = edges[edges[:, 0] < edges[:, 1]]
            if len(edges) > 0:
                edge_index = torch.tensor(edges.T, dtype=torch.long, device=self.device)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

            self.temporal_graphs.append({"start": str(start_week), "end": str(end_week), "x": x_t, "edge_index": edge_index})

        self._log_event("temporal_graphs_built", {"n_graphs": len(self.temporal_graphs)})
        print(f"✅ Temporal graphs built: {len(self.temporal_graphs)}")
        return self.temporal_graphs

    # ---------------- Model training ----------------
    def train_tgn_temporal_batches(self, optimizer, epochs: int = 3, neg_sample_ratio: int = 1):
        """
        Trains the TGNN across temporal snapshots using edge prediction.
        - For each weekly graph, positive samples = existing edges, negatives = random pairs.
        - Uses binary cross-entropy loss to distinguish true vs. random connections.
        - Gradients are propagated through both the GAT layers and GRU memory.
        """
        if not self.temporal_graphs:
            raise ValueError("No temporal graphs. Call build_temporal_graphs() first.")

        if self.model is None:
            raise ValueError("No model attached. Assign a MemoryTGNN to `self.model` before training.")

        print(f"Training TGNN ({self.model.num_layers} layers, {self.model.num_heads} heads)")
        bce_loss = nn.BCEWithLogitsLoss()
        memory, best_loss = None, float("inf")
        patience, patience_counter = 5, 0

        for epoch in range(epochs):
            self.model.train()
            total_loss, total_samples = 0.0, 0

            for g in self.temporal_graphs:
                self._check_for_commands()
                x_t, edge_index = g["x"], g["edge_index"]
                optimizer.zero_grad()

                # Forward pass
                z, memory = self.model(x_t, edge_index, memory=memory)
                # Detach memory so backprop through time per epoch is controlled
                memory = memory.detach()

                if edge_index.numel() == 0:
                    continue

                # Positive edges
                src, dst = edge_index
                pos_pairs = torch.cat([z[src], z[dst]], dim=1)
                logits_pos = self.model.decoder(pos_pairs).squeeze()
                labels_pos = torch.ones_like(logits_pos)

                # Negative edges (random pairs)
                num_pos = logits_pos.numel()
                num_neg = int(num_pos * neg_sample_ratio)
                if num_neg > 0:
                    rand_src = torch.randint(0, z.size(0), (num_neg,), device=self.device)
                    rand_dst = torch.randint(0, z.size(0), (num_neg,), device=self.device)
                    neg_pairs = torch.cat([z[rand_src], z[rand_dst]], dim=1)
                    logits_neg = self.model.decoder(neg_pairs).squeeze()
                    labels_neg = torch.zeros_like(logits_neg)
                    logits = torch.cat([logits_pos, logits_neg])
                    labels = torch.cat([labels_pos, labels_neg])
                else:
                    logits, labels = logits_pos, labels_pos

                # Loss and optimization
                loss = bce_loss(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

            avg_loss = total_loss / max(total_samples, 1)
            print(f"Epoch {epoch+1}/{epochs} | Avg loss: {avg_loss:.6f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss, patience_counter = avg_loss, 0
                try:
                    self._save_checkpoint(epoch, avg_loss)
                except Exception:
                    pass
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⛔ Early stopping at epoch {epoch+1}")
                    break

        print("✅ TGNN training complete.")

    def _save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint if trace_path provided. Customize as needed."""
        if not self.trace_path:
            return
        try:
            ckpt = {
                "epoch": int(epoch),
                "loss": float(loss),
                "model_state_dict": self.model.state_dict() if self.model is not None else None,
            }
            ckpt_path = os.path.join(os.path.dirname(self.trace_path) or ".", f"tgn_checkpoint_epoch{epoch+1}.pt")
            torch.save(ckpt, ckpt_path)
            self._log_event("checkpoint_saved", {"path": ckpt_path, "epoch": epoch + 1, "loss": loss})
        except Exception:
            pass

    # ---------------- Scoring ----------------
    def score_all_pairs_holdout(self) -> pd.DataFrame:
        """
        Computes pair scores for the final (holdout) graph snapshot.
        - Uses learned embeddings from the last TGNN state.
        - Evaluates every possible stock pair.
        """
        if self.model is None:
            print("Warning: No model attached. Cannot score pairs.")
            return pd.DataFrame()

        if not self.temporal_graphs:
            print("Warning: No temporal graphs. Cannot score pairs.")
            return pd.DataFrame()

        tickers = self.node_features["ticker"].unique().tolist()
        num_nodes = len(tickers)
        x_t = self.temporal_graphs[-1]["x"]
        edge_index = self.temporal_graphs[-1]["edge_index"]

        self.model.eval()
        with torch.no_grad():
            z, _ = self.model(x_t, edge_index)
            z = F.normalize(z, p=2, dim=1)
            src_idx, dst_idx = np.triu_indices(num_nodes, k=1)
            # Build pair embeddings in batches if very large (here kept simple)
            pair_embed = torch.cat([z[src_idx], z[dst_idx]], dim=1)
            scores = self.model.decoder(pair_embed).cpu().numpy().flatten()

        results = pd.DataFrame({
            "x": np.array(tickers)[src_idx],
            "y": np.array(tickers)[dst_idx],
            "score": scores,
        }).sort_values("score", ascending=False).reset_index(drop=True)

        self._log_event("pairs_scored", {"n_pairs": len(results)})
        return results
