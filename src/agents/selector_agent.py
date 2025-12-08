import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Ensure config is loadable as per your setup
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config import CONFIG
except ImportError:
    CONFIG = {"random_seed": 42}

# ==============================================================================
# 1. ENHANCED MODEL ARCHITECTURE (Vectorized + Memory Support)
# ==============================================================================

class EnhancedTGNN(nn.Module):
    def __init__(self, node_dim, hidden_dim=64, num_heads=4, dropout=0.2):
        super().__init__()
        
        # 1. Input Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Memory Gate (GRU Cell) - Allows state to evolve over snapshots
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 3. Graph Attention Layers
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout, edge_dim=1)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout, edge_dim=1)
        self.bn2 = BatchNorm(hidden_dim)
        
        # 4. Vectorized Pair Scorer Parameters
        # We learn a matrix W such that Score(u, v) = u^T W v
        self.bilinear_W = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.bilinear_W)
        
        self.dropout = dropout
    
    def forward_snapshot(self, x, edge_index, edge_weight, hidden_state=None):
        """
        Processes one temporal snapshot.
        Returns: 
            h_out: Normalized embeddings for the current step
            new_hidden_state: Raw embeddings to pass to the next GRU step
        """
        # A. Encode
        h = self.node_encoder(x)
        
        # B. Memory Update
        if hidden_state is not None:
            h = self.gru(h, hidden_state)
        
        new_hidden_state = h.clone() # Save state before graph diffusion

        # C. Message Passing
        if edge_index.numel() > 0:
            # Layer 1
            h_in = h
            h = self.gat1(h, edge_index, edge_attr=edge_weight)
            h = self.bn1(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in
            
            # Layer 2
            h_in = h
            h = self.gat2(h, edge_index, edge_attr=edge_weight)
            h = self.bn2(h)
            h = F.relu(h)
            h = h + h_in

        # D. Normalize (Crucial for cosine similarity)
        h_out = F.normalize(h, p=2, dim=1)
        
        return h_out, new_hidden_state

    def compute_ranking_loss(self, embeddings, pos_pairs, neg_pairs):
        """
        Computes Margin Ranking Loss: Positive pairs should score higher than Negative pairs.
        """
        # Score Positive Pairs
        pos_src = embeddings[pos_pairs[0]]
        pos_dst = embeddings[pos_pairs[1]]
        pos_scores = self._score_vectors(pos_src, pos_dst)
        
        # Score Negative Pairs
        neg_src = embeddings[neg_pairs[0]]
        neg_dst = embeddings[neg_pairs[1]]
        neg_scores = self._score_vectors(neg_src, neg_dst)
        
        # Loss: max(0, margin - (pos - neg))
        # We want pos > neg, so we use target=1
        loss = F.margin_ranking_loss(
            pos_scores, 
            neg_scores, 
            target=torch.ones_like(pos_scores), 
            margin=0.3
        )
        return loss

    def _score_vectors(self, src, dst):
        """ Computes x_src * W * x_dst + Cosine(x_src, x_dst) """
        # Bilinear: sum((src @ W) * dst)
        bilinear = torch.sum((src @ self.bilinear_W) * dst, dim=1)
        cosine = F.cosine_similarity(src, dst)
        return bilinear + cosine

    def get_all_scores_matrix(self, embeddings):
        """
        Fully Vectorized Scoring for Inference.
        Returns (N, N) matrix where [i,j] is score between node i and j
        """
        # Bilinear Part: H @ W @ H.T
        weighted_emb = embeddings @ self.bilinear_W
        bilinear_scores = weighted_emb @ embeddings.T
        
        # Cosine Part: H @ H.T (since H is already normalized)
        cosine_scores = embeddings @ embeddings.T
        
        return bilinear_scores + cosine_scores

# ==============================================================================
# 2. OPTIMIZED AGENT (Target Shifting + Efficient Training)
# ==============================================================================

@dataclass
class OptimizedSelectorAgent:

    df: pd.DataFrame
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    trace_path: str = "traces/selector.jsonl"
    
    # Hyperparameters
    corr_threshold: float = 0.60
    lookback_weeks: int = 4
    forecast_horizon: int = 1  # 1 week ahead prediction
    holdout_months: int = 18
    hidden_dim: int = 64
    num_heads: int = 4
    
    # Internal State
    model: Any = None
    scaler: Optional[StandardScaler] = None
    industry_encoder: Optional[OneHotEncoder] = None
    tickers: Optional[List[str]] = None
    ticker_to_idx: Optional[Dict[str, int]] = None
    
    # Data Containers
    train_df: Optional[pd.DataFrame] = None
    val_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None
    node_features: Optional[pd.DataFrame] = None
    
    def __post_init__(self):
        os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
        self._log_event("init", {"device": self.device})
    
    def _log_event(self, event: str, details: Dict[str, Any]):
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": "selector",
            "event": event,
            "details": details,
        }
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    
    # ------------------------------------------------------------------------
    # Feature Engineering (Preserved)
    # ------------------------------------------------------------------------
    
    def build_node_features(self, windows=[1, 2, 4], train_end_date=None) -> pd.DataFrame:
        df = self.df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        
        df["returns"] = df.groupby("ticker")["adj_close"].pct_change()
        df["log_returns"] = np.log1p(df["returns"])
        
        for window in windows:
            days = window * 5
            df[f"volatility_{window}w"] = df.groupby("ticker")["returns"].transform(
                lambda x: x.rolling(days, min_periods=max(1, days//2)).std()
            )
            df[f"momentum_{window}w"] = df.groupby("ticker")["adj_close"].transform(
                lambda x: x.pct_change(days)
            )

        if "sector" in df.columns:
            if self.industry_encoder is None:
                self.industry_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                industry_encoded = self.industry_encoder.fit_transform(df[["sector"]])
            else:
                industry_encoded = self.industry_encoder.transform(df[["sector"]])
                
            ind_cols = [f"ind_{i}" for i in range(industry_encoded.shape[1])]
            ind_df = pd.DataFrame(industry_encoded, columns=ind_cols, index=df.index)
            df = pd.concat([df, ind_df], axis=1)
            df.drop(columns=["sector"], inplace=True, errors="ignore")
        
        df.fillna(0.0, inplace=True)
        
        # Scale
        exclude = ["date", "ticker", "close", "adj_factor", "split_factor", "volume", "adj_close"]
        numeric_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            if train_end_date:
                train_mask = df["date"] < pd.to_datetime(train_end_date)
                self.scaler.fit(df.loc[train_mask, numeric_cols])
            else:
                self.scaler.fit(df[numeric_cols])
        
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        self.node_features = df
        return df

    def prepare_data(self, train_end_date: str = None):
        if train_end_date is None:
            last_date = self.df["date"].max()
            train_end_date = last_date - pd.DateOffset(months=self.holdout_months)
        
        train_end = pd.to_datetime(train_end_date)
        self.build_node_features(train_end_date=train_end)
        
        df = self.node_features.copy()
        self.tickers = sorted(df["ticker"].unique())
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}
        
        last_date = df["date"].max()
        mid_point = train_end + (last_date - train_end) / 2
        
        self.train_df = df[df["date"] < train_end].copy()
        self.val_df = df[(df["date"] >= train_end) & (df["date"] < mid_point)].copy()
        self.test_df = df[df["date"] >= mid_point].copy()
        
        return self.train_df, self.val_df, self.test_df

    # ------------------------------------------------------------------------
    # Graph Construction (Shifted Targets)
    # ------------------------------------------------------------------------

    def build_shifted_snapshots(self, df: pd.DataFrame, window_days: int = 20, mode='train'):
        """
        Creates snapshots where:
        - Input Graph: Derived from features at Time T
        - Target Graph: Derived from Correlation at Time T + Horizon
        """
        df = df.sort_values('date')
        
        # Pivots
        returns_pivot = df.pivot(index='date', columns='ticker', values='log_returns').fillna(0)
        returns_pivot = returns_pivot.reindex(columns=self.tickers, fill_value=0)
        
        dates = returns_pivot.index
        snapshots = []
        
        # Step size
        step = 5 if mode == 'train' else 10
        
        # We need enough room for lookback AND forecast
        forecast_days = self.forecast_horizon * 5
        
        for i in range(window_days, len(dates) - forecast_days, step):
            current_date = dates[i]
            
            # --- 1. Construct INPUT Graph (Observation) ---
            start_idx = i - window_days
            past_returns = returns_pivot.iloc[start_idx : i+1]
            
            # Input Edges: Correlation of the immediate past
            corr_matrix = past_returns.corr().values
            corr_matrix = np.nan_to_num(corr_matrix)
            
            # Filter edges for GNN message passing
            input_edges = np.argwhere(np.abs(corr_matrix) >= 0.5) # Lower threshold for input to allow flow
            input_edges = input_edges[input_edges[:, 0] < input_edges[:, 1]]
            
            if len(input_edges) == 0:
                 edge_index = torch.empty((2, 0), dtype=torch.long)
                 edge_weights = torch.empty(0, dtype=torch.float)
            else:
                 edge_index = torch.tensor(input_edges.T, dtype=torch.long)
                 edge_weights = torch.tensor([corr_matrix[u,v] for u,v in input_edges], dtype=torch.float)

            # --- 2. Construct TARGET Graph (Prediction) ---
            # We want to predict if pairs will be correlated in the FUTURE
            target_start = i + 1
            target_end = i + 1 + forecast_days
            future_returns = returns_pivot.iloc[target_start : target_end]
            
            if len(future_returns) < 2: continue
            
            future_corr = future_returns.corr().values
            future_corr = np.nan_to_num(future_corr)
            
            # Positive samples: High future correlation
            pos_edges = np.argwhere(np.abs(future_corr) >= self.corr_threshold)
            pos_edges = pos_edges[pos_edges[:, 0] < pos_edges[:, 1]]
            
            snapshots.append({
                'date': current_date,
                'edge_index': edge_index, # Input graph topology
                'edge_weights': edge_weights, # Input graph weights
                'target_pos_pairs': torch.tensor(pos_edges.T, dtype=torch.long) # Future ground truth
            })
            
        return snapshots

    def create_snapshot_features(self, df: pd.DataFrame, snapshot_date):
        exclude = ["date", "ticker", "close", "adj_factor", "split_factor", "volume", "adj_close", "div_amount"]
        feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
        
        # Get features closest to snapshot date
        snapshot_df = df[df['date'] <= snapshot_date].groupby('ticker').tail(1)
        # Reindex to ensure fixed order matching self.tickers
        node_features = snapshot_df.set_index('ticker')[feature_cols].reindex(self.tickers).fillna(0.0)
        return torch.tensor(node_features.values, dtype=torch.float)

    # ------------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------------

    def train(self, epochs: int = 20, lr: float = 0.001):
        seed = CONFIG.get("random_seed", 42)
        torch.manual_seed(seed)
        
        if self.train_df is None: raise ValueError("Call prepare_data() first")

        # Initialize Model
        sample_feat = self.create_snapshot_features(self.train_df, self.train_df['date'].iloc[0])
        self.model = EnhancedTGNN(
            node_dim=sample_feat.shape[1],
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        print("Building temporal snapshots...")
        train_snapshots = self.build_shifted_snapshots(self.train_df, self.lookback_weeks * 5, mode='train')
        val_snapshots = self.build_shifted_snapshots(self.val_df, self.lookback_weeks * 5, mode='val')
        
        print(f"Training on {len(train_snapshots)} snapshots, Validating on {len(val_snapshots)}")
        
        best_val_loss = float('inf')
        patience = 6
        
        for epoch in range(epochs):
            # --- TRAIN ---
            self.model.train()
            train_loss = 0.0
            
            # Reset Memory at start of epoch (or use truncated BPTT)
            hidden_state = None 
            
            for snap in train_snapshots:
                x = self.create_snapshot_features(self.train_df, snap['date']).to(self.device)
                edge_index = snap['edge_index'].to(self.device)
                edge_weights = snap['edge_weights'].to(self.device)
                target_pos = snap['target_pos_pairs'].to(self.device)
                
                # 1. Forward Pass
                # Pass hidden_state.detach() to prevent gradients flowing all the way back to t=0
                # But keep flow within reasonable truncation if needed. Here we detach every step for stability.
                h_in = hidden_state.detach() if hidden_state is not None else None
                embeddings, hidden_state = self.model.forward_snapshot(x, edge_index, edge_weights, h_in)
                
                if target_pos.size(1) == 0: continue

                # 2. Negative Sampling (On the fly)
                num_pos = target_pos.size(1)
                neg_src = torch.randint(0, len(self.tickers), (num_pos,), device=self.device)
                neg_dst = torch.randint(0, len(self.tickers), (num_pos,), device=self.device)
                neg_pairs = torch.stack([neg_src, neg_dst], dim=0)
                
                # 3. Loss
                loss = self.model.compute_ranking_loss(embeddings, target_pos, neg_pairs)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_snapshots)
            
            # --- VALIDATE ---
            self.model.eval()
            val_loss = 0.0
            hidden_state = None
            
            with torch.no_grad():
                for snap in val_snapshots:
                    x = self.create_snapshot_features(self.val_df, snap['date']).to(self.device)
                    edge_index = snap['edge_index'].to(self.device)
                    edge_weights = snap['edge_weights'].to(self.device)
                    target_pos = snap['target_pos_pairs'].to(self.device)
                    
                    embeddings, hidden_state = self.model.forward_snapshot(x, edge_index, edge_weights, hidden_state)
                    
                    if target_pos.size(1) == 0: continue
                    
                    # Consistent validation negative sampling
                    neg_src = torch.randint(0, len(self.tickers), (target_pos.size(1),), device=self.device)
                    neg_dst = torch.randint(0, len(self.tickers), (target_pos.size(1),), device=self.device)
                    neg_pairs = torch.stack([neg_src, neg_dst], dim=0)
                    
                    loss = self.model.compute_ranking_loss(embeddings, target_pos, neg_pairs)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_snapshots)
            
            # Logging & Early Stopping
            status = "no_improvement"
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                status = "improved"
                patience = 6
                # Save checkpoint if needed
            else:
                patience -= 1
                
            print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | {status}")
            
            self._log_event("training_status", {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "status": status,
                "best_loss": best_val_loss,
                "patience_left": patience
            })
            
            if patience <= 0:
                print("Early stopping triggered.")
                break

    # ------------------------------------------------------------------------
    # Inference (Vectorized)
    # ------------------------------------------------------------------------

    def score_pairs(self, use_validation: bool = True, top_k: int = 100):
        if self.model is None: raise ValueError("Model not trained")
        
        df = self.val_df if use_validation else self.test_df
        period_name = "validation" if use_validation else "test"
        
        self.model.eval()
        print(f"\nScoring pairs on {period_name}...")
        
        # Build snapshots just to get the sequence right
        snapshots = self.build_shifted_snapshots(df, self.lookback_weeks * 5, mode='val')
        if not snapshots: return pd.DataFrame()
        
        # We need to run the sequence to build up memory, 
        # but for efficiency we can just take the last few or the very last one 
        # if we assume state is reset. 
        # Ideally, pass the state from training, but here we warm start:
        hidden_state = None
        
        # Process all snapshots to update memory
        with torch.no_grad():
            for snap in snapshots[:-1]:
                x = self.create_snapshot_features(df, snap['date']).to(self.device)
                idx = snap['edge_index'].to(self.device)
                w = snap['edge_weights'].to(self.device)
                _, hidden_state = self.model.forward_snapshot(x, idx, w, hidden_state)
            
            # Final scoring on the LAST snapshot
            last_snap = snapshots[-1]
            x = self.create_snapshot_features(df, last_snap['date']).to(self.device)
            idx = last_snap['edge_index'].to(self.device)
            w = last_snap['edge_weights'].to(self.device)
            
            embeddings, _ = self.model.forward_snapshot(x, idx, w, hidden_state)
            
            # --- Vectorized Scoring ---
            score_matrix = self.model.get_all_scores_matrix(embeddings)
            
            # Mask diagonal and lower triangle to avoid duplicates and self-loops
            mask = torch.triu(torch.ones_like(score_matrix), diagonal=1).bool()
            valid_scores = score_matrix[mask]
            
            # Get Top K indices
            # flatten indices
            flat_indices = torch.topk(valid_scores, k=min(top_k * 5, len(valid_scores))).indices
            
            # We need to map back to (row, col). 
            # Since we masked, it's tricky to map back directly from valid_scores.
            # Easier approach: Set invalid to -inf and topk the whole matrix
            score_matrix[~mask] = -float('inf')
            
            top_vals, top_flat_indices = torch.topk(score_matrix.flatten(), k=top_k)
            
            rows = top_flat_indices // score_matrix.size(1)
            cols = top_flat_indices % score_matrix.size(1)
            
            results = []
            for r, c, score in zip(rows.cpu().numpy(), cols.cpu().numpy(), top_vals.cpu().numpy()):
                results.append({
                    'x': self.tickers[r],
                    'y': self.tickers[c],
                    'score': float(score),
                    'date': last_snap['date']
                })
                
        results_df = pd.DataFrame(results)
        print(f"✅ Scored {len(results_df)} pairs.")
        return results_df
