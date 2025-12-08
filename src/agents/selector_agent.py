import os
import sys
import json
import datetime
from datetime import timezone
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# ==============================================================================
# 1. ENHANCED MODEL ARCHITECTURE (BCE Loss + Vectorization)
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
        
        # 2. Memory Gate (GRU Cell)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 3. Graph Attention Layers
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout, edge_dim=1)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout, edge_dim=1)
        self.bn2 = BatchNorm(hidden_dim)
        
        # 4. Pair Scorer Parameters (Bilinear Matrix)
        self.bilinear_W = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.bilinear_W)
        
        self.dropout = dropout
    
    def forward_snapshot(self, x, edge_index, edge_weight, hidden_state=None):
        # A. Encode
        h = self.node_encoder(x)
        
        # B. Memory Update
        if hidden_state is not None:
            h = self.gru(h, hidden_state)
        
        new_hidden_state = h.clone()

        # C. Message Passing
        if edge_index.numel() > 0:
            h_in = h
            h = self.gat1(h, edge_index, edge_attr=edge_weight)
            h = self.bn1(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in
            
            h_in = h
            h = self.gat2(h, edge_index, edge_attr=edge_weight)
            h = self.bn2(h)
            h = F.relu(h)
            h = h + h_in

        # D. Normalize
        h_out = F.normalize(h, p=2, dim=1)
        
        return h_out, new_hidden_state

    def compute_binary_loss(self, embeddings, pos_pairs, neg_pairs, gamma=2.0, alpha=0.75):
        """ Binary Focal Loss """
        pos_src = embeddings[pos_pairs[0]]
        pos_dst = embeddings[pos_pairs[1]]
        pos_scores = self._score_vectors(pos_src, pos_dst)
        
        neg_src = embeddings[neg_pairs[0]]
        neg_dst = embeddings[neg_pairs[1]]
        neg_scores = self._score_vectors(neg_src, neg_dst)
        
        p_pos = torch.sigmoid(pos_scores)
        p_neg = torch.sigmoid(neg_scores)
        
        loss_pos = -alpha * torch.pow(1 - p_pos, gamma) * torch.log(p_pos + 1e-8)
        loss_neg = -(1 - alpha) * torch.pow(p_neg, gamma) * torch.log(1 - p_neg + 1e-8)
        
        return torch.mean(loss_pos) + torch.mean(loss_neg)

    def _score_vectors(self, src, dst):
        bilinear = torch.sum((src @ self.bilinear_W) * dst, dim=1)
        cosine = F.cosine_similarity(src, dst)
        return bilinear + cosine

    def get_all_scores_matrix(self, embeddings):
        weighted_emb = embeddings @ self.bilinear_W
        bilinear_scores = weighted_emb @ embeddings.T
        cosine_scores = embeddings @ embeddings.T
        return bilinear_scores + cosine_scores

# ==============================================================================
# 2. OPTIMIZED AGENT (Daily Rolling + Diagnostics)
# ==============================================================================

@dataclass
class OptimizedSelectorAgent:

    df: pd.DataFrame
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    trace_path: str = "traces/selector.jsonl"
    
    # Hyperparameters
    corr_threshold: float = 0.60
    lookback_weeks: int = 3
    forecast_horizon: int = 1 # We use 1 week (5 days) horizon for stability
    holdout_months: int = 18
    hidden_dim: int = 64
    num_heads: int = 3
    
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
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
            "agent": "selector",
            "event": event,
            "details": details,
        }
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    
    # ------------------------------------------------------------------------
    # Feature Engineering
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
    # Graph Construction (DAILY ROLLING)
    # ------------------------------------------------------------------------

    def build_shifted_snapshots(self, df: pd.DataFrame, window_days: int = 20, step_days: int = 1):
        """
        Creates graph snapshots by sliding a window across the data.
        step_days: Defaults to 1 for DAILY rolling updates.
        """
        df = df.sort_values('date')
        returns_pivot = df.pivot(index='date', columns='ticker', values='log_returns').fillna(0)
        returns_pivot = returns_pivot.reindex(columns=self.tickers, fill_value=0)
        
        dates = returns_pivot.index
        snapshots = []
        
        # We look for pairs that stay correlated for 1 week (5 trading days)
        # This provides a stable target.
        forecast_days = 5 
        
        # Iterate daily (step_days=1)
        for i in range(window_days, len(dates) - forecast_days, step_days):
            current_date = dates[i]
            
            # 1. Input Graph (Past 15 days)
            start_idx = i - window_days
            past_returns = returns_pivot.iloc[start_idx : i+1]
            corr_matrix = np.nan_to_num(past_returns.corr().values)
            
            # Input Edges: Loose correlation (>= 0.5)
            input_edges = np.argwhere(np.abs(corr_matrix) >= 0.5)
            input_edges = input_edges[input_edges[:, 0] < input_edges[:, 1]]
            
            if len(input_edges) == 0:
                 edge_index = torch.empty((2, 0), dtype=torch.long)
                 edge_weights = torch.empty(0, dtype=torch.float)
            else:
                 edge_index = torch.tensor(input_edges.T, dtype=torch.long)
                 edge_weights = torch.tensor([corr_matrix[u,v] for u,v in input_edges], dtype=torch.float)

            # 2. Target Graph (Next 5 days)
            target_start = i + 1
            target_end = i + 1 + forecast_days
            future_returns = returns_pivot.iloc[target_start : target_end]
            
            if len(future_returns) < 2: continue
            
            future_corr = np.nan_to_num(future_returns.corr().values)
            # Target Edges: Strict correlation (>= 0.6)
            pos_edges = np.argwhere(np.abs(future_corr) >= self.corr_threshold)
            pos_edges = pos_edges[pos_edges[:, 0] < pos_edges[:, 1]]
            
            snapshots.append({
                'date': current_date,
                'edge_index': edge_index,
                'edge_weights': edge_weights,
                'target_pos_pairs': torch.tensor(pos_edges.T, dtype=torch.long)
            })
            
        return snapshots

    def create_snapshot_features(self, df: pd.DataFrame, snapshot_date):
        exclude = ["date", "ticker", "close", "adj_factor", "split_factor", "volume", "adj_close", "div_amount"]
        feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
        
        snapshot_df = df[df['date'] <= snapshot_date].groupby('ticker').tail(1)
        node_features = snapshot_df.set_index('ticker')[feature_cols].reindex(self.tickers).fillna(0.0)
        return torch.tensor(node_features.values, dtype=torch.float)

    # ------------------------------------------------------------------------
    # Training Loop (Optimized for Daily Data)
    # ------------------------------------------------------------------------

    def train(self, epochs: int = 20, lr: float = 0.001):
        seed = 42
        torch.manual_seed(seed)
        
        if self.train_df is None: raise ValueError("Call prepare_data() first")

        # Initialize Model
        sample_feat = self.create_snapshot_features(self.train_df, self.train_df['date'].iloc[0])
        self.model = EnhancedTGNN(
            node_dim=sample_feat.shape[1],
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=0.2 
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        print("Building temporal snapshots (DAILY ROLLING)...")
        # Step=1 creates maximum training data
        train_snapshots = self.build_shifted_snapshots(self.train_df, self.lookback_weeks * 5, step_days=1)
        
        # Ensure validation has enough data
        val_snapshots = self.build_shifted_snapshots(self.val_df, self.lookback_weeks * 5, step_days=1)
        
        print(f"Training on {len(train_snapshots)} days, Validating on {len(val_snapshots)} days")
        
        best_val_loss = float('inf')
        patience = 8
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            pred_vals = [] 
            
            hidden_state = None 
            
            for snap in train_snapshots:
                x = self.create_snapshot_features(self.train_df, snap['date']).to(self.device)
                edge_index = snap['edge_index'].to(self.device)
                edge_weights = snap['edge_weights'].to(self.device)
                target_pos = snap['target_pos_pairs'].to(self.device)
                
                h_in = hidden_state.detach() if hidden_state is not None else None
                embeddings, hidden_state = self.model.forward_snapshot(x, edge_index, edge_weights, h_in)
                
                if target_pos.size(1) == 0: continue

                # Sampling Strategy: 90% Random, 10% Hard
                num_pos = target_pos.size(1)
                num_neg = num_pos 
                
                num_rand = int(num_neg * 0.9) 
                rand_src = torch.randint(0, len(self.tickers), (num_rand,), device=self.device)
                rand_dst = torch.randint(0, len(self.tickers), (num_rand,), device=self.device)
                neg_rand = torch.stack([rand_src, rand_dst], dim=0)
                
                num_hard = num_neg - num_rand
                if edge_index.size(1) > num_hard:
                    perm = torch.randperm(edge_index.size(1), device=self.device)[:num_hard]
                    neg_hard = edge_index[:, perm]
                else:
                    hard_src = torch.randint(0, len(self.tickers), (num_hard,), device=self.device)
                    hard_dst = torch.randint(0, len(self.tickers), (num_hard,), device=self.device)
                    neg_hard = torch.stack([hard_src, hard_dst], dim=0)
                
                neg_pairs = torch.cat([neg_rand, neg_hard], dim=1)
                
                loss = self.model.compute_binary_loss(embeddings, target_pos, neg_pairs, gamma=1.0)
                
                with torch.no_grad():
                    pos_scores = torch.sigmoid(self.model._score_vectors(embeddings[target_pos[0]], embeddings[target_pos[1]]))
                    pred_vals.append(pos_scores.mean().item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_snapshots)
            avg_pred = sum(pred_vals) / len(pred_vals) if pred_vals else 0.0
            
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
                    
                    neg_src = torch.randint(0, len(self.tickers), (target_pos.size(1),), device=self.device)
                    neg_dst = torch.randint(0, len(self.tickers), (target_pos.size(1),), device=self.device)
                    neg_pairs = torch.stack([neg_src, neg_dst], dim=0)
                    
                    loss = self.model.compute_binary_loss(embeddings, target_pos, neg_pairs, gamma=1.0)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_snapshots)
            
            status = "no_improvement"
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                status = "improved"
                patience = 8
            else:
                patience -= 1
            
            print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | AvgPosPred: {avg_pred:.3f} | {status}")
            
            if patience <= 0:
                print("Early stopping triggered.")
                break

    def score_pairs(self, use_validation: bool = True, top_k: int = 100):
        if self.model is None: raise ValueError("Model not trained")
        
        df = self.val_df if use_validation else self.test_df
        
        self.model.eval()
        print(f"\nScoring pairs...")
        
        # Use step=1 for scoring too, to ensure we catch the latest valid date
        snapshots = self.build_shifted_snapshots(df, self.lookback_weeks * 5, step_days=1)
        if not snapshots: return pd.DataFrame()
        
        hidden_state = None
        
        with torch.no_grad():
            for snap in snapshots[:-1]:
                x = self.create_snapshot_features(df, snap['date']).to(self.device)
                idx = snap['edge_index'].to(self.device)
                w = snap['edge_weights'].to(self.device)
                _, hidden_state = self.model.forward_snapshot(x, idx, w, hidden_state)
            
            last_snap = snapshots[-1]
            x = self.create_snapshot_features(df, last_snap['date']).to(self.device)
            idx = last_snap['edge_index'].to(self.device)
            w = last_snap['edge_weights'].to(self.device)
            
            embeddings, _ = self.model.forward_snapshot(x, idx, w, hidden_state)
            
            logit_matrix = self.model.get_all_scores_matrix(embeddings)
            prob_matrix = torch.sigmoid(logit_matrix)
            
            mask = torch.triu(torch.ones_like(prob_matrix), diagonal=1).bool()
            prob_matrix[~mask] = -1.0 
            
            top_vals, top_flat_indices = torch.topk(prob_matrix.flatten(), k=top_k)
            
            rows = top_flat_indices // prob_matrix.size(1)
            cols = top_flat_indices % prob_matrix.size(1)
            
            results = []
            for r, c, score in zip(rows.cpu().numpy(), cols.cpu().numpy(), top_vals.cpu().numpy()):
                if score > 0.0: 
                    results.append({
                        'x': self.tickers[r],
                        'y': self.tickers[c],
                        'score': float(score),
                        'date': last_snap['date']
                    })
                
        results_df = pd.DataFrame(results)
        print(f"✅ Scored {len(results_df)} pairs.")
        return results_df
