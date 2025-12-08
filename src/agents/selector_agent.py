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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import CONFIG


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

        self.criterion = nn.BCEWithLogitsLoss()
    
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

    def compute_binary_loss(self, embeddings, pos_pairs, neg_pairs):
        """
        Computes Binary Cross Entropy Loss.
        Forces positive pairs -> 1.0 and negative pairs -> 0.0
        """
        # Score Positive Pairs (Target = 1)
        pos_src = embeddings[pos_pairs[0]]
        pos_dst = embeddings[pos_pairs[1]]
        pos_scores = self._score_vectors(pos_src, pos_dst)
        
        # Score Negative Pairs (Target = 0)
        neg_src = embeddings[neg_pairs[0]]
        neg_dst = embeddings[neg_pairs[1]]
        neg_scores = self._score_vectors(neg_src, neg_dst)
        
        # Combine
        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([
            torch.ones_like(pos_scores), 
            torch.zeros_like(neg_scores)
        ])
        
        return self.criterion(all_scores, all_labels)

    def _score_vectors(self, src, dst):
        """ Computes x_src * W * x_dst + Cosine(x_src, x_dst) """
        bilinear = torch.sum((src @ self.bilinear_W) * dst, dim=1)
        cosine = F.cosine_similarity(src, dst)
        return bilinear + cosine

    def get_all_scores_matrix(self, embeddings):
        """ Vectorized Inference (Bilinear + Cosine) """
        weighted_emb = embeddings @ self.bilinear_W
        bilinear_scores = weighted_emb @ embeddings.T
        cosine_scores = embeddings @ embeddings.T
        return bilinear_scores + cosine_scores

# ==============================================================================
# 2. OPTIMIZED AGENT (Forecasting + Binary Selection)
# ==============================================================================

@dataclass
class OptimizedSelectorAgent:

    df: pd.DataFrame
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    trace_path: str = "traces/selector.jsonl"
    
    # Hyperparameters
    corr_threshold: float = 0.60
    lookback_weeks: int = 4
    forecast_horizon: int = 1
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
            "timestamp": datetime.datetime.utcnow().isoformat(),
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
    # Graph Construction (Shifted Targets)
    # ------------------------------------------------------------------------

    def build_shifted_snapshots(self, df: pd.DataFrame, window_days: int = 20, mode='train'):
        df = df.sort_values('date')
        returns_pivot = df.pivot(index='date', columns='ticker', values='log_returns').fillna(0)
        returns_pivot = returns_pivot.reindex(columns=self.tickers, fill_value=0)
        
        dates = returns_pivot.index
        snapshots = []
        step = 5 if mode == 'train' else 10
        forecast_days = self.forecast_horizon * 5
        
        for i in range(window_days, len(dates) - forecast_days, step):
            current_date = dates[i]
            
            # 1. Input Graph (Past)
            start_idx = i - window_days
            past_returns = returns_pivot.iloc[start_idx : i+1]
            corr_matrix = np.nan_to_num(past_returns.corr().values)
            
            input_edges = np.argwhere(np.abs(corr_matrix) >= 0.5)
            input_edges = input_edges[input_edges[:, 0] < input_edges[:, 1]]
            
            if len(input_edges) == 0:
                 edge_index = torch.empty((2, 0), dtype=torch.long)
                 edge_weights = torch.empty(0, dtype=torch.float)
            else:
                 edge_index = torch.tensor(input_edges.T, dtype=torch.long)
                 edge_weights = torch.tensor([corr_matrix[u,v] for u,v in input_edges], dtype=torch.float)

            # 2. Target Graph (Future)
            target_start = i + 1
            target_end = i + 1 + forecast_days
            future_returns = returns_pivot.iloc[target_start : target_end]
            
            if len(future_returns) < 2: continue
            
            future_corr = np.nan_to_num(future_returns.corr().values)
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
            self.model.train()
            train_loss = 0.0
            hidden_state = None 
            
            for snap in train_snapshots:
                x = self.create_snapshot_features(self.train_df, snap['date']).to(self.device)
                edge_index = snap['edge_index'].to(self.device)
                edge_weights = snap['edge_weights'].to(self.device)
                target_pos = snap['target_pos_pairs'].to(self.device)
                
                h_in = hidden_state.detach() if hidden_state is not None else None
                embeddings, hidden_state = self.model.forward_snapshot(x, edge_index, edge_weights, h_in)
                
                if target_pos.size(1) == 0: continue

                # Negative Sampling
                num_pos = target_pos.size(1)
                # Sample 1:1 negatives
                neg_src = torch.randint(0, len(self.tickers), (num_pos,), device=self.device)
                neg_dst = torch.randint(0, len(self.tickers), (num_pos,), device=self.device)
                neg_pairs = torch.stack([neg_src, neg_dst], dim=0)
                
                # --- CHANGE: Compute Binary Loss ---
                loss = self.model.compute_binary_loss(embeddings, target_pos, neg_pairs)
                
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
                    
                    neg_src = torch.randint(0, len(self.tickers), (target_pos.size(1) * 2,), device=self.device)
                    neg_dst = torch.randint(0, len(self.tickers), (target_pos.size(1) * 2,), device=self.device)
                    neg_pairs = torch.stack([neg_src, neg_dst], dim=0)
                    
                    loss = self.model.compute_binary_loss(embeddings, target_pos, neg_pairs)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_snapshots)
            
            status = "no_improvement"
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                status = "improved"
                patience = 6
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
    # Inference (Vectorized + Probability)
    # ------------------------------------------------------------------------

    def score_pairs(self, use_validation: bool = True, top_k: int = 100):
        if self.model is None: raise ValueError("Model not trained")
        
        df = self.val_df if use_validation else self.test_df
        
        self.model.eval()
        print(f"\nScoring pairs...")
        
        snapshots = self.build_shifted_snapshots(df, self.lookback_weeks * 5, mode='val')
        if not snapshots: return pd.DataFrame()
        
        hidden_state = None
        
        # Run history to build memory
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
            
            # Raw scores (Logits)
            logit_matrix = self.model.get_all_scores_matrix(embeddings)
            
            # Convert to Probability (Sigmoid) for "Real Pair" confidence
            prob_matrix = torch.sigmoid(logit_matrix)
            
            # Mask diagonal/duplicates
            mask = torch.triu(torch.ones_like(prob_matrix), diagonal=1).bool()
            
            # We want specific values, not just top K relative.
            # But to keep output manageable, we take top K sorted by probability.
            prob_matrix[~mask] = -1.0 # Filter out invalid
            
            top_vals, top_flat_indices = torch.topk(prob_matrix.flatten(), k=top_k)
            
            rows = top_flat_indices // prob_matrix.size(1)
            cols = top_flat_indices % prob_matrix.size(1)
            
            results = []
            for r, c, score in zip(rows.cpu().numpy(), cols.cpu().numpy(), top_vals.cpu().numpy()):
                if score > 0.0: # Only keep positive probabilities
                    results.append({
                        'x': self.tickers[r],
                        'y': self.tickers[c],
                        'score': float(score), # This is now a probability (0.0 - 1.0)
                        'date': last_snap['date']
                    })
                
        results_df = pd.DataFrame(results)
        print(f"✅ Scored {len(results_df)} pairs.")
        return results_df
