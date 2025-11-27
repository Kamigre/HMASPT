import os
import json
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


class SimplifiedTGNN(nn.Module):
    
    def __init__(self, node_dim, hidden_dim=32, num_heads=2, dropout=0.1):
        super().__init__()
        
        # Node encoder (single layer)
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # Graph attention (single layer GAT)
        self.gat = GATConv(
            hidden_dim, 
            hidden_dim, 
            heads=num_heads,
            concat=False,
            dropout=dropout
        )
        
        # Pair scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_weight=None, pair_index=None):

        # Encode nodes
        h = self.node_encoder(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Graph attention
        if edge_index.numel() > 0:
            h = self.gat(h, edge_index, edge_attr=edge_weight)
            h = F.relu(h)
        
        # Normalize embeddings
        h = F.normalize(h, p=2, dim=1)
        
        # Score pairs if requested
        if pair_index is not None:
            src_emb = h[pair_index[0]]
            dst_emb = h[pair_index[1]]
            pair_features = torch.cat([src_emb, dst_emb], dim=1)
            scores = self.scorer(pair_features).squeeze(-1)
            return scores
        
        return h


@dataclass
class OptimizedSelectorAgent:

    df: pd.DataFrame
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    trace_path: str = "traces/selector.jsonl"
    corr_threshold: float = 0.7
    lookback_weeks: int = 4  # Rolling window for correlation (4 weeks)
    holdout_years: int = 1
    hidden_dim: int = 32
    num_heads: int = 2
    model: Any = None
    scaler: Optional[StandardScaler] = None
    industry_encoder: Optional[OneHotEncoder] = None
    tickers: Optional[List[str]] = None
    ticker_to_idx: Optional[Dict[str, int]] = None
    train_df: Optional[pd.DataFrame] = None
    val_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None
    val_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    test_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    node_features: Optional[pd.DataFrame] = None
    temporal_graphs: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    
    def __post_init__(self):

        os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
        self._log_event("init", {
            "device": self.device, 
            "corr_threshold": self.corr_threshold,
            "lookback_weeks": self.lookback_weeks
        })
    
    def _log_event(self, event: str, details: Dict[str, Any]):

        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": "selector",
            "event": event,
            "details": details,
        }
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    
    def build_node_features(self, windows=[1, 2, 4], train_end_date=None) -> pd.DataFrame:

        df = self.df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        
        # Basic returns
        df["returns"] = df.groupby("ticker")["adj_close"].pct_change()
        df["log_returns"] = np.log1p(df["returns"])
        
        # Rolling features per window (in weeks, ~5 trading days per week)
        for window in windows:
            days = window * 5  # Convert weeks to trading days
            
            df[f"volatility_{window}w"] = df.groupby("ticker")["returns"].transform(
                lambda x: x.rolling(days, min_periods=max(1, days//2)).std()
            )
            df[f"momentum_{window}w"] = df.groupby("ticker")["adj_close"].transform(
                lambda x: x.pct_change(days)
            )
        
        # Fundamental features
        df["eps_yoy_growth"] = df.get("eps_yoy_growth", 0.0).fillna(0.0)
        df["peg_adj"] = df.get("peg_adj", 0.0).fillna(0.0)
        
        # Handle infinities and NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # One-hot encode industry
        if "sector" in df.columns:
            all_industries = sorted(df["sector"].unique())
            self.industry_encoder = OneHotEncoder(
                sparse_output=False, 
                dtype=int, 
                categories=[all_industries],
                handle_unknown='ignore'
            )
            industry_encoded = self.industry_encoder.fit_transform(df[["sector"]])
            industry_df = pd.DataFrame(
                industry_encoded, 
                columns=self.industry_encoder.get_feature_names_out()
            )
            df = pd.concat([df.reset_index(drop=True), industry_df.reset_index(drop=True)], axis=1)
            df.drop(columns=["sector"], inplace=True, errors="ignore")
        
        # Fill remaining NaNs
        df.fillna(0.0, inplace=True)
        
        # Identify numeric feature columns
        exclude_cols = ["date", "ticker", "close", "adj_factor", "split_factor", 
                       "div_amount", "volume", "adj_close"]
        numeric_cols = [c for c in df.columns 
                       if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
        
        # Fit scaler on training data only
        if self.scaler is None:
            self.scaler = StandardScaler()
            if train_end_date is not None:
                train_mask = df["date"] < pd.to_datetime(train_end_date)
                train_data = df.loc[train_mask, numeric_cols]
                self.scaler.fit(train_data)
            else:
                self.scaler.fit(df[numeric_cols])
        
        # Scale features
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        self.node_features = df
        
        self._log_event("node_features_built", {
            "n_rows": len(df),
            "n_features": len(numeric_cols),
            "windows": windows,
            "train_end_date": str(train_end_date) if train_end_date else None
        })
        
        return df
    
    def prepare_data(self, train_end_date: str = None):

        if train_end_date is None:
            # Auto-split based on holdout_years
            last_date = self.df["date"].max()
            train_end_date = last_date - pd.DateOffset(years=self.holdout_years)
        
        train_end = pd.to_datetime(train_end_date)
        
        # Build features
        self.build_node_features(train_end_date=train_end)
        
        df = self.node_features.copy()
        df["date"] = pd.to_datetime(df["date"])
        
        # Store tickers
        self.tickers = sorted(df["ticker"].unique())
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}
        
        # Calculate split dates
        last_date = df["date"].max()
        holdout_start = train_end
        mid_point = holdout_start + (last_date - holdout_start) / 2
        
        val_start = holdout_start
        val_end = mid_point
        test_start = mid_point
        test_end = last_date
        
        self.val_period = (val_start, val_end)
        self.test_period = (test_start, test_end)
        
        # Split data
        self.train_df = df[df["date"] < val_start].copy()
        self.val_df = df[(df["date"] >= val_start) & (df["date"] < val_end)].copy()
        self.test_df = df[df["date"] >= test_start].copy()
        
        self._log_event("data_prepared", {
            "n_tickers": len(self.tickers),
            "train_samples": len(self.train_df),
            "val_samples": len(self.val_df),
            "test_samples": len(self.test_df),
            "train_end": str(train_end.date()),
            "val_period": (str(val_start.date()), str(val_end.date())),
            "test_period": (str(test_start.date()), str(test_end.date()))
        })
        
        print(f"✅ Data prepared: {len(self.tickers)} tickers")
        print(f"   Training: {len(self.train_df)} samples (up to {train_end.date()})")
        print(f"   Validation: {len(self.val_df)} samples ({val_start.date()} → {val_end.date()})")
        print(f"   Test: {len(self.test_df)} samples ({test_start.date()} → {test_end.date()})")
        
        return self.train_df, self.val_df, self.test_df
    
    def build_temporal_snapshots(self, df: pd.DataFrame, window_days: int = 20):

        df = df.sort_values('date')
        
        # Pivot returns
        returns_pivot = df.pivot(
            index='date', 
            columns='ticker', 
            values='log_returns'
        ).fillna(0)
        
        # Ensure all tickers present
        returns_pivot = returns_pivot.reindex(columns=self.tickers, fill_value=0)
        
        snapshots = []
        dates = returns_pivot.index
        
        # Create weekly snapshots (every 5 trading days)
        for i in range(window_days, len(dates), 5):  # Step by 5 days (1 week)
            end_date = dates[i]
            start_idx = max(0, i - window_days)
            
            # Get window of returns
            window_returns = returns_pivot.iloc[start_idx:i+1]
            
            # Compute correlation for this window
            corr_matrix = window_returns.corr().values
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            
            # Build edges
            edges = np.argwhere(np.abs(corr_matrix) >= self.corr_threshold)
            edges = edges[edges[:, 0] < edges[:, 1]]
            
            if len(edges) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_weights = torch.empty(0, dtype=torch.float)
            else:
                edge_index = torch.tensor(edges.T, dtype=torch.long)
                edge_weights = torch.tensor(
                    [corr_matrix[i, j] for i, j in edges], 
                    dtype=torch.float
                )
            
            snapshots.append({
                'date': end_date,
                'edge_index': edge_index,
                'edge_weights': edge_weights,
                'num_edges': len(edges)
            })
        
        return snapshots
    
    def create_snapshot_features(self, df: pd.DataFrame, snapshot_date):

        exclude_cols = ["date", "ticker", "close", "adj_factor", "split_factor", 
                       "div_amount", "volume", "adj_close"]
        feature_cols = [c for c in df.columns 
                       if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
        
        # Get data up to snapshot date (last 5 days for averaging)
        snapshot_df = df[df['date'] <= snapshot_date].groupby('ticker').tail(5)
        
        # Aggregate features per ticker
        node_features = snapshot_df.groupby('ticker')[feature_cols].mean()
        node_features = node_features.reindex(self.tickers, fill_value=0.0)
        
        # Features are already scaled
        return torch.tensor(node_features.values, dtype=torch.float)
    
    def train(self, epochs: int = 5, lr: float = 0.001, batch_size: int = 512, 
              snapshot_stride: int = 1):

        if self.train_df is None:
            raise ValueError("Call prepare_data() first")
        
        # Get number of features
        exclude_cols = ["date", "ticker", "close", "adj_factor", "split_factor", 
                       "div_amount", "volume", "adj_close"]
        feature_cols = [c for c in self.train_df.columns 
                       if c not in exclude_cols and np.issubdtype(self.train_df[c].dtype, np.number)]
        num_features = len(feature_cols)
        
        # Initialize model
        self.model = SimplifiedTGNN(
            node_dim=num_features,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        # Build temporal snapshots
        print("Building weekly temporal snapshots...")
        self.temporal_graphs = self.build_temporal_snapshots(
            self.train_df, 
            window_days=self.lookback_weeks * 5
        )
        
        # Subsample snapshots for faster training
        snapshots = self.temporal_graphs[::snapshot_stride]
        
        self._log_event("training_started", {
            "n_snapshots": len(snapshots),
            "total_snapshots": len(self.temporal_graphs),
            "snapshot_stride": snapshot_stride,
            "epochs": epochs,
            "n_nodes": len(self.tickers),
            "hidden_dim": self.hidden_dim
        })
        
        print(f"\nTraining TGNN:")
        print(f"  Nodes: {len(self.tickers)}")
        print(f"  Snapshots: {len(snapshots)} (stride={snapshot_stride})")
        print(f"  Hidden dim: {self.hidden_dim}")
        
        num_nodes = len(self.tickers)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            # Process each temporal snapshot
            for snap_idx, snapshot in enumerate(snapshots):
                edge_index = snapshot['edge_index'].to(self.device)
                edge_weights = snapshot['edge_weights'].to(self.device)
                
                # Skip snapshots with no edges
                if edge_index.numel() == 0:
                    continue
                
                # Create features for this snapshot
                x = self.create_snapshot_features(
                    self.train_df, 
                    snapshot['date']
                ).to(self.device)
                
                # Positive pairs from edges
                pos_pairs = edge_index.T
                num_pos = len(pos_pairs)
                
                if num_pos == 0:
                    continue
                
                # Negative sampling
                neg_src = torch.randint(0, num_nodes, (num_pos,), device=self.device)
                neg_dst = torch.randint(0, num_nodes, (num_pos,), device=self.device)
                
                # Ensure neg_src != neg_dst
                mask = neg_src == neg_dst
                while mask.any():
                    neg_dst[mask] = torch.randint(0, num_nodes, (mask.sum(),), device=self.device)
                    mask = neg_src == neg_dst
                
                neg_pairs = torch.stack([neg_src, neg_dst], dim=1)
                
                # Combine positive and negative
                all_pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
                labels = torch.cat([
                    torch.ones(num_pos, device=self.device),
                    torch.zeros(num_pos, device=self.device)
                ])
                
                # Shuffle
                perm = torch.randperm(len(all_pairs))
                all_pairs = all_pairs[perm]
                labels = labels[perm]
                
                # Mini-batch training on this snapshot
                for i in range(0, len(all_pairs), batch_size):
                    batch_pairs = all_pairs[i:i+batch_size].T
                    batch_labels = labels[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    scores = self.model(
                        x, 
                        edge_index, 
                        edge_weights,
                        pair_index=batch_pairs
                    )
                    
                    loss = criterion(scores, batch_labels)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            
            self._log_event("epoch_complete", {
                "epoch": epoch + 1,
                "avg_loss": avg_loss
            })
            
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        self._log_event("training_complete", {"epochs": epochs})
        print("✅ Training complete")
    
    def score_pairs(self, use_validation: bool = True, top_k: int = 100):

        if self.model is None:
            raise ValueError("Model not trained. Call train() first")
        
        # Select period
        if use_validation:
            df = self.val_df
            period_start, period_end = self.val_period
            period_name = "validation"
        else:
            df = self.test_df
            period_start, period_end = self.test_period
            period_name = "test"
        
        if df is None or len(df) == 0:
            raise ValueError(f"No {period_name} data available")
        
        self.model.eval()
        
        # Build temporal snapshots for scoring period
        print(f"\nScoring pairs on {period_name} period...")
        snapshots = self.build_temporal_snapshots(
            df, 
            window_days=self.lookback_weeks * 5
        )
        
        if len(snapshots) == 0:
            raise ValueError("No snapshots created for scoring period")
        
        # Use the most recent snapshot
        recent_snapshot = snapshots[-1]
        
        edge_index = recent_snapshot['edge_index'].to(self.device)
        edge_weights = recent_snapshot['edge_weights'].to(self.device)
        
        self._log_event("scoring_started", {
            "period": period_name,
            "scoring_date": str(recent_snapshot['date'].date()),
            "num_edges": recent_snapshot['num_edges']
        })
        
        print(f"  Scoring date: {recent_snapshot['date'].date()}")
        print(f"  Period edges: {edge_index.size(1)}")
        
        # Create features for most recent period
        x = self.create_snapshot_features(df, recent_snapshot['date']).to(self.device)
        
        # Score all possible pairs in batches
        num_nodes = len(self.tickers)
        src_idx, dst_idx = np.triu_indices(num_nodes, k=1)
        
        all_scores = []
        batch_size = 10000
        
        with torch.no_grad():
            for i in range(0, len(src_idx), batch_size):
                batch_src = torch.tensor(
                    src_idx[i:i+batch_size], 
                    device=self.device
                )
                batch_dst = torch.tensor(
                    dst_idx[i:i+batch_size], 
                    device=self.device
                )
                pair_index = torch.stack([batch_src, batch_dst], dim=0)
                
                scores = self.model(
                    x, 
                    edge_index, 
                    edge_weights,
                    pair_index=pair_index
                )
                
                all_scores.append(scores.cpu().numpy())
        
        all_scores = np.concatenate(all_scores)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'x': [self.tickers[i] for i in src_idx],
            'y': [self.tickers[i] for i in dst_idx],
            'score': all_scores
        }).sort_values('score', ascending=False).reset_index(drop=True)
        
        top_pairs = results.head(top_k)
        
        topk_dict = top_pairs.head(100).to_dict(orient="records")
        self._log_event("scoring_complete", {
            "period": period_name,
            "n_pairs": len(results),
            "top_100": topk_dict
        })
        
        print(f"✅ Scored {len(results)} pairs")
        print(f"   Top pair: {top_pairs.iloc[0]['x']}-{top_pairs.iloc[0]['y']} "
              f"(score: {top_pairs.iloc[0]['score']:.4f})")
        
        return top_pairs
