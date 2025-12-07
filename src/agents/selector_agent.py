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

class EnhancedTGNN(nn.Module):
    
    def __init__(self, node_dim, hidden_dim=64, num_heads=4, dropout=0.3):
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
        
        # 4. Pair Scorer (Bilinear + Cosine)
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_weight=None, pair_index=None, hidden_state=None):
        """
        Args:
            x: Node features [Num_Nodes, Feat_Dim]
            edge_index: Graph connectivity
            hidden_state: Memory from previous snapshot [Num_Nodes, Hidden_Dim]
        """

        # A. Encode current features
        h = self.node_encoder(x)
        
        # B. Integrate Memory (Lightweight Recurrence)
        if hidden_state is not None:
            # GRU update: New_State = GRU(Current_Features, Old_State)
            h = self.gru(h, hidden_state)
        
        # C. Graph Message Passing
        if edge_index.numel() > 0:
            # Layer 1
            h_in = h
            h = self.gat1(h, edge_index, edge_attr=edge_weight)
            h = self.bn1(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in  # Residual
            
            # Layer 2
            h_in = h
            h = self.gat2(h, edge_index, edge_attr=edge_weight)
            h = self.bn2(h)
            h = F.relu(h)
            h = h + h_in  # Residual
        
        # D. Normalize for similarity search (Output node embedding)
        h_out = F.normalize(h, p=2, dim=1)
        
        # E. Score Pairs (if requested)
        if pair_index is not None:
            src_emb = h_out[pair_index[0]]
            dst_emb = h_out[pair_index[1]]
            
            # Combine Bilinear Score (Interaction) + Cosine Similarity (Direction)
            scores = self.bilinear(src_emb, dst_emb).squeeze(-1)
            cos_sim = F.cosine_similarity(src_emb, dst_emb)
            return scores + cos_sim, h_out
        
        return h_out

# ==============================================================================
# 2. OPTIMIZED AGENT
# ==============================================================================

@dataclass
class OptimizedSelectorAgent:

    df: pd.DataFrame
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    trace_path: str = "traces/selector.jsonl"
    
    # --- Improved Hyperparameters ---
    corr_threshold: float = 0.65       # Graph construction threshold (keep broad)
    train_corr_threshold: float = 0.8  # NEW: Stricter threshold for training samples
    lookback_weeks: int = 4
    holdout_months: int = 18
    hidden_dim: int = 128              # INCREASED CAPACITY
    num_heads: int = 6                 # INCREASED CAPACITY
    dropout: float = 0.3               # INCREASED REGULARIZATION (Trace Analysis)
    
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
    val_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    test_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    node_features: Optional[pd.DataFrame] = None
    temporal_graphs: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    
    def __post_init__(self):
        # Create trace directory dynamically for robustness
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
    # Feature Engineering (Unchanged)
    # ------------------------------------------------------------------------
    
    def build_node_features(self, windows=[1, 2, 4], train_end_date=None) -> pd.DataFrame:
        """Constructs rich node features with volatility and momentum."""
        
        df = self.df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        
        # Returns
        df["returns"] = df.groupby("ticker")["adj_close"].pct_change()
        df["log_returns"] = np.log1p(df["returns"])
        
        # Rolling Windows
        for window in windows:
            days = window * 5
            
            # Volatility
            df[f"volatility_{window}w"] = df.groupby("ticker")["returns"].transform(
                lambda x: x.rolling(days, min_periods=max(1, days//2)).std()
            )
            # Momentum
            df[f"momentum_{window}w"] = df.groupby("ticker")["adj_close"].transform(
                lambda x: x.pct_change(days)
            )
            # Relative Volume (if available)
            if "volume" in df.columns:
                df[f"rel_vol_{window}w"] = df.groupby("ticker")["volume"].transform(
                    lambda x: x / (x.rolling(days).mean() + 1)
                )

        # Fundamentals & Industry
        df["eps_yoy_growth"] = df.get("eps_yoy_growth", 0.0).fillna(0.0)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        if "sector" in df.columns:
            if self.industry_encoder is None:
                self.industry_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                self.industry_encoder.fit(df[["sector"]])
            industry_encoded = self.industry_encoder.transform(df[["sector"]])
            
            ind_cols = [f"ind_{i}" for i in range(industry_encoded.shape[1])]
            ind_df = pd.DataFrame(industry_encoded, columns=ind_cols, index=df.index)
            df = pd.concat([df, ind_df], axis=1)
            df.drop(columns=["sector"], inplace=True, errors="ignore")
            
        df.fillna(0.0, inplace=True)
        
        # Identify numeric columns for scaling
        exclude = ["date", "ticker", "close", "adj_factor", "split_factor", "volume", "adj_close"]
        numeric_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
        
        # Scale
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
        
        # Time Splits
        last_date = df["date"].max()
        mid_point = train_end + (last_date - train_end) / 2
        
        self.val_period = (train_end, mid_point)
        self.test_period = (mid_point, last_date)
        
        self.train_df = df[df["date"] < train_end].copy()
        self.val_df = df[(df["date"] >= train_end) & (df["date"] < mid_point)].copy()
        self.test_df = df[df["date"] >= mid_point].copy()
        
        print(f"✅ Data Prepared. Tickers: {len(self.tickers)}")
        print(f"    Train: {len(self.train_df)} | Val: {len(self.val_df)} | Test: {len(self.test_df)}")
        
        return self.train_df, self.val_df, self.test_df

    # ------------------------------------------------------------------------
    # Graph Construction (IMPROVED: Ratio Stability Factor)
    # ------------------------------------------------------------------------

    def build_temporal_snapshots(self, df: pd.DataFrame, window_days: int = 20):
        """Builds graph snapshots with volatility-aware and ratio-stability-aware edge weights."""
        df = df.sort_values('date')
        
        # Returns Pivot (for Correlation)
        returns_pivot = df.pivot(index='date', columns='ticker', values='log_returns').fillna(0)
        returns_pivot = returns_pivot.reindex(columns=self.tickers, fill_value=0)
        
        # Volatility Pivot (for penalty)
        vol_col = [c for c in df.columns if 'volatility' in c][-1]  
        vol_pivot = df.pivot(index='date', columns='ticker', values=vol_col).fillna(0)
        vol_pivot = vol_pivot.reindex(columns=self.tickers, fill_value=0)
        
        # Price Pivot (for Ratio Stability) - NEW
        adj_close_pivot = df.pivot(index='date', columns='ticker', values='adj_close').fillna(1.0)
        adj_close_pivot = adj_close_pivot.reindex(columns=self.tickers, fill_value=1.0)

        snapshots = []
        dates = returns_pivot.index
        
        for i in range(window_days, len(dates), 5):
            end_date = dates[i]
            start_idx = max(0, i - window_days)
            
            # 1. Correlation
            window_returns = returns_pivot.iloc[start_idx:i+1]
            corr_matrix = window_returns.corr().values
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            
            # 2. Volatility Penalty
            avg_vols = vol_pivot.iloc[start_idx:i+1].mean().values
            vol_diff = np.abs(avg_vols[:, None] - avg_vols[None, :])
            vol_penalty = 1.0 / (1.0 + vol_diff * 10)  
            
            # 3. Ratio Stability Factor (NEW IMPROVEMENT)
            # This penalizes pairs whose price ratio fluctuates wildly
            window_adj_close = adj_close_pivot.iloc[start_idx:i+1].values
            num_nodes = len(self.tickers)
            ratio_stds = np.zeros_like(corr_matrix)
            
            # We calculate this approximately to keep it fast
            for n1 in range(num_nodes):
                for n2 in range(n1 + 1, num_nodes):
                    if abs(corr_matrix[n1, n2]) < self.corr_threshold: 
                        continue # Skip calculation for uncorrelated pairs
                        
                    ratio = window_adj_close[:, n1] / (window_adj_close[:, n2] + 1e-6)
                    std_dev = np.std(ratio)
                    ratio_stds[n1, n2] = ratio_stds[n2, n1] = std_dev
            
            max_std = np.max(ratio_stds) if np.max(ratio_stds) > 0 else 1.0
            ratio_stability_factor = 1.0 / (1.0 + ratio_stds / max_std * 5)

            # Final Adjacency
            adj_matrix = corr_matrix * vol_penalty * ratio_stability_factor
            
            # Edges
            edges = np.argwhere(np.abs(adj_matrix) >= self.corr_threshold)
            edges = edges[edges[:, 0] < edges[:, 1]]
            
            if len(edges) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_weights = torch.empty(0, dtype=torch.float)
            else:
                edge_index = torch.tensor(edges.T, dtype=torch.long)
                edge_weights = torch.tensor([adj_matrix[i, j] for i, j in edges], dtype=torch.float)
                
            snapshots.append({
                'date': end_date,
                'edge_index': edge_index,
                'edge_weights': edge_weights,
                'num_edges': len(edges)
            })
        
        return snapshots

    def create_snapshot_features(self, df: pd.DataFrame, snapshot_date):
        # Extract numeric features for the specific date
        exclude = ["date", "ticker", "close", "adj_factor", "split_factor", "div_amount", "volume", "adj_close"]
        feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
        
        snapshot_df = df[df['date'] <= snapshot_date].groupby('ticker').tail(5)
        node_features = snapshot_df.groupby('ticker')[feature_cols].mean()
        node_features = node_features.reindex(self.tickers, fill_value=0.0)
        return torch.tensor(node_features.values, dtype=torch.float)

    # ------------------------------------------------------------------------
    # Validation Method (Unchanged)
    # ------------------------------------------------------------------------
    def _validate_model(self, df: pd.DataFrame, criterion, snapshot_stride: int = 1):
        self.model.eval()
        
        val_graphs = self.build_temporal_snapshots(df, self.lookback_weeks * 5)
        snapshots = val_graphs[::snapshot_stride]
        if not snapshots: return float('inf')
        
        total_val_loss = 0.0
        num_batches = 0
        num_nodes = len(self.tickers)
        batch_size = 1024
        
        with torch.no_grad():
            hidden_state = None
            for snapshot in snapshots:
                edge_index = snapshot['edge_index'].to(self.device)
                edge_weights = snapshot['edge_weights'].to(self.device)
                x = self.create_snapshot_features(df, snapshot['date']).to(self.device)
                
                if edge_index.numel() == 0: continue
                
                # --- Sampling ---
                pos_pairs = edge_index.T
                num_pos = len(pos_pairs)
                if num_pos == 0: continue
                
                neg_src = torch.randint(0, num_nodes, (num_pos,), device=self.device)
                neg_dst = torch.randint(0, num_nodes, (num_pos,), device=self.device)
                neg_pairs = torch.stack([neg_src, neg_dst], dim=1)
                
                all_pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
                labels = torch.cat([
                    torch.ones(num_pos, device=self.device) * 0.9,  
                    torch.zeros(num_pos, device=self.device)
                ])
                
                # Full Graph Pass
                embeddings = self.model(x, edge_index, edge_weights, hidden_state=hidden_state)
                hidden_state = embeddings 
                
                for i in range(0, len(all_pairs), batch_size):
                    batch_pairs = all_pairs[i:i+batch_size].T
                    batch_labels = labels[i:i+batch_size]
                    
                    src_emb = embeddings[batch_pairs[0]]
                    dst_emb = embeddings[batch_pairs[1]]
                    scores = self.model.bilinear(src_emb, dst_emb).squeeze(-1) + F.cosine_similarity(src_emb, dst_emb)
                    
                    loss = criterion(scores, batch_labels)
                    total_val_loss += loss.item()
                    num_batches += 1

        self.model.train()
        return total_val_loss / max(num_batches, 1)

    # ------------------------------------------------------------------------
    # Training Loop (IMPROVED: Harder Sampling + Scheduler + Patience)
    # ------------------------------------------------------------------------

    def train(self, epochs: int = 10, lr: float = 0.001, batch_size: int = 1024, snapshot_stride: int = 1):
        
        seed = CONFIG.get("random_seed", 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        
        if self.train_df is None or self.val_df is None:  
            raise ValueError("Call prepare_data() first and ensure validation data exists.")

        # --- Early Stopping Setup ---
        # IMPROVEMENT: Increased patience to handle validation noise
        PATIENCE = 7  
        epochs_without_improvement = 0
        best_val_loss = float('inf')
        
        temp_model_path = os.path.join(os.path.dirname(self.trace_path), "best_model_temp.pth")
        
        # Setup Model
        feat_dim = self.create_snapshot_features(self.train_df, self.train_df['date'].iloc[0]).shape[1]
        
        self.model = EnhancedTGNN(
            node_dim=feat_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # IMPROVEMENT: Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        criterion = nn.BCEWithLogitsLoss()
        
        print("Building temporal snapshots...")
        self.temporal_graphs = self.build_temporal_snapshots(self.train_df, self.lookback_weeks * 5)
        snapshots = self.temporal_graphs[::snapshot_stride]
        
        print(f"\nTraining EnhancedTGNN (Nodes: {len(self.tickers)}, Snapshots: {len(snapshots)})")
        
        num_nodes = len(self.tickers)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            hidden_state = None  
            
            for snapshot in snapshots:
                full_edge_index = snapshot['edge_index'].to(self.device)
                full_edge_weights = snapshot['edge_weights'].to(self.device)
                x = self.create_snapshot_features(self.train_df, snapshot['date']).to(self.device)
                
                if full_edge_index.numel() == 0: continue
                
                # --- Sampling ---
                # IMPROVEMENT: Harder Positive Sampling
                # Only use edge with weight >= train_corr_threshold (0.8) as positive samples
                high_corr_mask = (full_edge_weights.abs() >= self.train_corr_threshold)
                pos_pairs = full_edge_index[:, high_corr_mask].T
                
                num_pos = len(pos_pairs)
                if num_pos == 0: continue
                
                # 2x Negative Sampling
                neg_src = torch.randint(0, num_nodes, (num_pos * 2,), device=self.device)
                neg_dst = torch.randint(0, num_nodes, (num_pos * 2,), device=self.device)
                neg_pairs = torch.stack([neg_src, neg_dst], dim=1)
                
                all_pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
                labels = torch.cat([
                    torch.ones(num_pos, device=self.device) * 0.9, # Label Smoothing
                    torch.zeros(num_pos * 2, device=self.device)
                ])
                
                # Shuffle
                perm = torch.randperm(len(all_pairs))
                all_pairs = all_pairs[perm]
                labels = labels[perm]
                
                # --- Forward Pass & Optimization ---
                for i in range(0, len(all_pairs), batch_size):
                    batch_pairs = all_pairs[i:i+batch_size].T
                    batch_labels = labels[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    
                    current_hidden_state = hidden_state.detach() if hidden_state is not None else None
                    
                    # 1. Forward pass for loss calculation
                    scores, embeddings = self.model(x, full_edge_index, full_edge_weights, pair_index=batch_pairs, hidden_state=current_hidden_state)
                    
                    # 2. Update hidden_state for next snapshot (detached)
                    hidden_state = self.model(x, full_edge_index, full_edge_weights, hidden_state=current_hidden_state).detach()

                    loss = criterion(scores, batch_labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_train_loss = total_loss / max(num_batches, 1)
            
            # --- Early Stopping Check ---
            val_loss = self._validate_model(self.val_df, criterion, snapshot_stride=snapshot_stride)
            
            # Step the scheduler
            scheduler.step(val_loss)
            
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), temp_model_path)
                self._log_event("training_status", {"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": val_loss, "status": "improved", "best_loss": best_val_loss})
                print("  🟢 Validation loss improved. Saving best model state.")
            else:
                epochs_without_improvement += 1
                self._log_event("training_status", {"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": val_loss, "status": "no_improvement", "patience_left": PATIENCE - epochs_without_improvement})
                
            if epochs_without_improvement >= PATIENCE:
                print(f"  🛑 Early stopping triggered. Validation loss hasn't improved for {PATIENCE} epochs. Loading best weights.")
                if os.path.exists(temp_model_path):
                    self.model.load_state_dict(torch.load(temp_model_path))
                    os.remove(temp_model_path)
                else:
                    print("⚠️ Warning: Best model state not found. Proceeding with current model.")
                break
                
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            
        print("✅ Training complete")

    # ------------------------------------------------------------------------
    # Score Pairs (Unchanged)
    # ------------------------------------------------------------------------

    def score_pairs(self, use_validation: bool = True, top_k: int = 100):
        if self.model is None: raise ValueError("Model not trained")
        
        df = self.val_df if use_validation else self.test_df
        period_name = "validation" if use_validation else "test"
        
        self.model.eval()
        print(f"\nScoring pairs on {period_name}...")
        
        snapshots = self.build_temporal_snapshots(df, self.lookback_weeks * 5)
        if not snapshots: raise ValueError("No snapshots")
        
        snapshot = snapshots[-1]
        
        edge_index = snapshot['edge_index'].to(self.device)
        edge_weights = snapshot['edge_weights'].to(self.device)
        x = self.create_snapshot_features(df, snapshot['date']).to(self.device)
        
        num_nodes = len(self.tickers)
        src_idx, dst_idx = np.triu_indices(num_nodes, k=1)
        
        all_scores = []
        batch_size = 10000
        
        with torch.no_grad():
            embeddings = self.model(x, edge_index, edge_weights, hidden_state=None)  
            
            for i in range(0, len(src_idx), batch_size):
                batch_src = torch.tensor(src_idx[i:i+batch_size], device=self.device)
                batch_dst = torch.tensor(dst_idx[i:i+batch_size], device=self.device)
                
                src_emb = embeddings[batch_src]
                dst_emb = embeddings[batch_dst]
                
                scores = self.model.bilinear(src_emb, dst_emb).squeeze(-1) + F.cosine_similarity(src_emb, dst_emb)
                all_scores.append(scores.cpu().numpy())
        
        all_scores = np.concatenate(all_scores)
        
        results = pd.DataFrame({
            'x': [self.tickers[i] for i in src_idx],
            'y': [self.tickers[i] for i in dst_idx],
            'score': all_scores
        }).sort_values('score', ascending=False).reset_index(drop=True)
        
        print(f"✅ Scored {len(results)} pairs. Top: {results.iloc[0]['x']}-{results.iloc[0]['y']}")
        return results.head(top_k)
