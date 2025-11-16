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
   
    def build_node_features(self, windows=[5, 15, 30]) -> pd.DataFrame:
        df = self.df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

        for window in windows:
            df[f"mean_{window}"] = df.groupby("ticker")["adj_close"].transform(lambda x: x.rolling(window).mean())
            df[f"std_{window}"] = df.groupby("ticker")["adj_close"].transform(lambda x: x.rolling(window).std())
            df[f"cum_return_{window}"] = df.groupby("ticker")["adj_close"].transform(
                lambda x: np.log(x / x.shift(1)).rolling(window).sum()
            )

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df["eps_yoy_ growth"] = df.get("eps_yoy_growth", 0.0).fillna(0.0)
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

            mask = (train_df["date"].dt.to_period("W") >= start_week) & (train_df["date"].dt.to_period("W") <= end_week)
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

    def train_tgn_temporal_batches(self, optimizer, batch_size=16, epochs=3, neg_sample_ratio=1):

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

                model_out = self.model(x, edge_index=edge_index, memory=memory)

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

                if getattr(self, "_train_decoder", None) is None and getattr(self, "decoder", None) is None:
                    d = z.size(1)
                    self._train_decoder = nn.Sequential(
                        nn.Linear(2 * d, d),
                        nn.ReLU(),
                        nn.Linear(d, 1)
                    ).to(self.device)
                    decoder = self._train_decoder
                    optimizer.add_param_group({"params": self._train_decoder.parameters()})
                elif getattr(self, "decoder", None) is None:
                    decoder = self._train_decoder

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
