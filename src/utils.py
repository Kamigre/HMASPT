import os
import json
import math
from typing import Any
import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


def half_life(spread: np.ndarray) -> float:
    """
    Compute half-life of mean reversion for a spread series following
    the approach: fit AR(1): ds_t = a + b * s_{t-1} + eps, half-life = -ln(2) / ln(b)
    If b >= 1 or the regression fails, return a large number.
    """
    if not STATSMODELS_AVAILABLE:
        return np.inf
    
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
    if not STATSMODELS_AVAILABLE:
        return pd.Series(dtype=float)
    
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
    """Save object to JSON file with proper serialization."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x), indent=2)
