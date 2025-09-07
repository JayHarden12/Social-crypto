import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import json
import os

try:
    from joblib import dump, load  # noqa: F401
except Exception:  # pragma: no cover
    dump = None
    load = None


def parse_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct dtypes and sorted dates."""
    df = df.copy()
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    # Parse Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')
    # Numeric casting
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop fully empty rows
    df = df.dropna(how='all')
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic technical indicators suitable for tabular ML."""
    df = df.copy()
    price = df['Close'].astype(float)
    df['ret_1d'] = price.pct_change()
    df['log_ret_1d'] = np.log1p(df['ret_1d'])
    df['ma_7'] = price.rolling(7).mean()
    df['ma_14'] = price.rolling(14).mean()
    df['ema_7'] = price.ewm(span=7, adjust=False).mean()
    df['vol_7'] = df['ret_1d'].rolling(7).std()
    df['vol_14'] = df['ret_1d'].rolling(14).std()
    # RSI(14)
    delta = price.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    df['rsi_14'] = 100 - (100 / (1 + rs))
    # Lags
    for l in [1, 2, 3, 5, 7, 14]:
        df[f'close_lag_{l}'] = price.shift(l)
        df[f'ret_lag_{l}'] = df['ret_1d'].shift(l)
    # Volume features (if present)
    if 'Volume' in df.columns:
        vol = df['Volume']
        df['vol_ma_7'] = vol.rolling(7).mean()
        df['vol_ma_14'] = vol.rolling(14).mean()
        # pct_change can create inf when previous value is 0
        df['vol_chg_1d'] = vol.pct_change()
    # Replace any +/- inf with NaN for safe downstream dropping
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def make_supervised(df: pd.DataFrame, horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """Create features X and target y for a given prediction horizon (days)."""
    if 'Close' not in df.columns:
        raise ValueError('Dataset must contain a Close column')
    df = df.copy()
    df['target_close'] = df['Close'].shift(-horizon)
    # Drop rows with NaNs at the end due to shift
    feat_cols = [c for c in df.columns if c not in ['target_close']]
    valid = df.dropna(subset=feat_cols + ['target_close'])
    X = valid.drop(columns=['target_close'])
    y = valid['target_close']
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, model_name: str):
    model_name = (model_name or 'Linear Regression').lower()
    if model_name in ['linear regression', 'linear']:
        model = LinearRegression(n_jobs=None) if 'n_jobs' in LinearRegression().get_params() else LinearRegression()
    elif model_name in ['random forest', 'rf', 'randomforest']:
        model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    elif model_name in ['knn', 'k-nearest neighbors', 'knearest']:
        model = KNeighborsRegressor(n_neighbors=10, weights='distance')
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    model.fit(X, y)
    return model


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    # Compatibility with older scikit-learn versions (no 'squared' kwarg)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true))
    r2 = r2_score(y_true, y_pred)
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape) if np.isfinite(mape) else np.nan,
        'R2': float(r2),
    }


def get_feature_importance(model, feature_names: pd.Index) -> pd.DataFrame:
    """Extract feature importance or coefficients if available."""
    if hasattr(model, 'feature_importances_'):
        vals = model.feature_importances_
        kind = 'importance'
    elif hasattr(model, 'coef_'):
        vals = np.ravel(model.coef_)
        kind = 'coefficient'
    else:
        return pd.DataFrame({'feature': feature_names, 'value': np.nan})
    return (
        pd.DataFrame({'feature': feature_names, 'value': vals})
        .sort_values('value', ascending=False)
        .reset_index(drop=True)
        .assign(kind=kind)
    )


def permutation_importance_df(model, X_val: pd.DataFrame, y_val: pd.Series, n_repeats: int = 5, random_state: int = 42) -> pd.DataFrame:
    try:
        r = permutation_importance(model, X_val, y_val, n_repeats=n_repeats, random_state=random_state)
    except Exception:
        return pd.DataFrame({'feature': X_val.columns, 'value': np.nan, 'kind': 'permutation'})
    df = pd.DataFrame({'feature': X_val.columns, 'value': r.importances_mean})
    return df.sort_values('value', ascending=False).reset_index(drop=True).assign(kind='permutation')


def detect_sentiment_columns(df: pd.DataFrame) -> List[str]:
    """Heuristic: pick columns that look like sentiment/intensity/engagement metrics."""
    if df is None or df.empty:
        return []
    keywords = ['sent', 'vader', 'polarity', 'subjectivity', 'emo', 'tweet', 'reddit', 'like', 'retweet', 'comment', 'engage']
    cols = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in keywords):
            cols.append(c)
    # keep only numeric
    return [c for c in cols if np.issubdtype(df[c].dtype, np.number)]


def save_model(model, feature_columns: List[str], path_dir: str = 'models', name: str = 'best_model') -> Optional[str]:
    os.makedirs(path_dir, exist_ok=True)
    pkl_path = os.path.join(path_dir, f"{name}.pkl")
    meta_path = os.path.join(path_dir, f"{name}.json")
    if dump is None:
        return None
    dump(model, pkl_path)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'feature_columns': feature_columns}, f)
    return pkl_path



def simple_backtest(dates: pd.Series,
                    close_series: pd.Series,
                    y_pred: np.ndarray,
                    horizon: int = 1,
                    threshold: float = 0.0) -> pd.DataFrame:
    """A naive long-only strategy: if predicted return > threshold, be long next period.

    Returns a DataFrame with strategy vs buy-and-hold cumulative returns.
    """
    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'close_t': close_series.values,
        'pred_close_tp': y_pred,
    }).sort_values('Date')
    # Predicted return over horizon based on today's close
    df['pred_ret'] = (df['pred_close_tp'] / df['close_t']) - 1.0
    # Future realized return over horizon (shifted -horizon already handled before prediction)
    df['fwd_ret'] = df['close_t'].pct_change(periods=horizon).shift(-horizon)
    # Signal for next period
    df['signal'] = (df['pred_ret'] > threshold).astype(int)
    # Strategy return applies the signal to the future realized return
    df['strategy_ret'] = df['signal'] * df['fwd_ret']
    # Cumulative performance
    df['buy_hold_cum'] = (1 + df['fwd_ret'].fillna(0)).cumprod()
    df['strategy_cum'] = (1 + df['strategy_ret'].fillna(0)).cumprod()
    return df.dropna(subset=['buy_hold_cum', 'strategy_cum'])
