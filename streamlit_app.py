import io
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from features import (
    parse_dataset,
    add_technical_features,
    make_supervised,
    train_model,
    evaluate,
    get_feature_importance,
    permutation_importance_df,
    simple_backtest,
    detect_sentiment_columns,
    save_model,
)
from telemetry import log_event
import time


st.set_page_config(page_title="Sentiment-Driven Crypto Predictor (Baseline)", layout="wide")


@st.cache_data(show_spinner=False)
def load_default_csv() -> pd.DataFrame:
    df = pd.read_csv('dataset.csv')
    return parse_dataset(df)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return parse_dataset(df)


def sidebar_controls(df: pd.DataFrame) -> dict:
    st.sidebar.header("Controls")
    role = st.sidebar.selectbox("Role", ["Trader", "Analyst", "Researcher"], index=1)
    symbols = df['Symbol'].dropna().unique().tolist() if 'Symbol' in df.columns else []
    symbol = st.sidebar.selectbox("Symbol", options=symbols or ['(all)'])
    min_d, max_d = df['Date'].min(), df['Date'].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_d.date(), max_d.date()) if pd.notnull(min_d) and pd.notnull(max_d) else None,
    )
    horizon = st.sidebar.select_slider("Forecast Horizon (days)", options=[1, 7], value=1)
    auto_select_best = st.sidebar.toggle("Auto-select best model", value=True)
    model_name = st.sidebar.selectbox("Model", ["Linear Regression", "Random Forest", "KNN"], disabled=auto_select_best)
    test_ratio = st.sidebar.slider("Test Size (%)", 10, 40, 20, step=5)
    threshold = st.sidebar.slider("Backtest Threshold (predicted return)", -0.02, 0.05, 0.0, step=0.005)
    # Hyperparameters (Researcher only)
    params = {}
    if role == 'Researcher':
        st.sidebar.markdown("### Hyperparameters")
        rf_estimators = st.sidebar.slider("RF trees", 100, 800, 400, step=50)
        knn_neighbors = st.sidebar.slider("KNN neighbors", 3, 50, 10, step=1)
        params.update({'rf_estimators': rf_estimators, 'knn_neighbors': knn_neighbors})
    return {
        'role': role,
        'symbol': symbol,
        'date_range': date_range,
        'horizon': horizon,
        'model_name': model_name,
        'auto_select_best': auto_select_best,
        'test_ratio': test_ratio / 100.0,
        'threshold': threshold,
        'params': params,
    }


def filter_df(df: pd.DataFrame, symbol: Optional[str], date_range) -> pd.DataFrame:
    out = df.copy()
    if symbol and symbol != '(all)' and 'Symbol' in out.columns:
        out = out[out['Symbol'] == symbol]
    if date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        out = out[(out['Date'] >= start) & (out['Date'] <= end)]
    return out


def render_overview():
    st.title("Social-Sentiment-Driven Crypto Price Predictor (Baseline)")
    st.caption("Implements PRD core: EDA, interpretable models, feature importance, backtesting.")
    st.markdown(
        "This baseline uses market data only (no sentiment columns present in dataset). "
        "You can optionally upload a merged dataset with sentiment features to extend models."
    )


def render_eda(df: pd.DataFrame):
    st.subheader("Exploratory Data Analysis")
    c1, c2 = st.columns((3, 2))
    with c1:
        st.markdown("Price over time")
        st.line_chart(df.set_index('Date')[['Close']])
    with c2:
        st.markdown("Summary statistics")
        st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap']].describe())
    st.markdown("Correlation matrix")
    num = df.select_dtypes(include=[np.number])
    if len(num.columns) >= 2:
        corr = num.corr(numeric_only=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(corr.values, cmap='coolwarm', aspect='auto')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=8)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for a correlation matrix.")
    # Sentiment vs Price correlation (if sentiment-like columns exist)
    sent_cols = detect_sentiment_columns(df)
    if sent_cols:
        st.markdown("Sentiment vs Price Correlation")
        tgt = df[['Date', 'Close']].copy()
        tgt['ret_1d'] = tgt['Close'].pct_change()
        data = pd.concat([tgt[['ret_1d']], df[sent_cols]], axis=1).dropna()
        corr = data.corr(numeric_only=True)[sent_cols].loc[['ret_1d']]
        st.dataframe(corr)


def render_modeling(df: pd.DataFrame, controls: dict):
    st.subheader("Model Training and Evaluation")
    # Feature engineering
    with st.spinner("Computing technical features ..."):
        fe = add_technical_features(df)
    # Supervised dataset
    X, y = make_supervised(fe.drop(columns=['Name']) if 'Name' in fe.columns else fe, horizon=controls['horizon'])
    # Define feature set: drop non-numeric and identifiers
    drop_cols = {'Date', 'Symbol'}
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    X_num = X.select_dtypes(include=[np.number]).copy()
    # Extra safety: remove any inf/NaN rows left after supervised framing
    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    valid_idx = X_num.dropna().index.intersection(y.dropna().index)
    X_num = X_num.loc[valid_idx]
    y = y.loc[valid_idx]
    # Train-test split by time (no shuffling)
    n = len(X_num)
    if n < 100:
        st.warning("Dataset after feature engineering is quite small; results may be unstable.")
    split = int((1 - controls['test_ratio']) * n)
    X_train, X_test = X_num.iloc[:split], X_num.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    # Guard: ensure we have at least 1 sample in both splits
    if split < 1 or (n - split) < 1 or X_train.empty or X_test.empty or len(y_train) == 0 or len(y_test) == 0:
        st.warning("Not enough data after cleaning to create train/test splits. Try widening the date range or reducing the horizon/test size.")
        return
    # Train and optionally auto-select best model
    model_candidates = ["Linear Regression", "Random Forest", "KNN"]
    model_metrics = []
    selected_model_name = controls['model_name']
    selected_model = None
    y_pred = None
    if controls.get('auto_select_best', False):
        for name in model_candidates:
            try:
                m = train_model(X_train, y_train, name)
                p = m.predict(X_test)
                met = evaluate(y_test, p)
                model_metrics.append({"model": name, **met})
            except Exception:
                model_metrics.append({"model": name, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan})
        # Pick by lowest RMSE
        valid = [mm for mm in model_metrics if np.isfinite(mm.get('RMSE', np.nan))]
        if valid:
            best = sorted(valid, key=lambda d: d['RMSE'])[0]
            selected_model_name = best['model']
        st.markdown("Model comparison (validation)")
        if model_metrics:
            st.dataframe(pd.DataFrame(model_metrics))
    # Train selected model on train split for evaluation display
    if selected_model_name.lower().startswith('random') and controls['role'] == 'Researcher':
        from sklearn.ensemble import RandomForestRegressor
        selected_model = RandomForestRegressor(n_estimators=controls['params'].get('rf_estimators', 400), random_state=42, n_jobs=-1)
        selected_model.fit(X_train, y_train)
    elif selected_model_name.lower().startswith('k') and controls['role'] == 'Researcher':
        from sklearn.neighbors import KNeighborsRegressor
        selected_model = KNeighborsRegressor(n_neighbors=controls['params'].get('knn_neighbors', 10), weights='distance')
        selected_model.fit(X_train, y_train)
    else:
        selected_model = train_model(X_train, y_train, selected_model_name)
    t0 = time.perf_counter()
    y_pred = selected_model.predict(X_test)
    latency_ms = (time.perf_counter() - t0) * 1000
    metrics = evaluate(y_test, y_pred)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"Metrics - {selected_model_name}")
        st.json(metrics)
        st.caption(f"Prediction latency: {latency_ms:.1f} ms on validation batch")
    with c2:
        st.markdown("Feature importance / coefficients")
        fi = get_feature_importance(selected_model, X_train.columns)
        st.dataframe(fi.head(20))
        if fi['value'].isna().all():
            st.markdown("Permutation importance (model-agnostic)")
            perm = permutation_importance_df(selected_model, X_test, y_test, n_repeats=5)
            st.dataframe(perm.head(20))
    # Predictions table
    st.markdown("Predictions vs Actuals")
    pred_df = pd.DataFrame({
        'Date': df['Date'].iloc[-len(y_test):].values,
        'Actual_Close': y_test.values,
        'Pred_Close': y_pred,
    })
    st.dataframe(pred_df.tail(50), use_container_width=True)
    # Backtest
    st.markdown("Backtest: simple long-only strategy")
    bt = simple_backtest(
        dates=pred_df['Date'],
        close_series=df['Close'].iloc[-len(y_test):],
        y_pred=y_pred,
        horizon=controls['horizon'],
        threshold=controls['threshold'],
    )
    if not bt.empty:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(bt['Date'], bt['buy_hold_cum'], label='Buy & Hold')
        ax.plot(bt['Date'], bt['strategy_cum'], label='Strategy')
        ax.legend()
        ax.set_ylabel('Cumulative Return (x)')
        st.pyplot(fig, use_container_width=True)
        st.caption(
            "Naive strategy: hold next period when predicted return exceeds threshold; "
            "shown for interpretability and comparison with buy-and-hold."
        )
    else:
        st.info("Backtest not available due to insufficient data after alignment.")
    # Train final model on all available data for forecasting next period and predict
    try:
        final_model = train_model(X_num, y, selected_model_name)
        # Build a full feature matrix aligned to training columns for the latest available row
        full_feats = fe.copy()
        full_feats_num = full_feats.drop(columns=[c for c in ['Date', 'Symbol', 'Name'] if c in full_feats.columns])
        full_feats_num = full_feats_num.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        # Restrict to training feature columns
        missing_cols = [c for c in X_num.columns if c not in full_feats_num.columns]
        for c in missing_cols:
            full_feats_num[c] = 0.0
        full_feats_num = full_feats_num[X_num.columns]
        future_idx = full_feats_num.dropna().index
        if len(future_idx) > 0:
            last_idx = future_idx[-1]
            X_future = full_feats_num.loc[[last_idx]]
            pred_next = float(final_model.predict(X_future)[0])
            last_date = fe.loc[last_idx, 'Date'] if 'Date' in fe.columns else None
            last_close = float(df.loc[last_idx, 'Close']) if last_idx in df.index else float(df['Close'].iloc[-1])
            pred_date = (pd.to_datetime(last_date) + pd.to_timedelta(controls['horizon'], unit='D')) if last_date is not None else None
            delta = pred_next - last_close
            pct = (pred_next / last_close - 1.0) if last_close != 0 else np.nan
            st.markdown("---")
            st.subheader("Next Period Forecast (best model)")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Predicted Close", f"{pred_next:,.4f}", f"{delta:,.4f}")
            with m2:
                st.metric("Predicted Return", f"{pct*100:,.2f}%")
            with m3:
                st.metric("Forecast Date", f"{pred_date.date() if pred_date is not None else 'N/A'}")
            # Trader view: simple action suggestion
            if controls['role'] == 'Trader':
                st.info("Signal: " + ("BUY" if pct > controls['threshold'] else "HOLD"))
                # Trader manual input scenario
                st.markdown("Manual Input Scenario")
                with st.expander("Enter custom values for today's data", expanded=False):
                    # Override today's close and volume (if present)
                    default_close = float(last_close)
                    override_close = st.number_input(
                        "Today's Close (override)", value=default_close, step=0.01, format="%.6f"
                    )
                    override_volume = None
                    if 'Volume' in df.columns:
                        try:
                            default_vol = float(df.loc[last_idx, 'Volume']) if last_idx in df.index else float(df['Volume'].iloc[-1])
                        except Exception:
                            default_vol = float(df['Volume'].iloc[-1])
                        vol_step = 1_000.0 if default_vol >= 1_000 else 1.0
                        override_volume = st.number_input(
                            "Today's Volume (override)", value=default_vol, step=vol_step
                        )
                    # Optional sentiment bump
                    sent_cols_all2 = detect_sentiment_columns(fe)
                    bump2 = 0
                    if sent_cols_all2:
                        bump2 = st.slider("Sentiment bump (%) [optional]", -50, 200, 0, step=5)
                    if st.button("Predict with manual inputs"):
                        # Build a scenario dataframe with overridden raw fields for the most recent date
                        df_scen = df.copy()
                        if last_idx in df_scen.index:
                            if 'Close' in df_scen.columns:
                                df_scen.loc[last_idx, 'Close'] = override_close
                            if override_volume is not None and 'Volume' in df_scen.columns:
                                df_scen.loc[last_idx, 'Volume'] = override_volume
                        else:
                            # Fallback to last available row
                            if 'Close' in df_scen.columns:
                                df_scen.loc[df_scen.index[-1], 'Close'] = override_close
                            if override_volume is not None and 'Volume' in df_scen.columns:
                                df_scen.loc[df_scen.index[-1], 'Volume'] = override_volume
                        # Recompute features and align to training columns
                        fe_scen = add_technical_features(df_scen)
                        full_feats_num_scen = fe_scen.drop(columns=[c for c in ['Date', 'Symbol', 'Name'] if c in fe_scen.columns])
                        full_feats_num_scen = full_feats_num_scen.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
                        missing_cols_scen = [c for c in X_num.columns if c not in full_feats_num_scen.columns]
                        for c in missing_cols_scen:
                            full_feats_num_scen[c] = 0.0
                        full_feats_num_scen = full_feats_num_scen[X_num.columns]
                        scen_idx = full_feats_num_scen.dropna().index
                        if len(scen_idx) > 0:
                            use_idx = last_idx if last_idx in scen_idx else scen_idx[-1]
                            X_future_scen = full_feats_num_scen.loc[[use_idx]].copy()
                            if sent_cols_all2 and bump2 != 0:
                                for c in sent_cols_all2:
                                    if c in X_future_scen.columns:
                                        X_future_scen[c] = X_future_scen[c] * (1 + bump2 / 100.0)
                            pred_manual = float(final_model.predict(X_future_scen)[0])
                            delta_manual = pred_manual - float(override_close)
                            pct_manual = (pred_manual / float(override_close) - 1.0) if float(override_close) != 0 else np.nan
                            st.success(f"Manual-input predicted close: {pred_manual:,.4f} ({pct_manual*100:,.2f}%)")
                        else:
                            st.info("Not enough recent data to compute features for the manual input scenario.")
            # What-if sentiment scenario: adjust sentiment columns by a factor
            sent_cols_all = detect_sentiment_columns(fe)
            if sent_cols_all:
                st.markdown("What-if: increase sentiment by (%)")
                bump = st.slider("Sentiment bump", -50, 200, 0, step=5)
                adj = full_feats_num.loc[[last_idx]].copy()
                for c in sent_cols_all:
                    if c in adj.columns:
                        adj[c] = adj[c] * (1 + bump / 100.0)
                pred_adj = float(final_model.predict(adj)[0])
                pct_adj = (pred_adj / last_close - 1.0) if last_close != 0 else np.nan
                st.caption(f"What-if predicted close: {pred_adj:,.4f} ({pct_adj*100:,.2f}%)")
            # Save model (Researcher only)
            if controls['role'] == 'Researcher':
                if st.button("Save best model"):
                    path = save_model(final_model, list(X_num.columns))
                    st.success(f"Model saved to: {path if path else 'Not available'}")
        else:
            st.info("Not enough recent data to compute a clean feature row for forecasting.")
    except Exception as e:
        st.warning(f"Unable to produce next-period forecast: {e}")
    # Download predictions
    csv_buf = io.StringIO()
    pred_df.to_csv(csv_buf, index=False)
    st.download_button("Download Predictions CSV", csv_buf.getvalue(), file_name="predictions.csv", mime="text/csv")


def main():
    render_overview()
    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader("Upload dataset CSV (optional)", type=["csv"]) 
    if uploaded is not None:
        df = load_uploaded_csv(uploaded)
    else:
        df = load_default_csv()

    controls = sidebar_controls(df)
    df_filt = filter_df(df, controls['symbol'], controls['date_range'])

    tabs = st.tabs(["EDA", "Modeling", "Exports"])
    with tabs[0]:
        if df_filt.empty:
            st.warning("No data available for selected filters.")
        else:
            render_eda(df_filt)
    with tabs[1]:
        if df_filt.empty:
            st.warning("No data available for modeling with the selected filters.")
        else:
            render_modeling(df_filt, controls)
    with tabs[2]:
        st.subheader("Exports and Downloads")
        # Export filtered dataset
        buf = io.StringIO()
        df_filt.to_csv(buf, index=False)
        st.download_button("Download filtered dataset", buf.getvalue(), file_name="filtered_dataset.csv", mime="text/csv")
        # Export engineered features
        fe = add_technical_features(df_filt)
        fe_buf = io.StringIO()
        fe.to_csv(fe_buf, index=False)
        st.download_button("Download engineered features", fe_buf.getvalue(), file_name="engineered_features.csv", mime="text/csv")
        log_event('export_click', {'type': 'dataset_and_features'})


if __name__ == '__main__':
    main()
