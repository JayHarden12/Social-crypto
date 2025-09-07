# Social-Sentiment-Driven Crypto Predictor (Baseline)

This Streamlit app implements core functionality from the attached PRD using the provided `dataset.csv` (market data). Since the dataset includes no sentiment columns, the app focuses on market-driven features but supports uploading a merged CSV that contains sentiment fields for extended modelling.

## Features
- Roles: Trader, Analyst, Researcher (role-appropriate controls and views)
- EDA: time-series price chart, summary stats, correlation matrix
- Sentiment: detects sentiment-like columns; shows correlation vs returns; what-if slider
- Feature engineering: returns, moving averages, volatility, RSI, lags
- Models: Linear Regression, Random Forest, KNN with auto-select best (by RMSE)
- Hyperparameters: RF trees and KNN neighbors (Researcher role)
- Forecast horizons: 1 day and 7 days; next-period forecast shown with date and return
- Metrics: MAE, RMSE, MAPE, R²; latency for prediction path
- Interpretability: feature importance/coefficients; permutation importance fallback
- Backtest: simple thresholded long-only vs buy-and-hold
- Exports: predictions, filtered dataset, engineered features; optional model saving

## Project Structure
- `streamlit_app.py` — Streamlit UI and workflow
- `features.py` — data parsing, feature engineering, modelling, backtesting helpers
- `dataset.csv` — provided historical OHLCV data (example: AAVE)
- `requirements.txt` — dependencies

## Run Locally
1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the app
   ```bash
   streamlit run streamlit_app.py
   ```
3. In the app sidebar, optionally upload a merged dataset with sentiment columns to extend the feature set.

## Mapping to PRD
- User-friendly interface: Streamlit UI with roles and clear tabs.
- Transparency/interpretability: linear/RF/KNN, coefficients/importances, permutation importance.
- Real-time responsiveness: shows prediction latency; lightweight models and caching.
- Visualization suite: price charts, correlation matrix, sentiment vs return table.
- Back-testing & simulated trading: naive long-only vs buy-and-hold.
- Telemetry: local CSV logging for export actions (extend as needed).

Open items (for a v2):
- Live social ingestion pipeline and alignment (e.g., Twitter/Reddit APIs).
- Scenario analysis UX beyond a single slider; multi-asset support.
- Strategy backtests with transaction costs and risk metrics.

## Notes
- The PRD calls for sentiment integration and role-based UX. This baseline covers market-only modelling with interpretable algorithms and extensible inputs.
- For larger datasets or more sophisticated strategies (e.g., transaction costs, risk metrics), extend the `simple_backtest` logic.
- For feature interpretability on KNN, consider permutation importance in a future iteration.
