# Stock_Anomally_detector
Real-time stock anomaly detector using LSTM Autoencoder + rule-based volume spike detection. Streams live AAPL data via Finnhub WebSocket with yfinance fallback. FastAPI backend broadcasts predictions over WebSocket to a Streamlit dashboard showing live price charts and anomaly alerts.
