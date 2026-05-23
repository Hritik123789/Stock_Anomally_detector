# Real-Time Stock Anomaly Detector

LSTM Autoencoder-based anomaly detection system for live stock data with real-time visualization.

## 🚀 Live Demo

**Dashboard**: [https://stockanomallydetector-8kupsufidl72b7fimwbeky.streamlit.app/](https://stockanomallydetector-8kupsufidl72b7fimwbeky.streamlit.app/)

**Backend API**: [https://stock-anomaly-api.onrender.com](https://stock-anomaly-api.onrender.com)

> **Note**: Free tier services may take 30-60 seconds to wake up from inactivity.

## Features

- **Real-time Data**: Finnhub WebSocket integration with yfinance fallback
- **LSTM Autoencoder**: Deep learning model trained on OHLCV features
- **Dual Detection**: Reconstruction error threshold + volume spike detection
- **Live Dashboard**: Streamlit interface with 3 interactive charts
- **GPU Accelerated**: CUDA support for faster inference

## Architecture

```
┌─────────────┐     WebSocket      ┌──────────────┐     WebSocket     ┌────────────┐
│  Finnhub/   │ ──────────────────> │   FastAPI    │ ────────────────> │ Streamlit  │
│  yfinance   │                     │   Backend    │                   │ Dashboard  │
└─────────────┘                     └──────────────┘                   └────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │ LSTM Model   │
                                    │ (5 features) │
                                    └──────────────┘
```

## Project Structure

```
stock-anomaly-detector/
├── data/
│   ├── fetch_data.py          # yfinance data fetcher for AAPL
│   └── aapl_1m.csv            # Historical training data
├── model/
│   ├── lstm_model.py          # LSTM Autoencoder (input_size=5)
│   ├── preprocess.py          # Data preprocessing & windowing
│   ├── train.py               # Model training script
│   ├── scaler.pkl             # MinMaxScaler (generated)
│   ├── lstm_autoencoder.pt    # Trained model weights (generated)
│   └── X_windows.npy          # Training windows (generated)
├── api/
│   └── main.py                # FastAPI WebSocket server
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Finnhub API (Optional)

If you want live market data during trading hours:
- Sign up at https://finnhub.io/
- Get your free API key
- Replace `FINNHUB_API_KEY` in `api/main.py` (line 17)

**Note**: The system automatically falls back to yfinance if Finnhub is inactive (market closed).

### 3. Train the Model

```bash
# Fetch historical data (optional - data/aapl_1m.csv already included)
python data/fetch_data.py

# Preprocess data and create windows
cd model
python preprocess.py

# Train LSTM Autoencoder
python train.py
cd ..
```

**Training Output**:
- `model/scaler.pkl` - Fitted MinMaxScaler
- `model/X_windows.npy` - Sliding windows (N, 30, 5)
- `model/lstm_autoencoder.pt` - Model weights + threshold

## Running the Application

### Start Backend Server

```bash
python -m uvicorn api.main:app --reload
```

**Expected Output**:
```
Using device: cuda
Model loaded | Threshold: 0.008235
Finnhub thread started...
Server ready.
```

Server runs at: `http://127.0.0.1:8000`

### Start Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard opens at: `http://localhost:8501`

## How It Works

### 1. Data Pipeline
- **Live Mode**: Finnhub WebSocket streams real-time trades during market hours
- **Fallback Mode**: yfinance fetches 5-day 1-minute history when market is closed
- **Buffer**: Maintains rolling 30-window of OHLCV data

### 2. Model Architecture
- **Input**: 30 timesteps × 5 features (Open, High, Low, Close, Volume)
- **Encoder**: 2-layer LSTM (hidden_size=64) compresses sequence
- **Decoder**: 2-layer LSTM reconstructs original sequence
- **Loss**: Mean Squared Error (MSE)

### 3. Anomaly Detection
Two detection mechanisms:
1. **Reconstruction Error**: Flags anomaly if error > 95th percentile threshold
2. **Volume Spike**: Flags anomaly if volume > 5× rolling average

### 4. Dashboard Tabs
- **Price & Anomalies**: Live price chart with anomaly markers
- **Anomaly Score**: Reconstruction error vs threshold
- **Volume**: Volume bars with spike detection

## Configuration

Edit `api/main.py` constants:

```python
FINNHUB_API_KEY = "your_key_here"  # Line 17
TICKER = "AAPL"                     # Line 18
WINDOW_SIZE = 30                    # Line 19
VOLUME_SPIKE_MULTIPLIER = 5         # Line 20
```

## Model Details

- **Training Data**: AAPL 1-minute OHLCV (5 features only)
- **Window Size**: 30 timesteps
- **Epochs**: 20
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Threshold**: 95th percentile of training reconstruction errors
- **Device**: CUDA if available, else CPU

## Troubleshooting

### Model Loading Error
If you see "size mismatch" errors:
```bash
# Delete Python cache
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# Retrain model
cd model
python preprocess.py
python train.py
cd ..
```

### No Data Streaming
- Check if market is open (9:30 AM - 4:00 PM ET, Mon-Fri)
- System automatically uses yfinance fallback when Finnhub is inactive
- Verify `data/aapl_1m.csv` exists for CSV fallback

### Dashboard Not Updating
- Ensure FastAPI server is running on port 8000
- Check WebSocket connection in browser console
- Restart both server and dashboard

## Deployment

### Local Development
Follow the setup instructions above to run locally.

### Cloud Deployment
Choose your preferred platform:

- **[AWS Deployment Guide](DEPLOYMENT_AWS.md)** - EC2, ECS, Lambda options
- **[GCP Deployment Guide](DEPLOYMENT_GCP.md)** - Compute Engine, Cloud Run, GKE options  
- **[Docker Deployment Guide](DEPLOYMENT_DOCKER.md)** - Works on any platform

**Quick Docker Start**:
```bash
docker-compose up --build
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- Internet connection (for live data)

## Notes

- Model trained on 5 OHLCV features (no technical indicators)
- Threshold dynamically loaded from trained model checkpoint
- Volume spike detection helps catch sudden market events
- GPU acceleration significantly speeds up inference
- Free Finnhub tier supports US stocks only
