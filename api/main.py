import asyncio
import json
import os
import pickle
import threading
import time
from collections import deque

import numpy as np
import pandas as pd
import torch
import websocket
import yfinance as yf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append("model")
from lstm_model import LSTMAutoencoder

FINNHUB_API_KEY = "d88sqgpr01qs9ff61dvgd88sqgpr01qs9ff61e00"
TICKER = "AAPL"
WINDOW_SIZE = 30
VOLUME_SPIKE_MULTIPLIER = 5
THRESHOLD = None

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

ohlcv_buffer = deque(maxlen=WINDOW_SIZE)
current_tick = None
model = None
scaler = None
device = None
finnhub_active = False
yfinance_mode = False


def load_model_and_scaler():
    global model, scaler, device, THRESHOLD

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    checkpoint = torch.load("model/lstm_autoencoder.pt", map_location=device, weights_only=False)
    model = LSTMAutoencoder(input_size=5).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    THRESHOLD = checkpoint["threshold"]
    print(f"Model loaded | Threshold: {THRESHOLD:.6f}")


def detect_anomaly(ohlcv_data):
    global THRESHOLD

    if len(ohlcv_data) < WINDOW_SIZE:
        return None, None

    df = pd.DataFrame(ohlcv_data)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Use only 5 OHLCV features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values.astype(np.float32)

    scaled = scaler.transform(data).astype(np.float32)
    X = torch.tensor(scaled).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed = model(X)
        error = torch.mean((X - reconstructed) ** 2).cpu().item()

    is_anomaly = error > THRESHOLD

    # volume spike rule
    volumes = df['Volume'].values
    rolling_avg = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[-1]
    multiplier = volumes[-1] / rolling_avg if rolling_avg > 0 else 1

    if multiplier > VOLUME_SPIKE_MULTIPLIER:
        is_anomaly = True
        print(f"Volume spike: {multiplier:.1f}x ({volumes[-1]:,.0f} vs avg {rolling_avg:,.0f})")

    print(f"Error: {error:.6f} | Threshold: {THRESHOLD:.6f} | Vol: {multiplier:.2f}x | Anomaly: {is_anomaly}")

    return float(error), is_anomaly


def process_row(row, timestamp):
    global current_tick

    ohlcv = {
        "open": float(row["Open"]),
        "high": float(row["High"]),
        "low": float(row["Low"]),
        "close": float(row["Close"]),
        "volume": int(row["Volume"])
    }
    ohlcv_buffer.append(ohlcv)

    error, is_anomaly = None, False
    if len(ohlcv_buffer) == WINDOW_SIZE:
        error, is_anomaly = detect_anomaly(list(ohlcv_buffer))

    current_tick = {
        "timestamp": int(timestamp * 1000),
        "price": float(row["Close"]),
        "volume": int(row["Volume"]),
        "reconstruction_error": round(error, 6) if error is not None else None,
        "threshold": round(THRESHOLD, 6) if THRESHOLD else None,
        "is_anomaly": bool(is_anomaly)
    }


def fetch_yfinance_data():
    global yfinance_mode
    yfinance_mode = True
    print(f"Starting yfinance fallback for {TICKER}...")

    while True:
        try:
            stock = yf.Ticker(TICKER)
            df = stock.history(period="5d", interval="1m")

            if df.empty:
                print("No data from yfinance. Loading from CSV...")
                df = pd.read_csv("data/aapl_1m.csv", index_col=0, parse_dates=True)

            if df.empty:
                print("No CSV data. Retrying in 60s...")
                time.sleep(60)
                continue

            print(f"Streaming {len(df)} rows...")

            for timestamp, row in df.iterrows():
                ts = timestamp.timestamp() if hasattr(timestamp, 'timestamp') else time.time()
                process_row(row, ts)
                time.sleep(1)

            print("Done streaming. Refreshing in 60s...")
            time.sleep(60)

        except Exception as e:
            print(f"yfinance error: {e}. Retrying in 60s...")
            time.sleep(60)


def on_message(ws, message):
    global finnhub_active
    data = json.loads(message)

    if data.get("type") == "trade":
        trades = data.get("data", [])
        if trades:
            finnhub_active = True

        for trade in trades:
            if trade["s"] == TICKER:
                # build ohlcv from single tick
                ohlcv_row = {
                    "Open": trade["p"],
                    "High": trade["p"],
                    "Low": trade["p"],
                    "Close": trade["p"],
                    "Volume": trade["v"]
                }
                process_row(pd.Series(ohlcv_row), trade["t"] / 1000)


def on_error(ws, error):
    print(f"Finnhub error: {error}")


def on_close(ws, *args):
    print("Finnhub closed. Reconnecting in 5s...")
    threading.Timer(5.0, start_finnhub_connection).start()


def on_open(ws):
    print(f"Finnhub connected. Subscribing to {TICKER}...")
    ws.send(json.dumps({"type": "subscribe", "symbol": TICKER}))


def start_finnhub_connection():
    ws = websocket.WebSocketApp(
        f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws.run_forever()


@app.on_event("startup")
async def startup_event():
    load_model_and_scaler()

    threading.Thread(target=start_finnhub_connection, daemon=True).start()
    print("Finnhub thread started...")

    await asyncio.sleep(5)

    if not finnhub_active:
        print("No Finnhub data. Starting yfinance fallback...")
        threading.Thread(target=fetch_yfinance_data, daemon=True).start()
    else:
        print("Finnhub active.")

    print("Server ready.")


@app.get("/")
async def root():
    return {
        "status": "running",
        "ticker": TICKER,
        "buffer_size": len(ohlcv_buffer),
        "threshold": THRESHOLD,
        "data_source": "Finnhub" if finnhub_active else "yfinance" if yfinance_mode else "waiting"
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_tick = None

    try:
        while True:
            if current_tick and current_tick != last_tick:
                await websocket.send_json(current_tick)
                last_tick = current_tick
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")