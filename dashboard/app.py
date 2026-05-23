import json
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import websocket

# Use environment variable for deployment, fallback to localhost
FASTAPI_WS_URL = os.getenv("FASTAPI_WS_URL", "ws://localhost:8000/ws")

if "price_history" not in st.session_state:
    st.session_state.price_history = []
if "error_history" not in st.session_state:
    st.session_state.error_history = []
if "anomaly_log" not in st.session_state:
    st.session_state.anomaly_log = []
if "volume_history" not in st.session_state:
    st.session_state.volume_history = []
if "ws" not in st.session_state:
    st.session_state.ws = None
if "total_ticks" not in st.session_state:
    st.session_state.total_ticks = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "last_data_time" not in st.session_state:
    st.session_state.last_data_time = time.time()


def connect_websocket():
    try:
        return websocket.create_connection(FASTAPI_WS_URL, timeout=10)
    except:
        return None


def receive_data(ws):
    try:
        ws.settimeout(0.1)
        return json.loads(ws.recv())
    except websocket.WebSocketTimeoutException:
        return None
    except:
        return "ERROR"


st.set_page_config(page_title="Stock Anomaly Detector", page_icon="📈", layout="wide")
st.title("📈 Real-Time Stock Anomaly Detector")
st.caption("LSTM Autoencoder • AAPL • Reconstruction Error Based Detection")

# sidebar
with st.sidebar:
    st.header("Session Stats")
    uptime = int(time.time() - st.session_state.start_time)
    st.metric("Uptime", f"{uptime}s")
    st.metric("Total Ticks", st.session_state.total_ticks)
    st.metric("Anomalies Detected", len(st.session_state.anomaly_log))

    if len(st.session_state.error_history) > 0:
        rate = len(st.session_state.anomaly_log) / len(st.session_state.error_history) * 100
        st.metric("Anomaly Rate", f"{rate:.1f}%")

    if st.button("Reset", use_container_width=True):
        for key in ["price_history", "error_history", "anomaly_log", "volume_history"]:
            st.session_state[key] = []
        st.session_state.total_ticks = 0
        st.session_state.start_time = time.time()
        st.rerun()

# connect
if st.session_state.ws is None:
    st.session_state.ws = connect_websocket()
    if st.session_state.ws is None:
        st.error("Cannot connect to FastAPI server. Make sure it's running.")
        if st.button("Retry"):
            st.rerun()
        st.stop()

# receive
data = receive_data(st.session_state.ws)

if data == "ERROR":
    st.session_state.ws = None
    time.sleep(1)
    st.rerun()

if data and isinstance(data, dict):
    ts = datetime.fromtimestamp(data["timestamp"] / 1000)
    st.session_state.total_ticks += 1
    st.session_state.last_data_time = time.time()

    st.session_state.price_history.append({"time": ts, "price": data["price"]})
    st.session_state.volume_history.append({"time": ts, "volume": data["volume"]})

    if data["reconstruction_error"] is not None:
        st.session_state.error_history.append({
            "time": ts,
            "error": data["reconstruction_error"],
            "threshold": data["threshold"],
            "is_anomaly": data["is_anomaly"]
        })

    if data["is_anomaly"]:
        st.session_state.anomaly_log.append({
            "time": ts,
            "price": data["price"],
            "volume": data["volume"],
            "error": data["reconstruction_error"]
        })

    # keep last 200 points
    for key in ["price_history", "error_history", "volume_history"]:
        if len(st.session_state[key]) > 200:
            st.session_state[key] = st.session_state[key][-200:]

# metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.session_state.price_history:
        p = st.session_state.price_history
        delta = p[-1]["price"] - p[-2]["price"] if len(p) > 1 else 0
        st.metric("Price", f"${p[-1]['price']:.2f}", f"${delta:+.2f}")
    else:
        st.metric("Price", "Waiting...")

with col2:
    if st.session_state.volume_history:
        st.metric("Volume", f"{st.session_state.volume_history[-1]['volume']:,}")
    else:
        st.metric("Volume", "Waiting...")

with col3:
    if st.session_state.error_history:
        e = st.session_state.error_history[-1]
        st.metric(
            "Reconstruction Error",
            f"{e['error']:.6f}",
            "🚨 ANOMALY" if e["is_anomaly"] else "✅ Normal"
        )
    else:
        st.metric("Reconstruction Error", "Waiting...")

with col4:
    st.metric("Anomalies", len(st.session_state.anomaly_log))

# latest anomaly alert
if st.session_state.anomaly_log:
    a = st.session_state.anomaly_log[-1]
    st.error(f"🚨 Anomaly at {a['time'].strftime('%H:%M:%S')} | Price: ${a['price']:.2f} | Volume: {a['volume']:,} | Error: {a['error']:.6f}")

st.divider()

# charts
tab1, tab2, tab3 = st.tabs(["Price", "Reconstruction Error", "Volume"])

with tab1:
    if len(st.session_state.price_history) > 1:
        fig = go.Figure()
        times = [p["time"] for p in st.session_state.price_history]
        prices = [p["price"] for p in st.session_state.price_history]

        fig.add_trace(go.Scatter(x=times, y=prices, mode="lines", name="Price",
                                  line=dict(color="#667eea", width=2)))

        if st.session_state.anomaly_log:
            at = [a["time"] for a in st.session_state.anomaly_log if a["time"] >= times[0]]
            ap = [a["price"] for a in st.session_state.anomaly_log if a["time"] >= times[0]]
            if at:
                fig.add_trace(go.Scatter(x=at, y=ap, mode="markers", name="Anomaly",
                                          marker=dict(color="red", size=12, symbol="x")))

        fig.update_layout(height=400, template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for data...")

with tab2:
    if len(st.session_state.error_history) > 1:
        fig = go.Figure()
        times = [e["time"] for e in st.session_state.error_history]
        errors = [e["error"] for e in st.session_state.error_history]
        colors = ["red" if e["is_anomaly"] else "#10b981" for e in st.session_state.error_history]
        threshold = st.session_state.error_history[-1]["threshold"]

        fig.add_trace(go.Scatter(x=times, y=errors, mode="lines+markers", name="Error",
                                  line=dict(color="#10b981", width=2),
                                  marker=dict(color=colors, size=6)))

        if threshold:
            fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                          annotation_text=f"Threshold: {threshold:.6f}")

        fig.update_layout(height=400, template="plotly_white", hovermode="x unified",
                          yaxis_title="Reconstruction Error")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for data...")

with tab3:
    if len(st.session_state.volume_history) > 1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[v["time"] for v in st.session_state.volume_history],
            y=[v["volume"] for v in st.session_state.volume_history],
            marker=dict(color="#764ba2", opacity=0.7)
        ))
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for data...")

# anomaly log
if st.session_state.anomaly_log:
    st.subheader("Recent Anomalies")
    df = pd.DataFrame([{
        "Time": a["time"].strftime("%H:%M:%S"),
        "Price": f"${a['price']:.2f}",
        "Volume": f"{a['volume']:,}",
        "Error": f"{a['error']:.6f}"
    } for a in reversed(st.session_state.anomaly_log[-10:])])
    st.dataframe(df, use_container_width=True, hide_index=True)

time.sleep(1)
st.rerun()