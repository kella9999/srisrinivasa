import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import plotly.graph_objs as go
import xgboost as xgb
import pandas_ta as ta
import time
from datetime import datetime, timedelta

# ----- Page Setup -----
st.set_page_config(page_title="Crypto Candlestick Predictor", layout="wide")
st.title("Crypto Candlestick Prediction Dashboard")

# ----- State Management -----
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = datetime.now()

# ----- Dropdowns -----
coin_options = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "BNB/USDT": "BNBUSDT",
    "XRP/USDT": "XRPUSDT",
    "SOL/USDT": "SOLUSDT",
    "ADA/USDT": "ADAUSDT"
}
interval_map = {
    "1m": "1",
    "3m": "3",
    "5m": "5"
}

st.sidebar.header("Settings")
selected_coin_label = st.sidebar.selectbox("Select Coin", list(coin_options.keys()))
selected_interval_label = st.sidebar.selectbox("Select Timeframe", list(interval_map.keys()))
alert_threshold = st.sidebar.slider("Price Spike Alert (%)", 1.0, 10.0, 5.0)

selected_coin = coin_options[selected_coin_label]
selected_interval = interval_map[selected_interval_label]
interval_seconds = {"1": 60, "3": 180, "5": 300}[selected_interval]

# Tabs
tab1, tab2 = st.tabs(["Dashboard", "Settings"])

# ---- Tab 1: Dashboard (Chart, Data, Model) ----
with tab1:
    st.subheader(f"Live Chart: {selected_coin_label} ({selected_interval_label})")
    tradingview_html = f"""
    <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_x&symbol=BINANCE:{selected_coin}&interval={selected_interval}&theme=dark&style=1&locale=en"
    width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
    """
    components.html(tradingview_html, height=500)

    # Live Data
    st.subheader(f"Live Data Feed: {selected_coin_label} ({selected_interval_label})")
    csv_path = f"data/{selected_coin}_{selected_interval_label}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp', ascending=False).head(20)
        df = df.sort_values(by='timestamp', ascending=True)

        st.dataframe(df.tail(20), use_container_width=True)

        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])
        fig.update_layout(title="Mini OHLC Chart", xaxis_title="Time", yaxis_title="Price", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data found for {selected_coin} at {selected_interval_label}. Ensure WebSocket is running.")

    # Model Prediction
    st.subheader(f"Model Prediction: {selected_coin_label} ({selected_interval_label})")
    model_path = f"models/{selected_coin}_{selected_interval_label}_xgb.pkl"
    if os.path.exists(model_path) and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp', ascending=True)

        # Technical indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['ema'] = ta.ema(df['close'], length=20)
        df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
        df['doji'] = ta.cdl_doji(df['open'], df['high'], df['low'], df['close'])
        df['hammer'] = ta.cdl_hammer(df['open'], df['high'], df['low'], df['close'])
        df = df.dropna()

        # Prepare input
        latest_candle = df.tail(1)[['rsi', 'ema', 'macd', 'doji', 'hammer', 'open', 'high', 'low', 'close', 'volume']]

        # Load model
        model = xgb.XGBClassifier()
        model.load_model(model_path)

        # Predict
        prediction = model.predict(latest_candle)[0]
        pred_label = "UP ⬆️" if prediction == 1 else "DOWN ⬇️"

        # Price spike alert
        if not df.empty:
            last_close = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close
            price_change = ((last_close - prev_close) / prev_close) * 100
            if abs(price_change) > alert_threshold:
                st.warning(f"🚨 Price Spike Alert: {price_change:.2f}% change detected!")

        # Output
        st.metric(label="Predicted Next Candle", value=pred_label)
        st.table(pd.DataFrame({
            "coin": [selected_coin_label],
            "time interval": [selected_interval_label],
            "actual": ["up/down"],
            "predict": [pred_label],
            "true/false": ["Actual vs predict after time pass"],
            "remaining second for time interval": [interval_seconds],
            "Manual refresh button": [st.button("Refresh")]
        }))

        # Auto-refresh
        current_time = datetime.now()
        time_since_last = (current_time - st.session_state['last_update']).total_seconds()
        if time_since_last >= interval_seconds:
            st.session_state['last_update'] = current_time
            st.experimental_rerun()

        st.metric(label="Remaining Seconds", value=f"{max(0, interval_seconds - time_since_last):.0f}")
    else:
        st.error(f"Model or data not found. Ensure {model_path} and {csv_path} exist.")

# ---- Tab 2: Settings ----
with tab2:
    st.subheader("Settings")
    st.write(f"Current Coin: {selected_coin_label}, Timeframe: {selected_interval_label}")
    st.write(f"Price Spike Alert Threshold: {alert_threshold}%")
    if st.button("Reset Data Feed"):
        if os.path.exists(csv_path):
            os.remove(csv_path)
            st.success("Data feed reset. Restart WebSocket to regenerate.")
        else:
            st.warning("No data file to reset.")
    st.write("Order Book Analysis (Coming Soon)")
    st.write("On-Chain Metrics (Next Phase)")
