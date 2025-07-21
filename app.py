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
st.title("\ud83d\udcc8 Crypto Candlestick Prediction Dashboard")

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

st.sidebar.header("\ud83d\udd27 Settings")
selected_coin_label = st.sidebar.selectbox("Select Coin", list(coin_options.keys()))
selected_interval_label = st.sidebar.selectbox("Select Timeframe", list(interval_map.keys()))
alert_threshold = st.sidebar.slider("Price Spike Alert (%)", 1.0, 10.0, 5.0)

selected_coin = coin_options[selected_coin_label]
selected_interval = interval_map[selected_interval_label]
interval_seconds = {"1": 60, "3": 180, "5": 300}[selected_interval]

# Tabs
tab1, tab2 = st.tabs(["\ud83d\udcca Dashboard", "\u2699\ufe0f Settings"])

# ---- Tab 1: Dashboard (Combined Chart, Live Data, Model Prediction) ----
with tab1:
    # Chart Section
    st.subheader(f"\ud83d\udcca Live Chart: {selected_coin_label} ({selected_interval_label})")
    tradingview_html = f"""
    <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_x&symbol=BINANCE:{selected_coin}&interval={selected_interval}&theme=dark&style=1&locale=en&utm_source=local"
    width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
    """
    components.html(tradingview_html, height=500)

    # Live Data Section
    st.subheader(f"\ud83d\udcf1 Live Data Feed: {selected_coin_label} ({selected_interval_label})")
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

    # Model Prediction Section
    st.subheader(f"\ud83e\udd16 Model Prediction: {selected_coin_label} ({selected_interval_label})")
    clf_path = f"models/xgb_{selected_coin}_{selected_interval_label}_classifier.pkl"
    reg_path = f"models/xgb_{selected_coin}_{selected_interval_label}_regressor.pkl"

    if os.path.exists(clf_path) and os.path.exists(reg_path) and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp', ascending=True)

        # Add technical indicators and candlestick patterns
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['ema'] = ta.ema(df['close'], length=20)
        df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
        df['doji'] = ta.cdl_doji(df['open'], df['high'], df['low'], df['close'])
        df['hammer'] = ta.cdl_hammer(df['open'], df['high'], df['low'], df['close'])
        df = df.dropna()

        latest = df.tail(1)[['rsi', 'ema', 'macd', 'doji', 'hammer', 'open', 'high', 'low', 'close', 'volume']]

        import joblib
        clf = joblib.load(clf_path)
        reg = joblib.load(reg_path)

        direction = clf.predict(latest)[0]
        next_price = reg.predict(latest)[0]
        pred_label = "\ud83d\udcc8 UP" if direction == 1 else "\ud83d\udd3d DOWN"

        # Spike alert
        if not df.empty:
            last_close = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close
            price_change = ((last_close - prev_close) / prev_close) * 100
            if abs(price_change) > alert_threshold:
                st.warning(f"\ud83d\udea8 Price Spike Alert: {price_change:.2f}% change detected!")

        # Show output
        st.metric(label="\ud83d\udd2e Predicted Next Candle", value=pred_label)
        st.info(f"\ud83d\udcb0 Predicted Next Close Price: {round(next_price, 2)} USDT")
        st.table(pd.DataFrame({
            "coin": [selected_coin_label],
            "time interval": [selected_interval_label],
            "actual": ["up/down"],
            "predict": [pred_label],
            "true/false": ["Actual vs predict after time pass"],
            "remaining second for time interval": [interval_seconds],
            "Manual refresh button": [st.button("Refresh")]
        }))

        current_time = datetime.now()
        time_since_last = (current_time - st.session_state['last_update']).total_seconds()
        if time_since_last >= interval_seconds:
            st.session_state['last_update'] = current_time
            st.experimental_rerun()
        st.metric(label="\u23f3 Remaining Seconds", value=f"{max(0, interval_seconds - time_since_last):.0f}")
    else:
        st.error("Model or data not found. Ensure all required files exist.")

# ---- Tab 2: Settings ----
with tab2:
    st.subheader("\u2699\ufe0f Settings")
    st.write("Configure your dashboard preferences here.")
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
