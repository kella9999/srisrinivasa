import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import plotly.graph_objs as go

# ----- Page Setup -----
st.set_page_config(page_title="Crypto Candlestick Predictor", layout="wide")
st.title("📈 Crypto Candlestick Prediction Dashboard")

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

st.sidebar.header("🔧 Controls")
selected_coin_label = st.sidebar.selectbox("Select Coin", list(coin_options.keys()))
selected_interval_label = st.sidebar.selectbox("Select Timeframe", list(interval_map.keys()))

selected_coin = coin_options[selected_coin_label]
selected_interval = interval_map[selected_interval_label]

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Chart & Prediction", "📡 Live Data Feed", "🤖 Model Prediction"])

# ---- Tab 1: TradingView Chart & Prediction UI ----
with tab1:
    st.subheader(f"📊 Live Chart: {selected_coin_label} ({selected_interval_label})")
    tradingview_html = f"""
    <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_x&symbol=BINANCE:{selected_coin}&interval={selected_interval}&theme=dark&style=1&locale=en&utm_source=local"
    width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
    """
    components.html(tradingview_html, height=500)

    st.subheader("🤖 Model Prediction (Placeholder)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🔮 Predicted Candle", value="📈 UP")
    with col2:
        st.metric(label="🎯 Actual Candle", value="📉 DOWN")
    with col3:
        st.metric(label="📊 Accuracy", value="67%")

# ---- Tab 2: Live Data Feed ----
with tab2:
    st.subheader(f"📡 Live Data Feed: {selected_coin_label} ({selected_interval_label})")
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
        st.warning(f"No data found for {selected_coin} at {selected_interval_label}. Start WebSocket first.")

# ---- Tab 3: Model Prediction Placeholder ----
with tab3:
    st.subheader("🤖 Stage 5: Model Prediction Engine")
    st.info("This tab will run your saved model on recent data and show prediction vs actual results.")
    st.code("Coming soon: LSTM/XGBoost prediction + real-time evaluation")
