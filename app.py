
import streamlit as st
import streamlit.components.v1 as components

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

# ----- TradingView Chart Embed -----
st.subheader(f"📊 Live Chart: {selected_coin_label} ({selected_interval_label})")
tradingview_html = f"""
<iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_x&symbol=BINANCE:{selected_coin}&interval={selected_interval}&theme=dark&style=1&locale=en&utm_source=local"
width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
"""
components.html(tradingview_html, height=500)

# ----- Prediction Panel -----
st.subheader("🤖 Model Prediction (Placeholder)")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="🔮 Predicted Candle", value="📈 UP")
with col2:
    st.metric(label="🎯 Actual Candle", value="📉 DOWN")
with col3:
    st.metric(label="📊 Accuracy", value="67%")

st.info("📌 This is a placeholder. In the next stage, this panel will compare real predictions from your model.")

# ----- Footer -----
st.caption("Built with ❤️ using Streamlit | Stage 3: Chart + Prediction UI")
