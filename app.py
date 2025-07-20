import streamlit as st

st.set_page_config(page_title="Crypto Candlestick Predictor", layout="wide")
st.title("📈 Crypto Candlestick Prediction Dashboard")
st.markdown("Welcome! Choose a coin and timeframe to begin.")

# Placeholder controls
coin = st.selectbox("Select Coin", ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT"])
interval = st.selectbox("Select Timeframe", ["1m", "3m", "5m"])
if st.button("Start Stream"):
    st.success("Streaming started... (placeholder)")
