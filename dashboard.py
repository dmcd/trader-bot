import streamlit as st
import pandas as pd
import json
import time
import os

# Page Config
st.set_page_config(
    page_title="Trader Bot Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 AI Trader Bot Dashboard")

# Auto-refresh logic
if 'refresh_rate' not in st.session_state:
    st.session_state.refresh_rate = 5

if 'currency' not in st.session_state:
    st.session_state.currency = 'USD'

refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 60, st.session_state.refresh_rate)
st.session_state.refresh_rate = refresh_rate

# Currency toggle
currency = st.sidebar.radio("Display Currency", ['USD', 'AUD'], index=0 if st.session_state.currency == 'USD' else 1)
st.session_state.currency = currency

# Exchange rate (approximate)
usd_to_aud = 1.53  # Update this periodically or fetch from API

# --- Data Loading ---
def load_history():
    data = []
    if os.path.exists("trade_history.jsonl"):
        with open("trade_history.jsonl", "r") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except:
                    pass
    return pd.DataFrame(data)

def load_logs():
    if os.path.exists("bot.log"):
        with open("bot.log", "r") as f:
            return f.readlines()[-100:] # Last 100 lines
    return []

# --- Main Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🧠 Strategy Decisions")
    df = load_history()
    
    if not df.empty:
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        
        # Metrics
        latest = df.iloc[0]
        current_pnl_usd = latest.get('pnl', 0)
        
        # Convert to selected currency
        if currency == 'AUD':
            current_pnl = current_pnl_usd * usd_to_aud
            currency_symbol = 'AUD'
        else:
            current_pnl = current_pnl_usd
            currency_symbol = 'USD'
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Trading PnL ({currency_symbol})", f"${current_pnl:,.2f}")
        m2.metric("Total Decisions", len(df))
        m3.metric("Last Action", latest.get('action', 'N/A'))
        
        # Dataframe
        st.dataframe(
            df[['timestamp', 'action', 'pnl', 'reason', 'market_data']],
            width="stretch",
            hide_index=True
        )
    else:
        st.info("No trade history found yet. Start the bot!")

with col2:
    st.subheader("📜    Logs")
    logs = load_logs()
    log_text = "".join(logs)
    st.text_area("Log Output", log_text, height=600, disabled=True)

# --- Auto Refresh ---
time.sleep(refresh_rate)
st.rerun()
