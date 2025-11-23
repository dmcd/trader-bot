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

refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 60, st.session_state.refresh_rate)
st.session_state.refresh_rate = refresh_rate

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
        current_pnl = latest.get('pnl', 0)
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Current PnL (AUD)", f"${current_pnl:,.2f}")
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
    st.subheader("📜 System Logs")
    logs = load_logs()
    log_text = "".join(logs)
    st.text_area("Log Output", log_text, height=600, disabled=True)

# --- Auto Refresh ---
time.sleep(refresh_rate)
st.rerun()
