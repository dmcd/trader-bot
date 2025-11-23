import streamlit as st
import pandas as pd
import time
import os
from database import TradingDatabase

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
    """Load trade history from the SQLite database for today's session."""
    try:
        db = TradingDatabase()
        from datetime import date
        today = date.today().isoformat()
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT timestamp, action, price, quantity, fee, reason FROM trades WHERE session_id = (SELECT id FROM sessions WHERE date = ?)",
            (today,)
        )
        rows = cursor.fetchall()
        db.close()
        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "action", "price", "quantity", "fee", "reason"])
            # Compute trade_value as price * quantity
            df["trade_value"] = df["price"] * df["quantity"]

            # Ensure timestamp column is datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading trade history from DB: {e}")
        return pd.DataFrame()

def load_logs():
    if os.path.exists("bot.log"):
        with open("bot.log", "r") as f:
            return f.readlines()[-100:] # Last 100 lines
    return []

def load_session_stats():
    """Load current session statistics from database."""
    try:
        db = TradingDatabase()
        from datetime import date
        today = date.today().isoformat()
        cursor = db.conn.cursor()
        cursor.execute("SELECT id FROM sessions WHERE date = ?", (today,))
        row = cursor.fetchone()
        if row:
            session_id = row['id']
            stats = db.get_session_stats(session_id)
            db.close()
            return stats
        db.close()
        return None
    except Exception as e:
        st.error(f"Error loading session stats: {e}")
        return None

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
        
        # Load session stats for accurate PnL
        session_stats = load_session_stats()
        
        if session_stats:
            gross_pnl_usd = session_stats.get('gross_pnl', 0)
            
            # Convert to selected currency
            if currency == 'AUD':
                gross_pnl = gross_pnl_usd * usd_to_aud
                currency_symbol = 'AUD'
            else:
                gross_pnl = gross_pnl_usd
                currency_symbol = 'USD'

            # Calculate net PnL after fees and LLM costs
            net_pnl = gross_pnl - session_stats.get('total_fees', 0) - session_stats.get('total_llm_cost', 0)
            
            total_costs = session_stats.get('total_fees', 0) + session_stats.get('total_llm_cost', 0)
            cost_ratio = (total_costs / abs(gross_pnl) * 100) if gross_pnl != 0 else 0

            # Display extended metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Net PnL ({currency_symbol})", f"${net_pnl:,.2f}")
            m2.metric(f"Gross PnL ({currency_symbol})", f"${gross_pnl:,.2f}")
            m3.metric("Total Trades", session_stats.get('total_trades', 0))
            m4.metric("LLM Costs", f"${session_stats.get('total_llm_cost', 0):.4f}")

            # Cost summary
            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Fees", f"${session_stats.get('total_fees', 0):.2f}")
            col_b.metric("Total Costs", f"${total_costs:.2f}")
            col_c.metric("Cost Ratio", f"{cost_ratio:.2f}%")

            # Profitability status
            status = "✅ Profitable" if net_pnl > 0 else "❌ Unprofitable"
            st.metric("Status", status)
        else:
            # Fallback if no stats
            st.warning("No session stats available.")
        
        # Dataframe - Rename 'pnl' to 'Trade Value' for clarity
        df = df.rename(columns={'pnl': 'trade_value'})
        st.dataframe(
            df[['timestamp', 'action', 'trade_value', 'reason']],
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
