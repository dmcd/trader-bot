import streamlit as st
import pandas as pd
import time
import os
import subprocess
from database import TradingDatabase

# Bot status detection
def is_bot_running():
    """Check if the trading bot process is running."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'strategy_runner.py'],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    except:
        return False

def start_bot():
    """Start the trading bot in the background."""
    try:
        subprocess.Popen(
            ['python', 'strategy_runner.py'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return True
    except Exception as e:
        st.error(f"Failed to start bot: {e}")
        return False

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

# Bot Status & Controls
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Bot Status")

# Check if bot is running
bot_running = is_bot_running()

# Display status
if bot_running:
    st.sidebar.success("✅ Bot Running")
else:
    st.sidebar.error("⏹️ Bot Stopped")

st.sidebar.markdown("---")
st.sidebar.subheader("🎮 Bot Controls")

db = TradingDatabase()

# Show Start or Stop button based on bot status
if bot_running:
    if st.sidebar.button("⏹️ Stop Bot", type="secondary", use_container_width=True):
        db.create_command("STOP_BOT")
        st.sidebar.warning("Command sent: Stop Bot")
        time.sleep(1)  # Give it a moment to process
        st.rerun()
else:
    if st.sidebar.button("▶️ Start Bot", type="primary", use_container_width=True):
        if start_bot():
            st.sidebar.success("Bot started successfully!")
            time.sleep(1)  # Give it a moment to start
            st.rerun()

# Close All Positions button (always available)
st.sidebar.markdown("---")
if st.sidebar.button("🛑 Close All Positions", type="secondary", use_container_width=True):
    if bot_running:
        db.create_command("CLOSE_ALL_POSITIONS")
        st.sidebar.success("Command sent: Close All Positions")
    else:
        st.sidebar.warning("Bot must be running to close positions")

db.close()

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
            "SELECT timestamp, symbol, action, price, quantity, fee, liquidity, reason FROM trades WHERE session_id = (SELECT id FROM sessions WHERE date = ?)",
            (today,)
        )
        rows = cursor.fetchall()
        db.close()
        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "symbol", "action", "price", "quantity", "fee", "liquidity", "realized_pnl", "reason"])
            # Compute trade_value as price * quantity
            df["trade_value"] = df["price"] * df["quantity"]

            # Ensure timestamp column is datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading trade history from DB: {e}")
        return pd.DataFrame()

def get_latest_prices(session_id, symbols):
    """Fetch the latest market price for each symbol in the session."""
    prices = {}
    try:
        db = TradingDatabase()
        cursor = db.conn.cursor()
        for symbol in symbols:
            cursor.execute(
                "SELECT price FROM market_data WHERE session_id = ? AND symbol = ? ORDER BY timestamp DESC LIMIT 1",
                (session_id, symbol)
            )
            row = cursor.fetchone()
            if row:
                prices[symbol] = row['price']
        db.close()
    except Exception as e:
        st.error(f"Error fetching prices: {e}")
    return prices

def calculate_pnl(df, current_prices):
    """
    Calculate Realized and Unrealized PnL based on FIFO/AvgCost logic.
    Returns:
        realized_pnl (float)
        unrealized_pnl (float)
        positions (list of dicts)
        df (DataFrame with 'pnl' column added)
        exposure (float)
        trade_spacing (dict)
    """
    realized_pnl = 0.0
    holdings = {} # {symbol: {'qty': 0.0, 'avg_cost': 0.0}}
    
    # Add pnl column to df (prefer stored realized_pnl if present)
    if 'realized_pnl' in df.columns:
        df['pnl'] = df['realized_pnl'].fillna(0.0)
    else:
        df['pnl'] = 0.0
    
    # Sort by timestamp to process chronologically
    df = df.sort_values('timestamp', ascending=True)

    # Trade spacing metrics
    trade_spacing = {"avg_seconds": None, "last_seconds": None}
    if len(df) >= 2:
        deltas = df['timestamp'].diff().dt.total_seconds().dropna()
        if not deltas.empty:
            trade_spacing["avg_seconds"] = deltas.mean()
            trade_spacing["last_seconds"] = deltas.iloc[-1]
    
    for index, row in df.iterrows():
        symbol = row['symbol']
        action = row['action']
        quantity = row['quantity']
        price = row['price']
        
        if symbol not in holdings:
            holdings[symbol] = {'qty': 0.0, 'avg_cost': 0.0}
            
        if action == 'BUY':
            # Update weighted average cost
            current_qty = holdings[symbol]['qty']
            current_cost = holdings[symbol]['avg_cost']
            
            new_qty = current_qty + quantity
            if new_qty > 0:
                new_cost = ((current_qty * current_cost) + (quantity * price)) / new_qty
                holdings[symbol]['qty'] = new_qty
                holdings[symbol]['avg_cost'] = new_cost
            else:
                # Should not happen in long-only, but handle gracefully
                holdings[symbol]['qty'] = 0.0
                holdings[symbol]['avg_cost'] = 0.0
                
        elif action == 'SELL':
            # Calculate realized PnL
            avg_cost = holdings[symbol]['avg_cost']
            trade_pnl = (price - avg_cost) * quantity
            realized_pnl += trade_pnl
            
            # Record PnL for this specific trade if missing
            if 'realized_pnl' not in df.columns or pd.isna(df.at[index, 'pnl']):
                df.at[index, 'pnl'] = trade_pnl
            
            # Update holding quantity
            holdings[symbol]['qty'] = max(0.0, holdings[symbol]['qty'] - quantity)
            # Avg cost doesn't change on sell
            
    # Calculate Unrealized PnL
    unrealized_pnl = 0.0
    active_positions = []
    exposure = 0.0
    
    for symbol, data in holdings.items():
        qty = data['qty']
        avg_cost = data['avg_cost']
        
        if qty > 1e-8: # Only count active positions (ignore tiny dust)
            current_price = current_prices.get(symbol, avg_cost) # Fallback to cost if no price
            position_value = qty * current_price
            cost_basis = qty * avg_cost
            pos_unrealized = position_value - cost_basis
            
            unrealized_pnl += pos_unrealized
            exposure += position_value
            
            active_positions.append({
                'Symbol': symbol,
                'Quantity': qty,
                'Avg Price': avg_cost,
                'Current Price': current_price,
                'Unrealized PnL': pos_unrealized,
                'Value': position_value
            })
            
    return realized_pnl, unrealized_pnl, active_positions, df, exposure, trade_spacing

def load_logs():
    if os.path.exists("bot.log"):
        with open("bot.log", "r") as f:
            lines = f.readlines()[-100:] # Last 100 lines
            return lines[::-1] # Reverse to show newest first
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
            return stats, session_id
        db.close()
        return None, None
    except Exception as e:
        st.error(f"Error loading session stats: {e}")
        return None, None

# --- Main Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Session Performance")
    df = load_history()
    
    # Load session stats
    session_stats, session_id = load_session_stats()
    
    if session_stats and not df.empty:
        # Get latest prices for PnL calculation
        symbols = df['symbol'].unique()
        current_prices = get_latest_prices(session_id, symbols)
        
        # Calculate PnL
        realized_pnl, unrealized_pnl, active_positions, df, exposure, trade_spacing = calculate_pnl(df, current_prices)
        
        gross_pnl_usd = realized_pnl + unrealized_pnl
        
        # Convert to selected currency
        if currency == 'AUD':
            gross_pnl = gross_pnl_usd * usd_to_aud
            realized_disp = realized_pnl * usd_to_aud
            unrealized_disp = unrealized_pnl * usd_to_aud
            currency_symbol = 'AUD'
        else:
            gross_pnl = gross_pnl_usd
            realized_disp = realized_pnl
            unrealized_disp = unrealized_pnl
            currency_symbol = 'USD'

        # Calculate net PnL after fees and LLM costs
        fees = session_stats.get('total_fees', 0)
        llm_cost = session_stats.get('total_llm_cost', 0)
        net_pnl = gross_pnl - fees - llm_cost
        
        total_costs = fees + llm_cost
        cost_ratio = (total_costs / abs(gross_pnl) * 100) if gross_pnl != 0 else 0
        fee_ratio = (fees / abs(gross_pnl) * 100) if gross_pnl != 0 else 0
        
        # Profitability status
        status = "✅ Profitable" if net_pnl > 0 else "❌ Unprofitable"

        # Display extended metrics - Row 1
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Net PnL ({currency_symbol})", f"${net_pnl:,.2f}")
        m2.metric(f"Gross PnL ({currency_symbol})", f"${gross_pnl:,.2f}")
        m3.metric(f"Realized PnL", f"${realized_disp:,.2f}")
        m4.metric(f"Unrealized PnL", f"${unrealized_disp:,.2f}")

        # Cost summary - Row 2
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Fees", f"${fees:.2f}", f"{fee_ratio:.1f}% of Gross" if gross_pnl else None)
        c2.metric("LLM Costs", f"${llm_cost:.4f}")
        c3.metric("Total Costs", f"${total_costs:.2f}", f"{cost_ratio:.1f}% of Gross" if gross_pnl else None)
        c4.metric("Total Trades", session_stats.get('total_trades', 0))
        
        st.markdown("---")
        
        # Trade spacing and exposure row
        s1, s2, s3, s4 = st.columns(4)
        avg_spacing = trade_spacing.get("avg_seconds")
        last_spacing = trade_spacing.get("last_seconds")
        spacing_text = f"{avg_spacing:.0f}s avg" if avg_spacing else "n/a"
        spacing_delta = f"{last_spacing:.0f}s last" if last_spacing else None
        s1.metric("Trade Spacing", spacing_text, spacing_delta)
        s2.metric("Exposure", f"${exposure:,.2f}")
        s3.metric("Positions", len(active_positions))
        s4.metric("Loop Interval", f"{st.session_state.refresh_rate}s refresh")
        
        # Active Positions Table - Always show heading and table
        st.subheader("📈 Active Positions")
        if active_positions:
            pos_df = pd.DataFrame(active_positions)
        else:
            # Create empty DataFrame with the same columns
            pos_df = pd.DataFrame(columns=['Symbol', 'Quantity', 'Avg Price', 'Current Price', 'Unrealized PnL', 'Value'])
        
        st.dataframe(
            pos_df,
            column_config={
                "Symbol": "Symbol",
                "Quantity": st.column_config.NumberColumn("Qty", format="%.4f"),
                "Avg Price": st.column_config.NumberColumn("Avg Price", format="$%.2f"),
                "Current Price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
                "Unrealized PnL": st.column_config.NumberColumn("Unrealized PnL", format="$%.2f"),
                "Value": st.column_config.NumberColumn("Value", format="$%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )

    elif session_stats:
        # Fallback if no trades yet
        st.info("Session started, waiting for trades...")
        st.metric("Starting Balance", f"${session_stats.get('starting_balance', 0):,.2f}")
    else:
        st.warning("No session stats available.")
    
    st.subheader("🧠 Strategy Decisions")
    if not df.empty:
        # Convert timestamp
        # df['timestamp'] = pd.to_datetime(df['timestamp']) # Already done in load_history
        df = df.sort_values('timestamp', ascending=False)
        
        # Dataframe - Rename 'pnl' to 'Trade PnL' for clarity
        st.dataframe(
            df[['timestamp', 'symbol', 'action', 'price', 'quantity', 'pnl', 'fee', 'liquidity', 'reason']],
            width=None, # Use full width
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
                "symbol": "Symbol",
                "action": "Action",
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "quantity": st.column_config.NumberColumn("Qty", format="%.4f"),
                "pnl": st.column_config.NumberColumn("Realized PnL", format="$%.2f"),
                "fee": st.column_config.NumberColumn("Fee", format="$%.4f"),
                "liquidity": "Liq",
                "reason": st.column_config.TextColumn("Reason", width="large"),
            },
            use_container_width=True
        )
    else:
        st.info("No trade history found yet. Start the bot!")

with col2:
    st.subheader("📜    Logs")
    logs = load_logs()
    log_text = "".join(logs)
    st.text_area("Log Output", log_text, height=900, disabled=True)

# --- Auto Refresh ---
time.sleep(refresh_rate)
st.rerun()
