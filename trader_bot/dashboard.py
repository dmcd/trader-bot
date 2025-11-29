import os
import subprocess
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from trader_bot.config import LOOP_INTERVAL_SECONDS
from trader_bot.database import TradingDatabase

# --- Helper Functions ---

def format_ratio_badge(ratio):
    if ratio is None:
        return None
    color = "#2ecc71"  # green
    label = "Good"
    bg = "rgba(46, 204, 113, 0.15)"
    if ratio > 25:
        color, label, bg = "#e74c3c", "High", "rgba(231, 76, 60, 0.15)"  # red
        arrow = "&uarr;"
    elif ratio > 10:
        color, label, bg = "#f39c12", "Moderate", "rgba(243, 156, 18, 0.15)"  # orange
        arrow = "&nearr;"
    else:
        arrow = "&nearr;"
    return (
        "<span style=\"display:inline-flex;align-items:center;gap:4px;"
        f"padding:2px 8px;border-radius:999px;font-weight:600;"
        f"font-size:0.9rem;line-height:1;background:{bg};color:{color};\">"
        f"<span style='font-size:0.8rem;font-weight:500;'>{arrow}</span>"
        f"<span>{label} ({ratio:.1f}% of Gross)</span>"
        "</span>"
    )

def is_bot_running():
    """Check if the trading bot process is running."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'trader_bot.strategy_runner'],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    except:
        return False

def start_bot():
    """Start the trading bot in the background."""
    repo_root = Path(__file__).resolve().parent.parent
    try:
        subprocess.Popen(
            ['python', '-m', 'trader_bot.strategy_runner'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=repo_root
        )
        return True
    except Exception as e:
        st.error(f"Failed to start bot: {e}")
        return False

def get_user_timezone():
    """Resolve the user's timezone from env or system local time."""
    tz_env = os.getenv("LOCAL_TIMEZONE") or os.getenv("TZ")
    if tz_env:
        try:
            return ZoneInfo(tz_env)
        except Exception:
            st.warning(f"Invalid timezone '{tz_env}', falling back to system local time.")
    try:
        tz = datetime.now().astimezone().tzinfo
        if tz:
            return tz
    except Exception:
        pass
    return ZoneInfo("UTC")


def get_timezone_label(tzinfo):
    if hasattr(tzinfo, "key") and tzinfo.key:
        return tzinfo.key
    try:
        label = tzinfo.tzname(datetime.now(tzinfo))
        if label:
            return label
    except Exception:
        pass
    return "Local"


def load_history(user_timezone):
    """Load trade history from the SQLite database for today's session."""
    try:
        user_timezone = user_timezone or ZoneInfo("UTC")
        db = TradingDatabase()
        today = date.today().isoformat()
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT timestamp, symbol, action, price, quantity, fee, liquidity, realized_pnl, reason FROM trades WHERE session_id = (SELECT id FROM sessions WHERE date = ?)",
            (today,)
        )
        rows = cursor.fetchall()
        db.close()
        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "symbol", "action", "price", "quantity", "fee", "liquidity", "realized_pnl", "reason"])
            df["trade_value"] = df["price"] * df["quantity"]
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601").dt.tz_convert(user_timezone)
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
    realized_pnl = 0.0
    holdings = {}
    
    if 'realized_pnl' in df.columns:
        df['pnl'] = df['realized_pnl'].fillna(0.0)
    else:
        df['pnl'] = 0.0
    
    df = df.sort_values('timestamp', ascending=True)

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
            current_qty = holdings[symbol]['qty']
            current_cost = holdings[symbol]['avg_cost']
            new_qty = current_qty + quantity
            if new_qty > 0:
                new_cost = ((current_qty * current_cost) + (quantity * price)) / new_qty
                holdings[symbol]['qty'] = new_qty
                holdings[symbol]['avg_cost'] = new_cost
            else:
                holdings[symbol]['qty'] = 0.0
                holdings[symbol]['avg_cost'] = 0.0
                
        elif action == 'SELL':
            avg_cost = holdings[symbol]['avg_cost']
            trade_pnl = (price - avg_cost) * quantity  # fee handled separately in totals
            realized_pnl += trade_pnl
            if 'realized_pnl' not in df.columns or pd.isna(df.at[index, 'pnl']):
                df.at[index, 'pnl'] = trade_pnl
            holdings[symbol]['qty'] = max(0.0, holdings[symbol]['qty'] - quantity)
            
    unrealized_pnl = 0.0
    active_positions = []
    exposure = 0.0
    
    for symbol, data in holdings.items():
        qty = data['qty']
        avg_cost = data['avg_cost']
        
        if qty > 1e-8:
            current_price = current_prices.get(symbol, avg_cost)
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
            lines = f.readlines()[-100:]
            return lines[::-1]
    return []

def load_session_stats():
    try:
        db = TradingDatabase()
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

def load_llm_stats(session_id):
    db = TradingDatabase()
    try:
        return db.get_recent_llm_stats(session_id)
    finally:
        db.close()

def load_open_orders(session_id):
    db = TradingDatabase()
    try:
        orders = db.get_open_orders(session_id)
        return orders
    finally:
        db.close()

def load_trade_plans(session_id):
    db = TradingDatabase()
    try:
        plans = db.get_open_trade_plans(session_id)
        # Defensive filter in case stale rows sneak in
        return [p for p in plans if (p.get("status", "open") == "open")]
    finally:
        db.close()

# --- Page Config ---
st.set_page_config(
    page_title="Dennis-Day Trading Bot",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 Dennis-Day Trading Bot")

user_timezone = get_user_timezone()
timezone_label = get_timezone_label(user_timezone)

# --- Sidebar ---





st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Bot Status")
bot_running = is_bot_running()
if bot_running:
    st.sidebar.success("✅ Bot Running")
else:
    st.sidebar.error("⏹️ Bot Stopped")

st.sidebar.markdown("---")
st.sidebar.subheader("🎮 Bot Controls")
db = TradingDatabase()
if bot_running:
    if st.sidebar.button("⏹️ Stop Bot", type="secondary", width="stretch"):
        db.create_command("STOP_BOT")
        st.sidebar.warning("Command sent: Stop Bot")
        time.sleep(1)
        st.rerun()
else:
    if st.sidebar.button("▶️ Start Bot", type="primary", width="stretch"):
        if start_bot():
            st.sidebar.success("Bot started successfully!")
            time.sleep(1)
            st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🛑 Close All Positions", type="secondary", width="stretch"):
    if bot_running:
        db.create_command("CLOSE_ALL_POSITIONS")
        st.sidebar.success("Command sent: Close All Positions")
    else:
        st.sidebar.warning("Bot must be running to close positions")
db.close()



# --- Main Content ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Session Performance")
    df = load_history(user_timezone)
    session_stats, session_id = load_session_stats()
    
    if session_stats and not df.empty:
        symbols = df['symbol'].unique()
        current_prices = get_latest_prices(session_id, symbols)
        realized_pnl, unrealized_pnl, active_positions, df, exposure, trade_spacing = calculate_pnl(df, current_prices)
        open_orders = load_open_orders(session_id)
        pending_exposure = 0.0
        if open_orders:
            for o in open_orders:
                side = (o.get('side') or '').upper()
                if side != 'BUY':
                    continue
                px = o.get('price') or current_prices.get(o.get('symbol'), 0) or 0
                qty = o.get('remaining')
                if qty is None:
                    qty = o.get('amount', 0)
                if px and qty:
                    pending_exposure += px * qty
        total_exposure = exposure + pending_exposure
        
        gross_pnl_usd = realized_pnl + unrealized_pnl
        
        gross_pnl = gross_pnl_usd
        realized_disp = realized_pnl
        unrealized_disp = unrealized_pnl
        currency_symbol = 'USD'

        fees = session_stats.get('total_fees', 0)
        llm_cost = session_stats.get('total_llm_cost', 0)
        net_pnl = gross_pnl - fees - llm_cost
        total_costs = fees + llm_cost
        
        cost_ratio = (total_costs / abs(gross_pnl) * 100) if gross_pnl != 0 else 0
        fee_ratio = (fees / abs(gross_pnl) * 100) if gross_pnl != 0 else 0
        fee_badge = format_ratio_badge(fee_ratio if gross_pnl else None)
        cost_badge = format_ratio_badge(cost_ratio if gross_pnl else None)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Net PnL ({currency_symbol})", f"${net_pnl:,.2f}")
        m2.metric(f"Gross PnL ({currency_symbol})", f"${gross_pnl:,.2f}")
        m3.metric(f"Realized PnL", f"${realized_disp:,.2f}")
        m4.metric(f"Unrealized PnL", f"${unrealized_disp:,.2f}")

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Fees", f"${fees:.2f}")
        c2.metric("LLM Costs", f"${llm_cost:.4f}")
        c3.metric("Total Costs", f"${total_costs:.2f}")
        c4.metric("Total Trades", session_stats.get('total_trades', 0))
        if fee_badge:
            c1.markdown(f"<div style='margin-top:-8px;'>{fee_badge}</div>", unsafe_allow_html=True)
        if cost_badge:
            c3.markdown(f"<div style='margin-top:-8px;'>{cost_badge}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        s1, s2, s3, s4 = st.columns(4)
        avg_spacing = trade_spacing.get("avg_seconds")
        last_spacing = trade_spacing.get("last_seconds")
        spacing_text = f"{avg_spacing:.0f}s avg" if avg_spacing else "n/a"
        spacing_delta = f"{last_spacing:.0f}s last" if last_spacing else None
        s1.metric("Trade Spacing", spacing_text, spacing_delta)
        s2.metric("Exposure (incl pending)", f"${total_exposure:,.2f}", f"pending ${pending_exposure:,.2f}")
        s3.metric("Positions", len(active_positions))
        loop_interval_display = f"{LOOP_INTERVAL_SECONDS}s loop"
        s4.metric("Loop Interval", loop_interval_display)
        
        st.subheader("📈 Active Positions")
        if active_positions:
            pos_df = pd.DataFrame(active_positions)
        else:
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
            width="stretch"
        )

    elif session_stats:
        st.info("Session started, waiting for trades...")
        st.metric("Starting Balance", f"${session_stats.get('starting_balance', 0):,.2f}")
    else:
        st.warning("No session stats available.")
    
    st.subheader("🧾 Trade History")
    if not df.empty:
        df = df.sort_values('timestamp', ascending=False)
        st.dataframe(
            df[['timestamp', 'symbol', 'action', 'price', 'quantity', 'pnl', 'fee', 'liquidity', 'reason']],
            width="stretch",
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(f"Date/Time ({timezone_label})", format="YYYY-MM-DD HH:mm:ss"),
                "symbol": "Symbol",
                "action": "Action",
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "quantity": st.column_config.NumberColumn("Qty", format="%.4f"),
                "pnl": st.column_config.NumberColumn("Realized PnL", format="$%.2f"),
                "fee": st.column_config.NumberColumn("Fee", format="$%.4f"),
                "liquidity": "Liq",
                "reason": st.column_config.TextColumn("Reason", width="large"),
            }
        )
    else:
        st.info("No trade history found yet. Start the bot!")

with col2:
    if session_stats and session_id:
        st.subheader("📌 Open Orders")
        open_orders = load_open_orders(session_id)
        if open_orders:
            oo_df = pd.DataFrame(open_orders)
            oo_df = oo_df.rename(columns={"amount": "quantity"})
            st.dataframe(
                oo_df[['order_id', 'symbol', 'side', 'price', 'quantity', 'remaining', 'status']],
                hide_index=True,
                column_config={
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "quantity": st.column_config.NumberColumn("Qty", format="%.4f"),
                    "remaining": st.column_config.NumberColumn("Remaining", format="%.4f"),
                },
                width="stretch",
                height=200
            )
        else:
            st.info("No open orders")

        st.subheader("🎯 Trade Plans")
        plans = load_trade_plans(session_id)
        if plans:
            tp_df = pd.DataFrame(plans)
            st.dataframe(
                tp_df[['id', 'symbol', 'side', 'size', 'stop_price', 'target_price', 'opened_at']],
                hide_index=True,
                column_config={
                    "size": st.column_config.NumberColumn("Size", format="%.4f"),
                    "stop_price": st.column_config.NumberColumn("Stop", format="$%.2f"),
                    "target_price": st.column_config.NumberColumn("Target", format="$%.2f"),
                    "opened_at": st.column_config.DatetimeColumn("Opened", format="HH:mm:ss"),
                },
                width="stretch",
                height=200
            )
        else:
            st.info("No open trade plans")

    st.subheader("📜 Logs")
    logs = load_logs()
    log_text = "".join(logs)
    st.text_area("Log Output", log_text, height=900, disabled=True)
    time.sleep(5)
    st.rerun()
