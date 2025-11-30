import json
import os
import subprocess
import time
from datetime import date, datetime, time as dt_time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from trader_bot.accounting import estimate_commissions_for_orders
from trader_bot.config import (
    ACTIVE_EXCHANGE,
    BOT_VERSION,
    DASHBOARD_REFRESH_SECONDS,
    LOOP_INTERVAL_SECONDS,
    LLM_MAX_SESSION_COST,
    LLM_PROVIDER,
)
from trader_bot.cost_tracker import CostTracker
from trader_bot.database import TradingDatabase

# --- Helper Functions ---

cost_tracker = CostTracker(ACTIVE_EXCHANGE, llm_provider=LLM_PROVIDER)


def resolve_base_currency(session_stats, account_snapshot, fallback="USD"):
    """Prefer explicit base currency hints from session or account snapshots."""
    for source in (session_stats, account_snapshot):
        if not source:
            continue
        base = source.get("base_currency")
        if base:
            return base
    return fallback


def ib_market_hours_status(now: datetime | None = None):
    """Return IB-focused market hours status for ASX cash session and FX."""
    ts = now or datetime.now(timezone.utc)
    syd = ZoneInfo("Australia/Sydney")
    local = ts.astimezone(syd)
    asx_open = local.weekday() < 5 and dt_time(10, 0) <= local.time() <= dt_time(16, 10)

    utc_ts = ts.astimezone(timezone.utc)
    utc_time = utc_ts.time()
    weekday = utc_ts.weekday()
    # FX weekend closure: Fri 21:00 UTC through Sun 21:00 UTC
    fx_closed = (weekday == 4 and utc_time >= dt_time(21, 0)) or weekday == 5 or (weekday == 6 and utc_time < dt_time(21, 0))
    fx_open = not fx_closed

    return [
        {"label": "ASX cash", "is_open": asx_open, "window": "10:00-16:00 AEST"},
        {"label": "FX (~24/5)", "is_open": fx_open, "window": "Closes Fri 21:00 UTC; reopens Sun 21:00 UTC"},
    ]


def build_market_hours_status(exchange: str, now: datetime | None = None):
    if exchange == "IB":
        return ib_market_hours_status(now)
    return []


def extract_circuit_status(health_states):
    """Summarize exchange/tool circuit state from persisted health rows."""
    summary = {}
    for entry in health_states or []:
        key = entry.get("key")
        if key not in {"exchange_circuit", "tool_circuit", "market_data"}:
            continue
        summary[key] = {
            "status": (entry.get("value") or "unknown").lower(),
            "detail": summarize_health_detail(entry.get("detail")),
            "updated_at": entry.get("updated_at"),
        }
    return summary


def build_venue_status_payload(exchange: str, session_stats, account_snapshot, health_states, now: datetime | None = None):
    base_currency = resolve_base_currency(session_stats, account_snapshot, "USD")
    return {
        "venue": exchange or "Unknown",
        "base_currency": base_currency,
        "market_hours": build_market_hours_status(exchange, now),
        "circuit": extract_circuit_status(health_states),
    }


def format_status_badge(label: str, is_open: bool, window: str | None = None) -> str:
    """Lightweight badge HTML for market hours display."""
    color = "#2ecc71" if is_open else "#e74c3c"
    bg = "rgba(46, 204, 113, 0.12)" if is_open else "rgba(231, 76, 60, 0.12)"
    state = "Open" if is_open else "Closed"
    window_text = f"<span style='color:#7f8c8d;font-weight:500;'> — {window}</span>" if window else ""
    return (
        "<span style=\"display:inline-flex;align-items:center;gap:6px;"
        "padding:4px 10px;border-radius:999px;font-weight:600;"
        "font-size:0.9rem;line-height:1;background:%s;color:%s;\">"
        "<span style='font-size:0.85rem;'>%s</span><span>%s</span>%s</span>"
        % (bg, color, "●", f"{label}: {state}", window_text)
    )


def format_venue_badge(venue: str, base_currency: str) -> str:
    """Small badge row highlighting active venue and base currency."""
    venue_label = (venue or "Unknown").upper()
    base_label = base_currency or "USD"
    parts = [
        "<div style=\"display:flex;align-items:center;gap:8px;flex-wrap:wrap;\">",
        "<span style='padding:6px 12px;border-radius:12px;font-weight:700;"
        "background:linear-gradient(135deg,#0fbcf9,#00a8ff);color:#0b1b2b;'>",
        f"Venue: {venue_label}</span>",
        "<span style='padding:6px 12px;border-radius:12px;font-weight:700;"
        "background:rgba(0,0,0,0.05);color:#2c3e50;border:1px solid rgba(0,0,0,0.05);'>",
        f"Base: {base_label}</span>",
        "</div>",
    ]
    return "".join(parts)


def format_circuit_badges(circuit_state: dict) -> str:
    """Format exchange/tool circuit status into pill badges."""
    if not circuit_state:
        return ""
    labels = {
        "exchange_circuit": "Exchange",
        "tool_circuit": "Tools",
        "market_data": "Market Data",
    }
    parts = []
    for key, label in labels.items():
        entry = circuit_state.get(key)
        if not entry:
            continue
        status = (entry.get("status") or "unknown").lower()
        color = "#2ecc71" if status == "ok" else "#f39c12" if status == "degraded" else "#e74c3c"
        bg = "rgba(46, 204, 113, 0.12)" if status == "ok" else "rgba(243, 156, 18, 0.12)" if status == "degraded" else "rgba(231, 76, 60, 0.12)"
        detail = entry.get("detail") or ""
        detail_text = f"<span style='color:#7f8c8d;font-weight:500;'> — {detail}</span>" if detail else ""
        parts.append(
            "<span style=\"display:inline-flex;align-items:center;gap:6px;"
            "padding:4px 10px;border-radius:999px;font-weight:650;"
            f"background:{bg};color:{color};\">"
            f"<span style='font-size:0.85rem;'>●</span><span>{label}: {status.upper()}</span>{detail_text}</span>"
        )
    return "<div style='display:flex;flex-wrap:wrap;gap:6px;'>" + "".join(parts) + "</div>"


def format_currency_value(amount: float, currency: str) -> str:
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{amount:,.2f}"

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


def load_history(session_id, user_timezone):
    """Load trade history from the SQLite database for a session."""
    if not session_id:
        return pd.DataFrame()
    try:
        user_timezone = user_timezone or ZoneInfo("UTC")
        db = TradingDatabase()
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT timestamp, symbol, action, price, quantity, fee, liquidity, realized_pnl, reason FROM trades WHERE session_id = ?",
            (session_id,)
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

def load_session_stats(session_id):
    if not session_id:
        return None, None
    try:
        db = TradingDatabase()
        stats = db.get_session_stats(session_id)
        db.close()
        return stats, session_id
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


def get_llm_burn_stats(session_stats):
    if not session_stats:
        return None
    try:
        start = session_stats.get("created_at") or session_stats.get("date")
        total = session_stats.get("total_llm_cost", 0.0) or 0.0
        return cost_tracker.calculate_llm_burn(total, start, budget=LLM_MAX_SESSION_COST)
    except Exception:
        return None

def load_health_state():
    db = TradingDatabase()
    try:
        return db.get_health_state()
    finally:
        db.close()

def load_account_snapshot(session_id):
    if not session_id:
        return None
    db = TradingDatabase()
    try:
        return db.get_latest_account_snapshot(session_id)
    finally:
        db.close()

def build_order_price_lookup(open_orders, latest_prices):
    lookup = dict(latest_prices or {})
    for order in open_orders or []:
        sym = order.get("symbol")
        price = order.get("price")
        if sym and price and sym not in lookup:
            lookup[sym] = price
    return lookup

def summarize_health_detail(detail):
    if not detail:
        return ""
    try:
        parsed = json.loads(detail)
        if isinstance(parsed, dict):
            parts = []
            for key, val in parsed.items():
                if val is None:
                    continue
                parts.append(f"{key}: {val}")
            return ", ".join(parts)
        return str(parsed)
    except Exception:
        return str(detail)

# --- Page Config ---
st.set_page_config(
    page_title="Dennis-Day Trading Bot",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

user_timezone = get_user_timezone()
timezone_label = get_timezone_label(user_timezone)
db_version_lookup = TradingDatabase()
current_session_id = db_version_lookup.get_session_id_by_version(BOT_VERSION)
available_versions = db_version_lookup.list_bot_versions()
db_version_lookup.close()
health_states = load_health_state()
session_stats, session_id = load_session_stats(current_session_id)
current_trades_df = load_history(session_id, user_timezone)
open_orders_cached = load_open_orders(session_id) if session_id else []
account_snapshot = load_account_snapshot(session_id)
current_prices = {}
if session_stats and not current_trades_df.empty:
    symbols = current_trades_df['symbol'].unique()
    current_prices = get_latest_prices(session_id, symbols)
venue_status = build_venue_status_payload(ACTIVE_EXCHANGE, session_stats, account_snapshot, health_states)
base_currency_label = venue_status.get("base_currency") or "USD"

col_header, col_status = st.columns([3, 1])
with col_header:
    st.title("🤖 Dennis-Day Trading Bot")
with col_status:
    st.markdown(f"<div style='padding-top: 20px;'>{format_venue_badge(venue_status.get('venue'), base_currency_label)}</div>", unsafe_allow_html=True)

# --- Sidebar ---





st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Bot Status")
st.sidebar.caption(f"Auto-refresh every {DASHBOARD_REFRESH_SECONDS} seconds")
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
tab_live, tab_costs, tab_health, tab_history = st.tabs(["Current Version", "Costs", "System Health", "History"])

with tab_live:
    mh_badges = "".join(
        format_status_badge(entry.get("label"), entry.get("is_open", False), entry.get("window"))
        for entry in (venue_status.get("market_hours") or [])
        if entry.get("label")
    )
    if mh_badges:
        st.markdown(f"<div style='display:flex;flex-wrap:wrap;gap:6px;'>{mh_badges}</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Current Performance")
        df = current_trades_df.copy()
        
        if session_stats and not df.empty:
            realized_pnl, unrealized_pnl, active_positions, df, exposure, trade_spacing = calculate_pnl(df, current_prices)
            open_orders = list(open_orders_cached)
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
            
            gross_pnl = realized_pnl + unrealized_pnl
            currency_symbol = base_currency_label

            fees = session_stats.get('total_fees', 0)
            llm_cost = session_stats.get('total_llm_cost', 0)
            net_pnl = gross_pnl - fees - llm_cost
            total_costs = fees + llm_cost
            
            cost_ratio = (total_costs / abs(gross_pnl) * 100) if gross_pnl != 0 else 0
            fee_ratio = (fees / abs(gross_pnl) * 100) if gross_pnl != 0 else 0
            fee_badge = format_ratio_badge(fee_ratio if gross_pnl else None)
            cost_badge = format_ratio_badge(cost_ratio if gross_pnl else None)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Net PnL ({currency_symbol})", format_currency_value(net_pnl, currency_symbol))
            m2.metric(f"Gross PnL ({currency_symbol})", format_currency_value(gross_pnl, currency_symbol))
            m3.metric("Realized PnL", format_currency_value(realized_pnl, currency_symbol))
            m4.metric("Unrealized PnL", format_currency_value(unrealized_pnl, currency_symbol))

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Fees", format_currency_value(fees, currency_symbol))
            c2.metric("LLM Costs", f"{currency_symbol} {llm_cost:,.4f}")
            c3.metric("Total Costs", format_currency_value(total_costs, currency_symbol))
            c4.metric("Total Trades", session_stats.get('total_trades', 0))
            if fee_badge:
                c1.markdown(f"<div style='margin-top:-8px;'>{fee_badge}</div>", unsafe_allow_html=True)
            if cost_badge:
                c3.markdown(f"<div style='margin-top:-8px;'>{cost_badge}</div>", unsafe_allow_html=True)

            burn_stats = get_llm_burn_stats(session_stats)
            if burn_stats:
                burn_rate = burn_stats.get("burn_rate_per_hour", 0.0)
                pct_budget = burn_stats.get("pct_of_budget", 0.0) * 100
                hours_to_cap = burn_stats.get("hours_to_cap")
                remaining_budget = burn_stats.get("remaining_budget", 0.0)
                eta_text = "idle" if hours_to_cap is None else f"~{hours_to_cap:.1f}h to cap"

                b1, b2 = st.columns(2)
                b1.metric("LLM Burn Rate", f"${burn_rate:.4f}/hr", f"{pct_budget:.1f}% of budget")
                b2.metric("LLM Budget Remaining", f"${remaining_budget:.2f}", eta_text)

                if pct_budget >= 80 or (hours_to_cap is not None and hours_to_cap < 2):
                    st.warning(
                        f"LLM budget is {pct_budget:.1f}% used; {eta_text}. Consider widening loop spacing or trimming tool calls.",
                        icon="⚠️",
                    )

            st.markdown("---")
            s1, s2, s3, s4 = st.columns(4)
            avg_spacing = trade_spacing.get("avg_seconds")
            last_spacing = trade_spacing.get("last_seconds")
            spacing_text = f"{avg_spacing:.0f}s avg" if avg_spacing else "n/a"
            spacing_delta = f"{last_spacing:.0f}s last" if last_spacing else None
            s1.metric("Trade Spacing", spacing_text, spacing_delta)
            s2.metric(
                f"Exposure (incl pending)",
                format_currency_value(total_exposure, currency_symbol),
                f"pending {format_currency_value(pending_exposure, currency_symbol)}",
            )
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
            st.info("Bot version started, waiting for trades...")
            st.metric("Starting Balance", f"${session_stats.get('starting_balance', 0):,.2f}")
        else:
            st.warning("No session stats available for this version.")
        
        st.subheader("🧾 Trade History")
        if not df.empty:
            df = df.sort_values('timestamp', ascending=False)
            st.dataframe(
                df[['timestamp', 'symbol', 'action', 'price', 'quantity', 'fee', 'realized_pnl', 'reason']],
                width="stretch",
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn(f"Date/Time ({timezone_label})", format="YYYY-MM-DD HH:mm:ss"),
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "quantity": st.column_config.NumberColumn("Qty", format="%.4f"),
                    "realized_pnl": st.column_config.NumberColumn("Realized PnL", format="$%.2f"),
                    "fee": st.column_config.NumberColumn("Fee", format="$%.4f"),
                    "reason": st.column_config.TextColumn("Reason", width="large"),
                }
            )
        else:
            st.info("No trade history for this version yet.")

    with col2:
        if account_snapshot:
            st.subheader("🏦 Account Summary")
            base_ccy = base_currency_label
            m_a, m_b = st.columns(2)
            net_liq = account_snapshot.get("net_liquidation")
            avail = account_snapshot.get("available_funds")
            excess = account_snapshot.get("excess_liquidity")
            buying = account_snapshot.get("buying_power")
            m_a.metric(f"Net Liq ({base_ccy})", format_currency_value(net_liq, base_ccy) if net_liq is not None else "n/a")
            m_b.metric(f"Avail Funds ({base_ccy})", format_currency_value(avail, base_ccy) if avail is not None else "n/a")
            m_c, m_d = st.columns(2)
            m_c.metric(f"Excess Liquidity ({base_ccy})", format_currency_value(excess, base_ccy) if excess is not None else "n/a")
            m_d.metric(f"Buying Power ({base_ccy})", format_currency_value(buying, base_ccy) if buying is not None else "n/a")
            cash_balances = account_snapshot.get("cash_balances") or {}
            if cash_balances:
                cash_rows = [{"currency": k, "cash": v} for k, v in cash_balances.items()]
                st.dataframe(
                    pd.DataFrame(cash_rows),
                    hide_index=True,
                    column_config={
                        "currency": "Currency",
                        "cash": st.column_config.NumberColumn("Cash", format="$%.2f"),
                    },
                    width="stretch",
                    height=180,
                )

        if session_stats and session_id:
            st.subheader("📌 Open Orders")
            open_orders = list(open_orders_cached)
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

            price_lookup = build_order_price_lookup(open_orders, current_prices)
            commission_rows = estimate_commissions_for_orders(open_orders, price_lookup, cost_tracker)
            st.subheader("🧾 Commission Estimates")
            if commission_rows:
                commission_df = pd.DataFrame(commission_rows)
                st.dataframe(
                    commission_df,
                    hide_index=True,
                    column_config={
                        "symbol": "Symbol",
                        "side": "Side",
                        "quantity": st.column_config.NumberColumn("Qty", format="%.4f"),
                        "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                        "estimated_fee": st.column_config.NumberColumn("Est. Fee", format="$%.4f"),
                    },
                    width="stretch",
                    height=220,
                )
            else:
                st.caption("No eligible orders to estimate commissions.")

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

with tab_costs:
    st.subheader("💵 Costs This Session")
    if not session_stats:
        st.info("No session data available yet.")
    else:
        total_fees = session_stats.get('total_fees', 0.0) or 0.0
        total_llm_cost = session_stats.get('total_llm_cost', 0.0) or 0.0
        gross_pnl = session_stats.get('gross_pnl', 0.0) or 0.0
        net_pnl = session_stats.get('net_pnl', 0.0) or (gross_pnl - total_fees - total_llm_cost)
        total_costs = total_fees + total_llm_cost
        cost_ratio = (total_costs / abs(gross_pnl) * 100) if gross_pnl else None

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Fees", format_currency_value(total_fees, base_currency_label))
        c2.metric("LLM Costs", f"{base_currency_label} {total_llm_cost:,.4f}")
        c3.metric("Total Costs", format_currency_value(total_costs, base_currency_label), f"{cost_ratio:.1f}% of gross" if cost_ratio is not None else None)

        c4, c5, c6 = st.columns(3)
        c4.metric("Net PnL", format_currency_value(net_pnl, base_currency_label))
        c5.metric("Gross PnL", format_currency_value(gross_pnl, base_currency_label))
        c6.metric("Total Trades", session_stats.get('total_trades', 0))

        burn_stats = get_llm_burn_stats(session_stats)
        if burn_stats:
            burn_rate = burn_stats.get("burn_rate_per_hour", 0.0)
            pct_budget = burn_stats.get("pct_of_budget", 0.0) * 100
            hours_to_cap = burn_stats.get("hours_to_cap")
            remaining_budget = burn_stats.get("remaining_budget", 0.0)
            eta_text = "idle" if hours_to_cap is None else f"~{hours_to_cap:.1f}h to cap"

            b1, b2, b3 = st.columns(3)
            b1.metric("LLM Burn Rate", f"${burn_rate:.4f}/hr")
            b2.metric("LLM Budget Remaining", f"${remaining_budget:.2f}", eta_text)
            b3.metric("% of Budget Used", f"{pct_budget:.1f}%")
            if pct_budget >= 80 or (hours_to_cap is not None and hours_to_cap < 2):
                st.warning(
                    f"LLM budget is {pct_budget:.1f}% used; {eta_text}. Consider widening loop spacing or trimming tool calls.",
                    icon="⚠️",
                )

        st.markdown("---")
        st.subheader("🧾 Fee Breakdown")
        if not current_trades_df.empty:
            fee_summary = current_trades_df.groupby('symbol').agg(
                total_fee=('fee', 'sum'),
                trades=('symbol', 'count'),
                notional=('trade_value', 'sum'),
            ).reset_index()
            fee_summary['avg_fee_bps'] = fee_summary.apply(
                lambda row: (row['total_fee'] / row['notional'] * 10000) if row['notional'] else 0.0,
                axis=1
            )
            st.dataframe(
                fee_summary,
                hide_index=True,
                column_config={
                    "symbol": "Symbol",
                    "total_fee": st.column_config.NumberColumn("Fees ($)", format="$%.4f"),
                    "trades": st.column_config.NumberColumn("Trades"),
                    "notional": st.column_config.NumberColumn("Notional ($)", format="$%.2f"),
                    "avg_fee_bps": st.column_config.NumberColumn("Avg Fee (bps)", format="%.2f"),
                },
                width="stretch",
            )
        else:
            st.info("No trades logged yet, so no fee data to display.")

with tab_health:
    st.subheader("🚦 System Health")
    if health_states:
        for entry in health_states:
            status_raw = entry.get("value") or "unknown"
            status = status_raw.upper()
            detail = summarize_health_detail(entry.get("detail"))
            updated = entry.get("updated_at")
            color = "#2ecc71" if status_raw == "ok" else "#f39c12" if status_raw == "degraded" else "#e74c3c"
            st.markdown(
                f"<div style='padding:10px 12px;margin-bottom:8px;border-radius:10px;background:rgba(0,0,0,0.02);border-left:5px solid {color};'>"
                f"<strong>{entry.get('key')}</strong>: <span style='color:{color};font-weight:700;'>{status}</span>"
                f"{' — ' + detail if detail else ''}"
                f"{' (' + updated + ')' if updated else ''}"
                "</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No health data available yet.")

with tab_history:
    st.subheader("📚 Historical Versions")
    if not available_versions:
        st.info("No historical versions found yet.")
    else:
        selected_version = st.selectbox("Select bot version", options=available_versions, index=0)
        db_hist = TradingDatabase()
        hist_session_id = db_hist.get_session_id_by_version(selected_version)
        db_hist.close()
        hist_df = load_history(hist_session_id, user_timezone)
        if hist_df.empty:
            st.info("No trades logged for this version.")
        else:
            hist_df = hist_df.sort_values('timestamp', ascending=False)
            st.dataframe(
                hist_df[['timestamp', 'symbol', 'action', 'price', 'quantity', 'fee', 'realized_pnl', 'reason']],
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn(f"Date/Time ({timezone_label})", format="YYYY-MM-DD HH:mm:ss"),
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "quantity": st.column_config.NumberColumn("Qty", format="%.4f"),
                    "realized_pnl": st.column_config.NumberColumn("Realized PnL", format="$%.2f"),
                    "fee": st.column_config.NumberColumn("Fee", format="$%.4f"),
                    "reason": st.column_config.TextColumn("Reason", width="large"),
                },
                width="stretch",
                height=600
            )

# Auto-refresh by sleeping then rerunning the app
time.sleep(DASHBOARD_REFRESH_SECONDS)
st.rerun()
