#!/bin/bash

# Kill any already-running Trader Bot processes so we don't start duplicates
kill_trader_processes() {
    if pgrep -f "strategy_runner.py" > /dev/null 2>&1 || pgrep -f "dashboard.py" > /dev/null 2>&1; then
        echo "Stopping existing Trader Bot processes..."
        pkill -f "strategy_runner.py" 2>/dev/null
        pkill -f "streamlit.*dashboard.py" 2>/dev/null
        sleep 1  # Give processes time to shut down gracefully
    fi
}

# Function to kill background processes on exit
cleanup() {
    echo "Stopping..."
    kill_trader_processes
    exit
}

# Trap SIGINT (Ctrl+C)
trap cleanup SIGINT

echo "🚀 Starting AI Trader Bot..."

# Ensure no other bot instances are running before starting a fresh one
kill_trader_processes

# Start the Strategy Runner in the background
echo "Starting Strategy Runner..."
python strategy_runner.py &
BOT_PID=$!

# Wait a moment for the bot to initialize
sleep 3

# Start the Dashboard
echo "Starting Dashboard..."
python -m streamlit run dashboard.py

# Cleanup on exit (handles both Ctrl+C and normal termination)
cleanup
