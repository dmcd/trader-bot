#!/bin/bash

# Function to kill background processes on exit
cleanup() {
    echo "Stopping Trader Bot..."
    kill $BOT_PID
    exit
}

# Trap SIGINT (Ctrl+C)
trap cleanup SIGINT

echo "🚀 Starting AI Trader Bot..."

# Start the Strategy Runner in the background
echo "Starting Strategy Runner..."
python strategy_runner.py &
BOT_PID=$!

# Wait a moment for the bot to initialize
sleep 3

# Start the Dashboard
echo "Starting Dashboard..."
python -m streamlit run dashboard.py

# Wait for the bot process
wait $BOT_PID
