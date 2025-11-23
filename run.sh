#!/bin/bash

# Function to kill background processes on exit
cleanup() {
    echo "Stopping..."
    
    # Kill ALL strategy_runner.py processes (including ones started via dashboard)
    if pgrep -f "strategy_runner.py" > /dev/null 2>&1; then
        echo "Stopping all Trader Bot processes..."
        pkill -f "strategy_runner.py" 2>/dev/null
        sleep 1  # Give processes time to shut down gracefully
    fi
    
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

# Cleanup on exit (handles both Ctrl+C and normal termination)
cleanup
