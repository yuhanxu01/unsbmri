#!/bin/bash
# Stop all running experiments

if [ ! -f logs/pids.txt ]; then
    echo "No PID file found. No experiments to stop."
    exit 0
fi

echo "Stopping all experiments..."

while read pid; do
    if ps -p $pid > /dev/null 2>&1; then
        echo "  Killing PID $pid..."
        kill $pid
    fi
done < logs/pids.txt

echo "All experiments stopped."
rm -f logs/pids.txt
