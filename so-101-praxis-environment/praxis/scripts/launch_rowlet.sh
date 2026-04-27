#!/bin/bash

# Launch script for Rowlet teleoperation system
# This script launches both the robot node and the teleop node

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Rowlet teleoperation system...${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${RED}Shutting down...${NC}"
    
    # Function to kill a process and its children
    kill_process_tree() {
        local pid=$1
        if [ ! -z "$pid" ] && kill -0 $pid 2>/dev/null; then
            # Kill the process group (negative PID means process group)
            kill -TERM -$pid 2>/dev/null || kill -TERM $pid 2>/dev/null
            # Wait a moment for graceful shutdown
            sleep 0.5
            # Force kill if still running
            kill -9 -$pid 2>/dev/null || kill -9 $pid 2>/dev/null
        fi
    }
    
    # Kill all processes by PID and their process groups
    kill_process_tree "$ROBOT_PID"
    kill_process_tree "$TELEOP_PID"
    
    # Kill any remaining child processes of this script
    pkill -TERM -P $$ 2>/dev/null
    sleep 0.5
    pkill -9 -P $$ 2>/dev/null
    
    # Also kill processes by name as a fallback
    pkill -TERM -f "robot_node_launcher.py.*robot=rowlet" 2>/dev/null
    pkill -TERM -f "rowlet-teleop.py" 2>/dev/null
    sleep 0.5
    pkill -9 -f "robot_node_launcher.py.*robot=rowlet" 2>/dev/null
    pkill -9 -f "rowlet-teleop.py" 2>/dev/null
    
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Launch robot node in the background
echo -e "${BLUE}Launching robot node for rowlet...${NC}"
python3 "$SCRIPT_DIR/ros/scripts/robot_node_launcher.py" robot=rowlet &
ROBOT_PID=$!

# Wait a moment for robot node to initialize
sleep 2

# Launch teleop node
echo -e "${BLUE}Launching teleop node for rowlet...${NC}"
python3 "$SCRIPT_DIR/ros/examples/rowlet-teleop.py" &
TELEOP_PID=$!

echo -e "${GREEN}Both nodes launched successfully!${NC}"
echo -e "Robot node PID: $ROBOT_PID"
echo -e "Teleop node PID: $TELEOP_PID"
echo -e "${GREEN}Press Ctrl+C to stop both nodes${NC}"

# Wait for both processes
wait $ROBOT_PID $TELEOP_PID

