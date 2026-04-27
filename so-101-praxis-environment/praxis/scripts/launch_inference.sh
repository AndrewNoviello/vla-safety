#!/bin/bash

# Launch script for Bimanual teleoperation system (Piplup + Rowlet)
# This script launches robot nodes, sensor nodes (DIGIT L/R), and teleop nodes for both robots

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Bimanual teleoperation system (Piplup + Rowlet)...${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${RED}Shutting down all nodes...${NC}"
    
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
    kill_process_tree "$PIPLUP_ROBOT_PID"
    kill_process_tree "$ROWLET_ROBOT_PID"
    kill_process_tree "$SENSORR_SENSOR_PID"
    kill_process_tree "$SENSORL_SENSOR_PID"
    kill_process_tree "$PIPLUP_TELEOP_PID"
    kill_process_tree "$ROWLET_TELEOP_PID"
    
    # Kill any remaining child processes of this script
    pkill -TERM -P $$ 2>/dev/null
    sleep 0.5
    pkill -9 -P $$ 2>/dev/null
    
    # Also kill processes by name as a fallback (for setsid processes that became orphaned)
    pkill -TERM -f "robot_node_launcher.py.*robot=" 2>/dev/null
    pkill -TERM -f "sensor_node_launcher.py.*sensor=" 2>/dev/null
    pkill -TERM -f "piplup-teleop.py" 2>/dev/null
    pkill -TERM -f "rowlet-teleop.py" 2>/dev/null
    sleep 0.5
    pkill -9 -f "robot_node_launcher.py.*robot=" 2>/dev/null
    pkill -9 -f "sensor_node_launcher.py.*sensor=" 2>/dev/null
    pkill -9 -f "piplup-teleop.py" 2>/dev/null
    pkill -9 -f "rowlet-teleop.py" 2>/dev/null
    
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Launch Piplup robot node in the background
echo -e "${BLUE}Launching robot node for Piplup...${NC}"
python3 "$SCRIPT_DIR/../ros/scripts/robot_node_launcher.py" robot=piplup &
PIPLUP_ROBOT_PID=$!

# Wait a moment for first robot node to initialize
sleep 2

# Launch Rowlet robot node in the background
echo -e "${BLUE}Launching robot node for Rowlet...${NC}"
python3 "$SCRIPT_DIR/../ros/scripts/robot_node_launcher.py" robot=rowlet &
ROWLET_ROBOT_PID=$!

# Wait a moment for second robot node to initialize
sleep 2

# Launch DIGIT sensor nodes
echo -e "${BLUE}Launching sensor node for DIGIT Right...${NC}"
setsid python3 "$SCRIPT_DIR/../ros/scripts/sensor_node_launcher.py" sensor=digitR > /dev/null 2>&1 &
SENSORR_SENSOR_PID=$!

sleep 1

echo -e "${BLUE}Launching sensor node for DIGIT Left...${NC}"
setsid python3 "$SCRIPT_DIR/../ros/scripts/sensor_node_launcher.py" sensor=gsminiL > /dev/null 2>&1 &
SENSORL_SENSOR_PID=$!

echo -e "${GREEN}All nodes launched successfully!${NC}"
echo -e "${YELLOW}Piplup Robot node PID: $PIPLUP_ROBOT_PID${NC}"
echo -e "${YELLOW}Rowlet Robot node PID: $ROWLET_ROBOT_PID${NC}"
echo -e "${YELLOW}DIGIT Right Sensor node PID: $SENSORR_SENSOR_PID${NC}"
echo -e "${YELLOW}DIGIT Left Sensor node PID: $SENSORL_SENSOR_PID${NC}"
echo -e "${GREEN}Press Ctrl+C to stop all nodes${NC}"

# Wait for all processes
wait $PIPLUP_ROBOT_PID $ROWLET_ROBOT_PID $SENSORR_SENSOR_PID $SENSORL_SENSOR_PID

