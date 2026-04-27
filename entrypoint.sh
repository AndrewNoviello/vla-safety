#!/bin/bash
set -e

# Hand off to the command — use -i if it's bash so ~/.bashrc is sourced
if [ "$1" = "bash" ]; then
    exec bash -i
else
    exec "$@"
fi

