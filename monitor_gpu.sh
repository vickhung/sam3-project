#!/bin/bash
# Monitor GPU usage while running the camera script

echo "Starting GPU monitoring..."
echo "Run the camera script in another terminal to see GPU usage"
echo "Press Ctrl+C to stop monitoring"
echo ""

watch -n 1 nvidia-smi

