#!/bin/bash
# Helper script to run Jupyter Notebook with the SAM 3 environment

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the virtual environment
source "$DIR/venv/bin/activate"

# Change to the repository directory
cd "$DIR/sam3"

# Start Jupyter Notebook
echo "Starting Jupyter Notebook..."
echo "Note: You need to authenticate with Hugging Face to download checkpoints if you haven't already."
echo "Run 'hf auth login' if needed."
jupyter notebook examples/sam3_image_predictor_example.ipynb

