#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Filtering all raw psychopy inputs..."
python -u "$SCRIPT_DIR/prescan.py"

echo "Preprocess the behavior csv files"
python -u "$SCRIPT_DIR/preprocess_behavior.py"

echo "Preprocess the mouse csv files"
python -u "$SCRIPT_DIR/preprocess_mouse.py"

echo "Concatenate all the preprocessed files"
python -u "$SCRIPT_DIR/batch_collect.py"
