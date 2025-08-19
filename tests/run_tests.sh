#!/bin/bash

export TEST_PATH="2025_08_16"

python3 conversation_collector.py
python3 response_quality.py
python3 time_breakdown.py
python3 quality_plots.py