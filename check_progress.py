#!/usr/bin/env python3
"""
Monitor progress of background full inference (Task 8).
Checks checkpoint file and log output to estimate completion time.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def check_progress():
    """Check progress of full 500-example inference."""
    checkpoint_file = Path("results/checkpoint_full_500_examples.json")
    log_file = Path("logs/full_inference_500.log")
    output_file = Path("results/full_500_examples.json")
    
    if not log_file.exists():
        print("Log file not found. Inference may not have started yet.")
        return
    
    # Read log file
    with open(log_file, "r") as f:
        lines = f.readlines()
    
    # Try to find progress information
    progress_lines = [l for l in lines if "Progress:" in l]
    
    if progress_lines:
        latest = progress_lines[-1].strip()
        print(f"Latest progress: {latest}")
    else:
        print("Inference in progress (loading data stage)")
    
    # Check if checkpoint exists
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
        print(f"Checkpoint: {len(checkpoint)} examples completed")
        return len(checkpoint)
    
    # Check if finished
    if output_file.exists():
        with open(output_file, "r") as f:
            results = json.load(f)
        print(f"COMPLETE: {len(results)} examples finished!")
        return len(results)
    
    return 0


if __name__ == "__main__":
    examples = check_progress()
