#!/usr/bin/env python3
"""
Monitor progress of background full inference.
Checks checkpoint file and log output to estimate completion time.
"""

import json
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime


def check_progress():
    """Check progress of full 500-example inference."""
    checkpoint_file = Path("results/checkpoint_inference.json")
    log_file = Path("results/inference.log")
    output_file = Path("results/inference_triviaqa_1000_1.5b.csv")

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
        df = pd.read_csv(output_file)
        print(f"COMPLETE: {len(df)} examples finished!")
        return len(results)
    
    return 0


if __name__ == "__main__":
    examples = check_progress()
