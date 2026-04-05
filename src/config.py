"""
Configuration for abstention sweep experiments.
Defines loss functions, model parameters, and experiment settings.
"""

from typing import Callable, Dict

from src.thresholds import tau_B, tau_CE, tau_U

# Loss function mappings
LOSS_FUNCTIONS: Dict[str, Callable] = {
    "utility": tau_U,
    "brier": tau_B,
    "cross-entropy": tau_CE,
}

# Default experimental parameters
DEFAULT_LAMBDA_RANGE = (0.0, 0.25)  # Loss weight range
DEFAULT_NUM_LAMBDAS = 51  # Number of lambda values to evaluate

# Model parameters
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_SERVER_URL = "http://127.0.0.1:4020"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 0.7

# Dataset parameters
DEFAULT_PILOT_SIZE = 50
DEFAULT_FULL_SIZE = 500
DEFAULT_SEED = 42

# Evaluation parameters
CONFIDENCE_BINS = 10
ECE_BINS = 10
