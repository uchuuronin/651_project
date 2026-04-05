"""
Abstention Threshold Optimization Package.

Core modules:
- thresholds: Optimal abstention threshold functions
- llama_cpp: LLP inference API client
- sweep: Optimization algorithm for abstention strategies
- config: Experimental configuration
"""

from src.config import LOSS_FUNCTIONS
from src.llama_cpp import LlamaServerClient
from src.sweep import AbstractionSweep
from src.thresholds import tau_B, tau_CE, tau_U

__all__ = [
    "tau_U",
    "tau_B",
    "tau_CE",
    "LlamaServerClient",
    "AbstractionSweep",
    "LOSS_FUNCTIONS",
]
