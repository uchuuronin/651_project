# Central config for paths, model settings, and experiment parameters.

from pathlib import Path
import logging
from src.thresholds import tau_U, tau_B, tau_CE 

SRC_DIR= Path(__file__).resolve().parent
ROOT_DIR= SRC_DIR.parent

DATA_DIR = ROOT_DIR/"data"
RESULTS_DIR = ROOT_DIR/"results"   
LOG_DIR = ROOT_DIR/"logs"

def csv_path(model_tag: str, dataset: str) -> Path:
    return RESULTS_DIR / f"inference_{model_tag}_{dataset}.csv"

def fig_path(name: str) -> Path:
    return RESULTS_DIR / f"fig_{name}.pdf"

def setup_dirs():
    for d in [DATA_DIR, RESULTS_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        
N_EXAMPLES= 500
N_PILOT= 50

LAMBDA_GRID = [0.05, 0.1, 0.15, 0.2, 0.25]
TAU_GRID= [i / 100 for i in range(1, 100)]

LOGGER = logging.getLogger("cs651")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    LOGGER.addHandler(handler)

LOSS_FUNCTIONS = {
    "utility": tau_U,
    "brier": tau_B,
    "cross-entropy": tau_CE,
}