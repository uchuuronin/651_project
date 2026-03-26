import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_data import normalize_answer
from src.config import DATA_DIR, RESULTS_DIR, LOGGER
from src.llama_cpp import LlamaCppPipeline


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4020
DEFAULT_LIM = 50


CSV_COLUMNS = [
    "question",
    "answer_pred",
    "verbalized_conf",
    "token_prob_first",
    "token_prob_mean",
    "correct",
]

DATASET_FILES = {
    "triviaqa": "triviaqa_full.csv",
    "popqa": "popqa_full.csv",
}


def set_seed(seed: int) -> None:
    # Inference is mostly deterministic when temperature=0.0, but set seeds for defensiveness.
    random.seed(seed)
    np.random.seed(seed)


def is_correct(predicted: str, aliases_json: str) -> bool:
    if not predicted:
        return False
    try:
        aliases = json.loads(aliases_json)
    except Exception:
        return False

    pred_norm = normalize_answer(predicted)
    return any(pred_norm == a or a in pred_norm or pred_norm in a for a in aliases)


def load_dataset_csv(dataset: str, lim: int) -> pd.DataFrame:
    data_path = DATA_DIR / dataset / DATASET_FILES[dataset]
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run `python fetch_data.py --dataset {dataset}` first."
        )
    df = pd.read_csv(data_path).head(lim)
    LOGGER.info(f"Loaded {len(df)} examples from {data_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for abstention experiments.")
    parser.add_argument("--lim", type=int, default=DEFAULT_LIM, help=f"Number of examples (default: {DEFAULT_LIM})")
    parser.add_argument("--dataset", choices=["triviaqa", "popqa"], default="triviaqa")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = RESULTS_DIR / f"inference_{args.dataset}_{args.lim}.csv"
    if out_path.exists():
        LOGGER.info(f"Results already exist at {out_path}, skipping inference.")
        return

    df = load_dataset_csv(args.dataset, args.lim)

    pipeline = LlamaCppPipeline(host=args.host, port=args.port)
    pipeline.load_model()

    questions = df["question"].tolist()
    results = pipeline.run(questions)

    # evaluate correctness against dataset aliases
    for result, (_, row) in zip(results, df.iterrows()):
        result["correct"] = is_correct(result["answer_pred"], row["aliases"])

    results_df = pd.DataFrame(results)

    # Enforce schema contract for downstream sweep/plots.
    missing = [c for c in CSV_COLUMNS if c not in results_df.columns]
    if missing:
        raise RuntimeError(f"Inference output missing required columns: {missing}")
    results_df = results_df[CSV_COLUMNS]

    results_df.to_csv(out_path, index=False)
    LOGGER.info(f"Saved {len(results_df)} rows to {out_path}")
    LOGGER.info(f"Accuracy: {float(results_df['correct'].mean()):.3f}")
    LOGGER.info(f"verbalized_conf missing: {float(results_df['verbalized_conf'].isna().mean()):.1%}")


if __name__ == "__main__":
    main()

