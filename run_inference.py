# Runs inference via llama-server and saves results CSV.
# Start server before running:
#   llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF --host 127.0.0.1 --port 4020

import json
import argparse
import pandas as pd
from src.config import DATA_DIR, RESULTS_DIR, LOGGER
from src.llama_cpp import LlamaCppPipeline

DEFAULT_LIM = 50
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4020
CSV_COLUMNS = ["question", "answer_pred", "correct","token_prob_first", "token_prob_mean", "verbalized_conf"]
DATASET_FILES = {
    "triviaqa":"triviaqa_full.csv",
    "popqa":"popqa_full.csv",
}


def load_data(dataset: str, n: int) -> pd.DataFrame:
    path = DATA_DIR / dataset / DATASET_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Fetch {dataset}.")
    df = pd.read_csv(path).head(n)
    LOGGER.info(f"Loaded {len(df)} examples from {path}")
    return df


def is_correct(predicted: str, aliases_json: str) -> bool:
    # Alias matching (not plain exact match)
    if not predicted:
        return False
    aliases = json.loads(aliases_json)
    pred = predicted.lower().strip()
    return any(pred == a for a in aliases)


def run_inference(dataset: str, lim: int, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    out_path = RESULTS_DIR / f"inference_{dataset}_{lim}.csv"
    if out_path.exists():
        LOGGER.info(f"Results already exist at {out_path}. Skipping inference.")
        return

    df =load_data(dataset, lim)

    pipeline = LlamaCppPipeline(host=host, port=port)
    pipeline.load_model()

    questions = df["question"].tolist()
    results =pipeline.run(questions)

    # evaluate correctness against aliases
    for result, (_, row) in zip(results,df.iterrows()):
        result["correct"] =is_correct(result["answer_pred"], row["aliases"])

    results_df = pd.DataFrame(results, columns=CSV_COLUMNS)
    results_df.to_csv(out_path, index=False)
    LOGGER.info(f"Saved {len(results_df)} rows to {out_path}")

    accuracy = results_df["correct"].mean()
    conf_null = results_df["verbalized_conf"].isna().mean()
    LOGGER.info(f"Accuracy: {accuracy:.3f}")
    LOGGER.info(f"Verbalized conf missing: {conf_null:.1%}; {'Signal unusable' if conf_null > 0.5 else 'OK'}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run inference for abstention experiments.")
    parser.add_argument("--lim", type=int, default=DEFAULT_LIM,help=f"Number of examples (default: {DEFAULT_LIM})")
    parser.add_argument("--dataset", choices=["triviaqa", "popqa"], default="triviaqa")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    run_inference(dataset=args.dataset, lim=args.lim, host=args.host, port=args.port)


if __name__ == "__main__":
    main()