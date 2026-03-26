# Downloads full dataset validation split and saves to disk.

import re
import json
import string
import argparse
import pandas as pd
from datasets import load_dataset
from src.config import DATA_DIR, LOGGER


def normalize_answer(text: str) -> str:
    """
    lowercase, strip punctuation, remove articles, collapse whitespace.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def extract_aliases(answer_dict: dict) -> list[str]:
    """
    Combines answer.aliases and answer.normalized_aliases, normalizes all, and deduplicates.
    """
    raw = answer_dict.get("aliases", []) + answer_dict.get("normalized_aliases", [])
    return list({normalize_answer(a) for a in raw if a})


DATASET_CONFIGS = {
    "triviaqa": {
        "hf_name":"trivia_qa",
        "hf_config": "rc",
        "hf_split":"validation",
        "filename":"triviaqa_full.csv",
    },
    "popqa": {
        "hf_name":"akariasai/PopQA",
        "hf_config": None,
        "hf_split": "test",
        "filename":"popqa_full.csv",
    },
}


def fetch(dataset: str):
    cfg = DATASET_CONFIGS[dataset]
    data_dir = DATA_DIR/dataset 
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir/cfg["filename"]

    if data_path.exists():
        LOGGER.info(f"Data already exists at {data_path}, skipping download.")
        return

    LOGGER.info(f"Loading {dataset} from HuggingFace...")
    dataset_obj = load_dataset(
        cfg["hf_name"],
        cfg["hf_config"],
        split=cfg["hf_split"],
    )
    LOGGER.info(f"Downloaded {len(dataset_obj)} examples.")

    rows = []
    for example in dataset_obj:
        if dataset == "triviaqa":
            aliases = extract_aliases(example["answer"])
            rows.append({
                "question_id": example["question_id"],
                "question": example["question"],
                "answer":normalize_answer(example["answer"]["value"]),
                "aliases":json.dumps(aliases),
            })
        elif dataset == "popqa":
            aliases = [normalize_answer(a) for a in example.get("possible_answers", [])]
            rows.append({
                "question_id": example.get("id", ""),
                "question": example["question"],
                "answer":normalize_answer(example["answer"]),
                "aliases":json.dumps(list(set(aliases))),
            })
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    df = pd.DataFrame(rows)
    df.to_csv(data_path, index=False)
    LOGGER.info(f"Saved {len(df)} examples to {data_path}")

    check = pd.read_csv(data_path)
    LOGGER.info(f"Verified: {len(check)} rows, columns: {list(check.columns)}")
    LOGGER.info(f"Sample Q: {check['question'].iloc[0]}")
    LOGGER.info(f"Sample A: {check['answer'].iloc[0]}")


def main():
    parser = argparse.ArgumentParser(description="Download dataset for abstention experiments.")
    parser.add_argument(
        "--dataset",
        choices=["triviaqa", "popqa"],
        default="triviaqa",
        help="Dataset to download (default: triviaqa)",
    )
    args = parser.parse_args()
    fetch(args.dataset)

if __name__ == "__main__":
    main()

# Usage:
# python fetch_data.py --dataset triviaqa
# python fetch_data.py --dataset popqa