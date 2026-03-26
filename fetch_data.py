# Downloads full TriviaQA validation split and saves to disk.

import re
import json
import string
import pandas as pd
from datasets import load_dataset
from src.config import TRIVIAQA_DIR, LOGGER


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


def main():
    data_path = TRIVIAQA_DIR / "triviaqa_full.csv"

    if data_path.exists():
        LOGGER.info(f"Data already exists at {data_path}, skipping download.")
        return

    LOGGER.info("Loading TriviaQA (rc, validation split) from HuggingFace...")
    dataset =load_dataset("trivia_qa", "rc", split="validation")
    LOGGER.info(f"Downloaded {len(dataset)} examples.")

    rows = []
    for example in dataset:
        aliases = extract_aliases(example["answer"])
        rows.append({
            "question_id":example["question_id"],
            "question":example["question"],
            "answer":normalize_answer(example["answer"]["value"]),
            "aliases":json.dumps(aliases),
        })

    df = pd.DataFrame(rows)
    df.to_csv(data_path, index=False)
    LOGGER.info(f"Saved {len(df)} examples to {data_path}")

    check = pd.read_csv(data_path)
    LOGGER.info(f"Verified: {len(check)} rows, columns: {list(check.columns)}")
    LOGGER.info(f"Sample Q: {check['question'].iloc[0]}")
    LOGGER.info(f"Sample A: {check['answer'].iloc[0]}")


if __name__ == "__main__":
    main()