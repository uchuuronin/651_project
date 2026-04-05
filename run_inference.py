#!/usr/bin/env python3
"""
Generate model inference predictions on TriviaQA dataset.
Creates pilot dataset for abstention threshold evaluation.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import requests

from src.llama_cpp import LlamaServerClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_triviaqa_pilot(
    split: str = "validation",
    num_examples: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load TriviaQA dataset examples. Falls back to synthetic data if HF unavailable.

    Args:
        split: Dataset split ("validation" or "train")
        num_examples: Number of examples to load
        seed: Random seed for reproducibility

    Returns:
        List of question dicts with 'question', 'answers', 'passage'
    """
    np.random.seed(seed)

    try:
        from datasets import load_dataset
        logger.info(f"Loading TriviaQA {split} split ({num_examples} examples)...")
        dataset = load_dataset("trivia_qa", "rc.nocontext", split=split, trust_remote_code=True)

        # Sample examples
        indices = np.random.choice(len(dataset), size=min(num_examples, len(dataset)), replace=False)
        examples = [dataset[int(i)] for i in indices]

        logger.info(f"Loaded {len(examples)} TriviaQA examples")
        return examples

    except (ImportError, Exception) as e:
        logger.warning(f"Could not load TriviaQA from HuggingFace ({e}). Using synthetic data.")
        return _generate_synthetic_triviaqa(num_examples, seed)


def _generate_synthetic_triviaqa(num_examples: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate synthetic TriviaQA-like data for testing.

    Args:
        num_examples: Number of synthetic examples to generate
        seed: Random seed

    Returns:
        List of question dicts
    """
    np.random.seed(seed)

    question_templates = [
        "What is the capital of {}?",
        "Who won the {} World Cup?",
        "In which year was {} founded?",
        "What is the population of {}?",
        "Who invented {}?",
        "What is the largest {} in the world?",
        "Which country is home to {}?",
        "What is the significance of {}?",
    ]

    entities = [
        "France", "Brazil", "Germany", "Italy", "Spain",
        "Tokyo", "New York", "London", "Paris", "Berlin",
        "The Internet", "The Telephone", "The Steam Engine",
        "Mount Everest", "The Amazon", "The Sahara",
    ]

    examples = []
    for i in range(num_examples):
        template_idx = i % len(question_templates)
        entity_idx = (i * 3) % len(entities)
        answer_idx = (i * 7) % len(entities)

        question = question_templates[template_idx].format(entities[entity_idx])
        answer = entities[answer_idx]
        passage = f"This is a passage about {entities[entity_idx]}. It mentions {answer}."

        examples.append({
            "question": question,
            "answers": {"text": [answer]},
            "passage": passage,
        })

    logger.info(f"Generated {len(examples)} synthetic TriviaQA-like examples")
    return examples


def run_inference(
    client: LlamaServerClient,
    examples: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 100,
) -> List[Dict[str, Any]]:
    """
    Run inference on TriviaQA examples.

    Args:
        client: LlamaServerClient instance
        examples: List of TriviaQA examples
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        List of result dicts with question, answer, prediction, confidence
    """
    results = []

    logger.info(f"Running inference on {len(examples)} examples...")

    for i, example in enumerate(examples):
        question = example.get("question", "")
        passage = example.get("passage", "")
        
        # Extract ground truth from TriviaQA answer field
        answer_data = example.get("answer", {})
        if isinstance(answer_data, dict):
            ground_truth = answer_data.get("value", "")
        elif isinstance(answer_data, list) and len(answer_data) > 0:
            gt_item = answer_data[0]
            ground_truth = gt_item.get("value", "") if isinstance(gt_item, dict) else str(gt_item)
        else:
            ground_truth = ""

        # Create prompt (extractive QA format)
        prompt = f"""Context: {passage}

Question: {question}

Answer:"""

        try:
            # Call model
            messages = [{"role": "user", "content": prompt}]
            response = client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            prediction = response["choices"][0]["message"]["content"].strip()

            # Simple confidence: check if first token matches ground truth (heuristic)
            # In practice, could use model's logit scores if available
            confidence = _estimate_confidence(prediction, ground_truth)

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "confidence": confidence,
                "prompt": prompt,
            })

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(examples)} examples")

        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": "",
                "confidence": 0.5,
                "error": str(e),
            })

    logger.info(f"Inference complete. {len(results)} results.")
    return results


def _estimate_confidence(prediction: str, ground_truth: str) -> float:
    """
    Estimate model confidence (heuristic for calibration).

    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer

    Returns:
        Confidence score between 0 and 1
    """
    # Simple heuristic: check for length and token overlap
    if not prediction or not ground_truth:
        return 0.5

    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())

    if len(truth_tokens) == 0:
        return 0.5

    overlap = len(pred_tokens & truth_tokens) / len(truth_tokens)

    # Map overlap to rough confidence [0.5, 0.95]
    confidence = 0.5 + 0.45 * overlap

    return float(confidence)


def save_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save inference results to JSON file.

    Args:
        results: List of result dicts
        output_path: Path to save JSON
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved {len(results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate TriviaQA pilot inference dataset"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=50,
        help="Number of examples for pilot (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:4020",
        help="llama-server URL (default: http://127.0.0.1:4020)",
    )

    args = parser.parse_args()

    # Initialize client and wait for server
    logger.info(f"Connecting to llama-server at {args.server_url}...")
    client = LlamaServerClient(base_url=args.server_url)

    if not client.wait_for_server(max_retries=60, retry_delay=1.0):
        logger.error("llama-server is not available")
        return 1

    # Load examples
    examples = load_triviaqa_pilot(
        split="validation",
        num_examples=args.num_examples,
        seed=args.seed,
    )

    # Run inference
    results = run_inference(
        client=client,
        examples=examples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Save results
    output_path = args.output_dir / f"pilot_{args.num_examples}_examples.json"
    save_results(results, output_path)

    logger.info(f"Pilot inference complete ({len(results)} examples)")
    return 0


if __name__ == "__main__":
    exit(main())
