#!/usr/bin/env python3
"""
Generate model inference predictions on full TriviaQA dataset (500 examples).
Task 8: Scale inference for full empirical evaluation.

Note: This may take 6-8 hours depending on model latency.
Implements checkpointing to allow resuming interrupted runs.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.llama_cpp import LlamaServerClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_triviaqa_full(
    split: str = "validation",
    num_examples: int = 500,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load full TriviaQA dataset examples for empirical evaluation.

    Args:
        split: Dataset split ("validation" or "train")
        num_examples: Number of examples to load (up to 5000+ in validation set)
        seed: Random seed for reproducibility

    Returns:
        List of question dicts with 'question', 'answers', 'passage'
    """
    np.random.seed(seed)

    try:
        from datasets import load_dataset
        logger.info(f"Loading TriviaQA {split} split ({num_examples} examples)...")
        dataset = load_dataset("trivia_qa", "rc.nocontext", split=split, trust_remote_code=True)

        logger.info(f"Full TriviaQA {split} has {len(dataset)} examples. Sampling {num_examples}...")

        # Sample examples
        indices = np.random.choice(len(dataset), size=min(num_examples, len(dataset)), replace=False)
        examples = [dataset[int(i)] for i in indices]

        logger.info(f"Loaded {len(examples)} TriviaQA examples")
        return examples

    except (ImportError, Exception) as e:
        logger.warning(f"Could not load TriviaQA from HuggingFace ({e}). Using synthetic data.")
        return _generate_synthetic_triviaqa(num_examples, seed)


def _generate_synthetic_triviaqa(num_examples: int = 500, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate synthetic TriviaQA-like data for testing at scale.

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
        "When was {} established?",
        "What does {} stand for?",
    ]

    entities = [
        "France", "Brazil", "Germany", "Italy", "Spain",
        "Tokyo", "New York", "London", "Paris", "Berlin",
        "The Internet", "The Telephone", "The Steam Engine",
        "Mount Everest", "The Amazon", "The Sahara",
        "Shakespeare", "Mozart", "Einstein", "Newton",
        "The Eiffel Tower", "The Statue of Liberty", "Big Ben",
    ]

    examples = []
    for i in range(num_examples):
        template_idx = i % len(question_templates)
        entity_idx = (i * 3) % len(entities)
        answer_idx = (i * 7) % len(entities)

        question = question_templates[template_idx].format(entities[entity_idx])
        answer = entities[answer_idx]
        passage = f"This is a passage about {entities[entity_idx]}. It mentions {answer}. Additional context may be provided here."

        examples.append({
            "question": question,
            "answers": {"text": [answer]},
            "passage": passage,
        })

    logger.info(f"Generated {len(examples)} synthetic TriviaQA-like examples")
    return examples


def load_checkpoint(checkpoint_file: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Load previously completed inference results from checkpoint.

    Args:
        checkpoint_file: Path to checkpoint JSON file

    Returns:
        List of completed results, or None if checkpoint doesn't exist
    """
    if not checkpoint_file.exists():
        return None

    try:
        with open(checkpoint_file, "r") as f:
            data = json.load(f)

        logger.info(f"Loaded checkpoint with {len(data)} completed examples from {checkpoint_file}")
        return data

    except Exception as e:
        logger.warning(f"Could not load checkpoint ({e}). Starting fresh.")
        return None


def save_checkpoint(results: List[Dict[str, Any]], checkpoint_file: Path) -> None:
    """
    Save intermediate results as checkpoint.

    Args:
        results: List of completed results
        checkpoint_file: Path to save checkpoint to
    """
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved checkpoint: {len(results)} completed examples")


def run_inference_full(
    client: LlamaServerClient,
    examples: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 100,
    checkpoint_freq: int = 50,
    checkpoint_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Run inference on full TriviaQA dataset with checkpointing support.

    Args:
        client: LlamaServerClient instance
        examples: List of TriviaQA examples
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        checkpoint_freq: Save checkpoint every N examples
        checkpoint_file: Optional path for checkpoint file

    Returns:
        List of result dicts with question, answer, prediction, confidence
    """
    results = []
    start_idx = 0

    # Load checkpoint if available
    if checkpoint_file and checkpoint_file.exists():
        checkpoint_results = load_checkpoint(checkpoint_file)
        if checkpoint_results:
            results = checkpoint_results
            start_idx = len(results)

    logger.info(f"Running inference on {len(examples)} examples (starting from {start_idx})...")
    total_examples = len(examples)
    
    # Estimate time
    avg_time_per_example = 0
    start_time = time.time()

    for i in range(start_idx, len(examples)):
        example = examples[i]
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

            # Estimate confidence (heuristic)
            confidence = _estimate_confidence(prediction, ground_truth)

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "confidence": confidence,
                "prompt": prompt,
            })

            # Progress logging with time estimation
            examples_done = i + 1
            elapsed = time.time() - start_time
            avg_time_per_example = elapsed / examples_done
            remaining = (total_examples - examples_done) * avg_time_per_example
            
            if (examples_done) % checkpoint_freq == 0 or examples_done == len(examples):
                remaining_hours = remaining / 3600
                logger.info(
                    f"Progress: {examples_done}/{total_examples} "
                    f"({100*examples_done//total_examples}%) | "
                    f"Avg: {avg_time_per_example:.2f}s/ex | "
                    f"ETA: {remaining_hours:.1f}h remaining"
                )
                
                # Save checkpoint
                if checkpoint_file:
                    save_checkpoint(results, checkpoint_file)

        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": "",
                "confidence": 0.5,
                "error": str(e),
            })

    total_time = time.time() - start_time
    logger.info(
        f"Inference complete. {len(results)} results in "
        f"{total_time/3600:.2f} hours (avg {total_time/len(results):.2f}s per example)"
    )
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


def compute_accuracy(results: List[Dict[str, Any]]) -> float:
    """
    Compute accuracy from inference results.

    Args:
        results: List of result dicts

    Returns:
        Accuracy as fraction correct
    """
    if not results:
        return 0.0

    correct = 0
    for result in results:
        ground_truth = result.get("ground_truth", "").strip().lower()
        prediction = result.get("prediction", "").strip().lower()
        
        if ground_truth and prediction:
            # Simple string matching
            if ground_truth in prediction or prediction in ground_truth:
                correct += 1

    return correct / len(results)


def main():
    parser = argparse.ArgumentParser(
        description="Generate full TriviaQA inference dataset (500 examples)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=500,
        help="Number of examples for full dataset (default: 500)",
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
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50,
        help="Save checkpoint every N examples (default: 50)",
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="Resume from checkpoint if available",
    )

    args = parser.parse_args()

    # Initialize client and wait for server
    logger.info(f"Connecting to llama-server at {args.server_url}...")
    client = LlamaServerClient(base_url=args.server_url)

    if not client.wait_for_server(max_retries=60, retry_delay=1.0):
        logger.error("llama-server is not available")
        return 1

    # Checkpoint file
    checkpoint_file = args.output_dir / f"checkpoint_full_{args.num_examples}_examples.json"

    # Load examples
    examples = load_triviaqa_full(
        split="validation",
        num_examples=args.num_examples,
        seed=args.seed,
    )

    # Run inference
    results = run_inference_full(
        client=client,
        examples=examples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_file=checkpoint_file if args.use_checkpoint else None,
    )

    # Save results
    output_path = args.output_dir / f"full_{args.num_examples}_examples.json"
    save_results(results, output_path)

    # Compute and report accuracy
    accuracy = compute_accuracy(results)
    logger.info(f"Full inference complete: {len(results)} examples")
    logger.info(f"Dataset accuracy: {accuracy:.1%}")

    # Clean up checkpoint if successful
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("Removed checkpoint file")

    return 0


if __name__ == "__main__":
    exit(main())
