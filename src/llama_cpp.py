# Inference backend for models running via llama-server (llama.cpp).
#
# Start server before running:
#  llama-server -hf Qwen/Qwen3-8B-GGUF --host 127.0.0.1 --port 4568

LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 4020

import re
import requests
from src.config import LOGGER


class LlamaCppPipeline:
    def __init__(self, host: str = LLAMA_HOST, port: int = LLAMA_PORT):
        self.base_url = f"http://{host}:{port}"
        self._ready = False

    def load_model(self):
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
            self._ready = True
            LOGGER.info(f"llama-server reachable at {self.base_url}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"llama-server not reachable at {self.base_url}. "
                f"Start with: llama-server -hf <model> --host {LLAMA_HOST} --port {LLAMA_PORT}"
            )

    def run(self, questions: list[str], n: int | None = None) -> list[dict]:
        """
        Runs inference on questions via llama-server HTTP API.
        """
        if not self._ready:
            raise RuntimeError("Call load_model() before run().")

        if n is not None:
            questions = questions[:n]

        results = []
        for i, question in enumerate(questions):
            LOGGER.info(f"[{i+1}/{len(questions)}] {question[:60]}...")
            results.append(self._run_single(question))

        return results

    def _run_single(self, question: str) -> dict:
        # Prompt for verbalized confidence elicitation (Wang et al.) to append "Confidence: X%" after its answer.
        prompt = (
            f"Answer the following question in one short phrase. "
            f"After your answer, on the same line write 'Confidence: X%' "
            f"where X is your confidence from 0 to 100.\n\n"
            f"Question: {question}\nAnswer:"
        )

        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 60,
                },
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            LOGGER.warning(f"Chat completion failed: {e}")
            return self._empty_result(question)

        answer, verbalized_conf = self._parse_response(raw)
        token_prob_first, token_prob_mean = self._get_token_probs(answer)

        return {
            "question": question,
            "answer": answer,
            "correct": None, #evaluated in run_inference.py against aliases
            "token_prob_first":token_prob_first,
            "token_prob_mean":token_prob_mean,
            "verbalized_conf":verbalized_conf,
        }

    def _parse_response(self, raw: str) -> tuple[str, float | None]:
        # Splits model output into answer and verbalized confidence.
        conf = None
        answer = raw

        match = re.search(r"[Cc]onfidence:\s*(\d+(?:\.\d+)?)\s*%", raw)
        if match:
            try:
                conf = max(0.0, min(1.0, float(match.group(1)) / 100.0))
            except ValueError:
                conf = None
            answer = raw[:match.start()].strip().rstrip(".")

        return answer.strip(), conf

    def _get_token_probs(self, answer: str) -> tuple[float | None, float | None]:
        # Gets log probabilities for answer tokens via /v1/completions.
        if not answer:
            return None, None

        try:
            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "prompt": answer,
                    "max_tokens": 1,
                    "logprobs": True,
                    "echo": True,
                },
                timeout=15,
            )
            resp.raise_for_status()
            token_logprobs = resp.json()["choices"][0]["logprobs"]["token_logprobs"]
            valid = [lp for lp in token_logprobs if lp is not None]
            if not valid:
                return None, None
            return float(valid[0]), float(sum(valid) / len(valid))
        except Exception as e:
            LOGGER.warning(f"Token prob extraction failed: {e}")
            return None, None

    def _empty_result(self, question: str) -> dict:
        return {
            "question": question,
            "answer":None,
            "correct":None,
            "token_prob_first":None,
            "token_prob_mean":None,
            "verbalized_conf":None,
        }