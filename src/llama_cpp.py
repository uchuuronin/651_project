# Inference backend for models running via llama-server (llama.cpp).
#
# Start server before running:
#   Qwen:  llama-server -hf Qwen/Qwen2.5-7B-Instruct-GGUF --host 127.0.0.1 --port 4020
#   Llama: llama-server -hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF --host 127.0.0.1 --port 4020

import re
import requests
from src.config import LOGGER

_STOP_TOKENS = ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "\nQuestion:", "\n\n"]

class LlamaCppPipeline:
    def __init__(self, host: str = "127.0.0.1", port: int = 4020):
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
                f"Start with: llama-server -hf <model> --host 127.0.0.1 --port 4020"
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
                f"{self.base_url}/completion",
                json={
                    "prompt": prompt,
                    "temperature": 0.0,
                    "n_predict": 60,
                    "n_probs": 1,
                    "stream": False,
                    "stop": _STOP_TOKENS,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("content", "").strip()
        except Exception as e:
            LOGGER.warning(f"Completion failed: {e}")
            return self._empty_result(question)

        answer, verbalized_conf = self._parse_response(raw)
        token_prob_first, token_prob_mean = self._extract_token_probs(data)

        return {
            "question": question,
            "answer_pred": answer,
            "correct": None,
            "token_prob_first": token_prob_first,
            "token_prob_mean": token_prob_mean,
            "verbalized_conf": verbalized_conf,
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
    
    def _extract_token_probs(self, data: dict) -> tuple[float | None, float | None]:
        # Reads completion_probabilities from /completion response.
        # /v1/completions does not return logprobs reliably 
        import math
        probs = data.get("completion_probabilities", [])
        if not probs:
            return None, None
        try:
            log_probs = [e["logprob"] for e in probs if "logprob" in e]
            if not log_probs:
                return None, None
            return math.exp(log_probs[0]), math.exp(sum(log_probs) / len(log_probs))
        except Exception as e:
            LOGGER.warning(f"Token prob extraction failed: {e}")
            return None, None    

    def _empty_result(self, question: str) -> dict:
        return {
        "question":question,
        "answer_pred":None,
        "correct":None,
        "token_prob_first":None,
        "token_prob_mean":None,
        "verbalized_conf":None,
        }