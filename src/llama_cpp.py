"""
Interface to llama.cpp server for inference.
Manages API calls to local llama-server instance.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class LlamaServerClient:
    """Client for interacting with llama-server OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://127.0.0.1:4020"):
        """
        Initialize the llama-server client.

        Args:
            base_url: URL to llama-server (default: localhost:4020)
        """
        self.base_url = base_url.rstrip('/')
        self.health_endpoint = f"{self.base_url}/health"
        self.completions_endpoint = f"{self.base_url}/v1/chat/completions"

    def health_check(self) -> bool:
        """
        Check if llama-server is running and healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(self.health_endpoint, timeout=2)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def wait_for_server(self, max_retries: int = 30, retry_delay: float = 1.0) -> bool:
        """
        Wait for llama-server to become available.

        Args:
            max_retries: Maximum number of retries
            retry_delay: Delay in seconds between retries

        Returns:
            True if server became available, False if timeout
        """
        for i in range(max_retries):
            if self.health_check():
                logger.info(f"Server is healthy (attempt {i+1}/{max_retries})")
                return True
            if i < max_retries - 1:
                time.sleep(retry_delay)
        logger.error(f"Server not healthy after {max_retries} retries")
        return False

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 200,
        top_p: float = 1.0,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Call the chat completion API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (default: 'default' for server's loaded model)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            timeout: Request timeout in seconds

        Returns:
            Response dict from llama-server API

        Raises:
            requests.RequestException if API call fails
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        response = requests.post(
            self.completions_endpoint,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        return response.json()

    def get_completion_text(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 200,
        **kwargs,
    ) -> str:
        """
        Get just the completion text from a chat completion call.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments to pass to chat_completion

        Returns:
            The assistant's response text

        Raises:
            requests.RequestException if API call fails
            KeyError if response format is unexpected
        """
        response = self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response["choices"][0]["message"]["content"]

    def batch_completions(
        self,
        message_batches: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 200,
        delay_between_requests: float = 0.1,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple chat completion requests in batch.

        Args:
            message_batches: List of message lists
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            delay_between_requests: Delay in seconds between requests
            verbose: Whether to log progress

        Returns:
            List of response dicts from llama-server API

        Raises:
            requests.RequestException if any API call fails
        """
        results = []
        for i, messages in enumerate(message_batches):
            if verbose and i % 10 == 0:
                logger.info(f"Processing batch {i}/{len(message_batches)}")

            response = self.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results.append(response)

            if i < len(message_batches) - 1:
                time.sleep(delay_between_requests)

        return results
