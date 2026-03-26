import pytest
import requests
import sys
import os
from unittest.mock import patch
import numpy as np
from src.llama_cpp import LlamaCppPipeline

@pytest.fixture
def pipeline():
    return LlamaCppPipeline(host="127.0.0.1", port=4020) 

class TestLoadModel:
    @patch("src.llama_cpp.requests.get")
    def test_success(self, mock_get, pipeline):
        mock_get.return_value.status_code = 200
        pipeline.load_model()
        assert pipeline._ready is True

    @patch("src.llama_cpp.requests.get")
    def test_server_not_running(self, mock_get, pipeline):
        mock_get.side_effect = requests.exceptions.ConnectionError
        with pytest.raises(RuntimeError):
            pipeline.load_model()

    def test_run_before_load_raises(self, pipeline):
        with pytest.raises(RuntimeError):
            pipeline.run(["What is the capital of France?"])

class TestParseResponse:
    def test(self, pipeline):
        answer, conf = pipeline._parse_response("Paris. Confidence: 92%")
        assert answer == "Paris"
        assert abs(conf - 0.92) < 1e-6

        _, conf = pipeline._parse_response("Paris")
        assert conf is None
        _, conf = pipeline._parse_response("Paris. Confidence: 150%")
        assert conf == 1.0

class TestGetTokenProbs:
    @patch("src.llama_cpp.requests.post")
    def test_success(self, mock_post, pipeline):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"logprobs": {"token_logprobs": [None, -0.5, -1.0]}}]
        }
        first, mean = pipeline._get_token_probs("Paris")
        assert abs(first - (np.exp(-0.5))) < 1e-6
        assert abs(mean - (np.exp(-0.75))) < 1e-6

    def test_empty_answer(self, pipeline):
        assert pipeline._get_token_probs("") == (None, None)

    @patch("src.llama_cpp.requests.post")
    def test_request_fails(self, mock_post, pipeline):
        mock_post.side_effect = requests.exceptions.ConnectionError
        assert pipeline._get_token_probs("Paris") == (None, None)
        
# python -m pytest tests/test_llama.py -v
        