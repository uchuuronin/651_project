import sys
import os
from fetch_data import normalize_answer, extract_aliases

class TestNormalizeAnswer:
    def test(self):
        assert normalize_answer("David Seville") == "david seville"
        assert normalize_answer("The Chipmunks") == "chipmunks"
        assert normalize_answer("A Rolling. Stone") == "rolling stone"
        assert normalize_answer("An   Apple") == "apple"
        assert normalize_answer("") == ""

class TestExtractAliases:
    def test_deduplicates_after_normalization(self):
        answer_dict = {
            "aliases": ["The Rolling Stones", "Rolling Stones"],
            "normalized_aliases": ["rolling stones"],
        }
        assert extract_aliases(answer_dict) == ["rolling stones"]

    def test_filters_empty_strings(self):
        answer_dict = {"aliases": ["", "Paris"], "normalized_aliases": []}
        result = extract_aliases(answer_dict)
        assert "" not in result
        assert "paris" in result

    def test_missing_key_handled(self):
        result = extract_aliases({"aliases": ["Only Aliases"]})
        assert "only aliases" in result
        
# python -m pytest tests/test_dataset.py -v
