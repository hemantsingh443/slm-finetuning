import unittest
from datasets import Dataset
from src.data.preprocess import preprocess_text, preprocess_dataset

class TestPreprocess(unittest.TestCase):
    def test_preprocess_text(self):
        config = {
            "lowercase": True,
            "normalize_whitespace": True,
            "strip_whitespace": True
        }
        raw_text = "  HELLO   WORLD\nNEW   LINE  "
        cleaned = preprocess_text(raw_text, config)
        self.assertEqual(cleaned, "hello world new line")

    def test_preprocess_dataset(self):
        raw_data = {
            "text": [
                "Too short",                             # 9 chars (should be filtered out by min_chars)
                "This is a valid text sentence length.", # 37 chars (should be kept)
                "A" * 6000,                              # 6000 chars (should be filtered out by max_chars)
                "",                                      # empty (filtered out)
            ],
            "source": ["src1", "src1", "src1", "src1"]
        }
        dataset = Dataset.from_dict(raw_data)
        
        config = {
            "lowercase": False,
            "normalize_whitespace": True,
            "strip_whitespace": True,
            "remove_empty_samples": True,
            "min_chars": 20,
            "max_chars": 5000
        }
        
        preprocessed = preprocess_dataset(dataset, config)
        
        # Verify filtering results
        self.assertEqual(len(preprocessed), 1)
        self.assertEqual(preprocessed[0]["text"], "This is a valid text sentence length.")
        self.assertEqual(preprocessed[0]["source"], "src1")


if __name__ == "__main__":
    unittest.main()
