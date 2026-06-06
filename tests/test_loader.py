import unittest
from unittest.mock import patch
from datasets import Dataset
from src.data.loader import load_and_adapt_dataset

class TestLoader(unittest.TestCase):
    @patch("src.data.loader.load_dataset")
    def test_load_and_adapt_dataset(self, mock_load):
        # Create mock raw dataset
        dummy_raw = {
            "text": ["  This is some raw  news text!  "],
            "label": [1]
        }
        mock_dataset = Dataset.from_dict(dummy_raw)
        mock_load.return_value = mock_dataset

        dataset_info = {
            "name": "fancyzhx/ag_news",
            "split": "test",
            "remove_columns": ["text", "label"],
            "source_name": "custom_ag_news"
        }

        # Run loading and adaptation
        adapted_ds = load_and_adapt_dataset(dataset_info, streaming=False)

        # Validate columns and contents
        self.assertEqual(adapted_ds.column_names, ["text", "source"])
        elem = adapted_ds[0]
        self.assertEqual(elem["text"], "This is some raw news text!")
        self.assertEqual(elem["source"], "custom_ag_news")


if __name__ == "__main__":
    unittest.main()
