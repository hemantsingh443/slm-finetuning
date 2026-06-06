import unittest
from unittest.mock import patch
from datasets import Dataset
from src.data.mixture import create_dataset_mixture

class TestMixture(unittest.TestCase):
    @patch("src.data.mixture.load_and_adapt_dataset")
    def test_create_dataset_mixture(self, mock_load):
        # Mock loader side effects to return specific dummy datasets
        def load_side_effect(dataset_info, streaming=False):
            name = dataset_info["name"]
            if "TinyStories" in name:
                raw_data = {
                    "text": ["Once upon a time in a test story."] * 10,
                    "source": ["TinyStories"] * 10
                }
            else:
                raw_data = {
                    "text": ["Wikipedia article text context here."] * 10,
                    "source": ["WikiText"] * 10
                }
            return Dataset.from_dict(raw_data)

        mock_load.side_effect = load_side_effect

        datasets_config = [
            {"name": "roneneldan/TinyStories", "weight": 0.5},
            {"name": "Salesforce/wikitext", "weight": 0.5}
        ]

        preprocessing_config = {
            "min_chars": 10,
            "max_chars": 5000,
            "remove_empty_samples": True
        }

        # Interleave datasets
        mixture = create_dataset_mixture(
            datasets_config=datasets_config,
            preprocessing_config=preprocessing_config,
            streaming=False,
            seed=42
        )

        # Assert correct columns and interleaved sources
        self.assertEqual(set(mixture.column_names), {"text", "source"})
        self.assertGreater(len(mixture), 0)
        sources = set(mixture["source"])
        self.assertEqual(sources, {"TinyStories", "WikiText"})


if __name__ == "__main__":
    unittest.main()
