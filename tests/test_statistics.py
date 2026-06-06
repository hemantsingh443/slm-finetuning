import os
import tempfile
import json
import yaml
import unittest
from unittest.mock import patch
from datasets import Dataset
from src.data.statistics import compute_dataset_stats, generate_and_save_all_stats

class TestStatistics(unittest.TestCase):
    def test_compute_dataset_stats(self):
        dummy_data = {
            "text": [
                "Hello World",                  # 11 chars, 2 words
                "This is a longer test text."   # 27 chars, 6 words
            ]
        }
        dataset = Dataset.from_dict(dummy_data)
        stats = compute_dataset_stats(dataset)

        self.assertEqual(stats["samples"], 2)
        self.assertEqual(stats["avg_chars"], 19.0)
        self.assertEqual(stats["avg_words"], 4.0)
        self.assertEqual(stats["max_chars"], 27)
        self.assertEqual(stats["min_chars"], 11)

    @patch("src.data.statistics.load_and_adapt_dataset")
    def test_generate_and_save_all_stats(self, mock_load):
        # Setup mock load output
        mock_load.return_value = Dataset.from_dict({
            "text": ["Mocked dataset text string sample."] * 5,
            "source": ["TinyStories"] * 5
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "datasets.yaml")
            output_path = os.path.join(tmpdir, "dataset_stats.json")
            mixture_path = os.path.join(tmpdir, "mixture_composition.json")

            config_content = {
                "datasets": {
                    "train": [
                        {"name": "roneneldan/TinyStories", "weight": 0.5, "max_samples": 5}
                    ],
                    "eval": []
                },
                "preprocessing": {
                    "min_chars": 5,
                    "max_chars": 1000,
                    "remove_empty_samples": True
                }
            }

            with open(config_path, "w") as f:
                yaml.dump(config_content, f)

            # Generate stats
            stats = generate_and_save_all_stats(config_path, output_path)

            # Check dataset_stats.json
            self.assertTrue(os.path.exists(output_path))
            with open(output_path, "r") as f:
                loaded_stats = json.load(f)
            self.assertIn("train:roneneldan/TinyStories:unknown", loaded_stats)
            self.assertEqual(loaded_stats["train:roneneldan/TinyStories:unknown"]["samples"], 5)

            # Check mixture_composition.json
            self.assertTrue(os.path.exists(mixture_path))
            with open(mixture_path, "r") as f:
                mixture = json.load(f)
            self.assertEqual(mixture["train_total_samples"], 5)
            self.assertEqual(mixture["datasets"][0]["name"], "TinyStories")
            self.assertEqual(mixture["datasets"][0]["weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
