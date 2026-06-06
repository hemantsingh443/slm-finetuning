import unittest
from datasets import Dataset
from src.data.tokenizer import LMTokenizerWrapper

class FakeTokenizer:
    def __init__(self):
        self.eos_token = "<eos>"
        self.pad_token = None
        self.eos_token_id = 2
        self.pad_token_id = 3

    def __call__(self, texts, add_special_tokens=False):
        # Convert each character code to mock tokens
        input_ids = []
        attention_mask = []
        for text in texts:
            ids = [ord(char) % 100 for char in text]
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


class TestTokenizer(unittest.TestCase):
    def test_tokenize_and_pack(self):
        # Define mock text dataset
        raw_data = {
            "text": ["Hello", "World!"],
            "source": ["TinyStories", "TinyStories"]
        }
        dataset = Dataset.from_dict(raw_data)

        tokenizer = FakeTokenizer()
        wrapper = LMTokenizerWrapper(tokenizer)

        # Check pad token fallback mapping
        self.assertEqual(tokenizer.pad_token, "<eos>")
        self.assertEqual(tokenizer.pad_token_id, tokenizer.pad_token_id)

        config = {
            "tokenization": {
                "max_length": 8,
                "num_proc": 1
            }
        }

        # Tokenize and pack sequences
        packed_ds = wrapper.tokenize_and_pack(dataset, config)

        # Check column names
        self.assertEqual(set(packed_ds.column_names), {"input_ids", "attention_mask", "labels"})
        self.assertGreater(len(packed_ds), 0)

        # Validate sequence shapes and labels masking
        for row in packed_ds:
            self.assertEqual(len(row["input_ids"]), 8)
            self.assertEqual(len(row["attention_mask"]), 8)
            self.assertEqual(len(row["labels"]), 8)

            # Check that padding tokens are masked out with 0 attention and -100 labels
            for idx, mask in enumerate(row["attention_mask"]):
                if mask == 0:
                    self.assertEqual(row["input_ids"][idx], tokenizer.pad_token_id)
                    self.assertEqual(row["labels"][idx], -100)
                else:
                    # Non-padding should retain labels mapped directly from input_ids
                    self.assertEqual(row["labels"][idx], row["input_ids"][idx])


if __name__ == "__main__":
    unittest.main()
