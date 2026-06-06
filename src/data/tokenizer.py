import datasets

class LMTokenizerWrapper:
    """Tokenizer wrapper for causal LM sequence packing."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_and_pack(self, dataset, config: dict):
        tokenization_config = config.get("tokenization", {})
        block_size = tokenization_config.get("max_length", 512)
        num_proc = tokenization_config.get("num_proc", 4)

        is_streaming = isinstance(dataset, datasets.IterableDataset)

        def tokenize_function(examples):
            texts = [
                str(text) + self.tokenizer.eos_token
                for text in examples["text"]
            ]
            return self.tokenizer(texts, add_special_tokens=False)

        remove_cols = [c for c in ["text", "source"] if not is_streaming and c in dataset.column_names]

        tokenized_ds = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=remove_cols,
            num_proc=None if is_streaming else num_proc,
        )

        def group_texts(examples):
            concatenated = {}
            for k in ["input_ids", "attention_mask"]:
                if k in examples:
                    concatenated[k] = sum(examples[k], [])

            total_length = len(concatenated["input_ids"])
            total_length = (total_length // block_size) * block_size

            result = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

            for i in range(0, total_length, block_size):
                chunk_input_ids = concatenated["input_ids"][i : i + block_size]
                chunk_attention_mask = concatenated["attention_mask"][i : i + block_size]

                result["input_ids"].append(chunk_input_ids)
                result["attention_mask"].append(chunk_attention_mask)
                result["labels"].append(chunk_input_ids.copy())

            return result

        packed_ds = tokenized_ds.map(
            group_texts,
            batched=True,
            num_proc=None if is_streaming else num_proc,
        )

        return packed_ds