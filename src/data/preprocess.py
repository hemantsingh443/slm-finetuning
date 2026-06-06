import re

def preprocess_text(text: str, config: dict) -> str:
    """Clean a single text string."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    if config.get("lowercase", False):
        text = text.lower()

    if config.get("normalize_whitespace", True):
        text = re.sub(r"\s+", " ", text)

    if config.get("strip_whitespace", True):
        text = text.strip() 

    return text


def preprocess_dataset(dataset, config: dict):
    """Apply preprocessing to a canonical dataset with 'text' and optional 'source'."""
    remove_empty_samples = config.get("remove_empty_samples", True)
    min_chars = config.get("min_chars", 20)
    max_chars = config.get("max_chars", 5000)

    def map_fn(example):
        text = preprocess_text(example.get("text"), config)
        out = {"text": text}

        # Keep source if present, but do not assume it always exists
        if "source" in example:
            out["source"] = example["source"]

        return out

    preprocessed_ds = dataset.map(map_fn)

    # Filter by min and max character lengths
    preprocessed_ds = preprocessed_ds.filter(
        lambda x: isinstance(x["text"], str)
        and min_chars <= len(x["text"]) <= max_chars
    )

    if remove_empty_samples:
        preprocessed_ds = preprocessed_ds.filter(
            lambda x: isinstance(x["text"], str) and len(x["text"]) > 0
        )

    return preprocessed_ds