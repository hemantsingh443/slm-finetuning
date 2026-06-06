from datasets import load_dataset
from src.data.adapters import get_adapter

def load_and_adapt_dataset(dataset_info, streaming=False):
    """Loads a Hugging Face dataset and maps it to a canonical format.
    
    The canonical format contains a 'text' column of type string and a 'source'
    column indicating the dataset source name.
    
    Args:
        dataset_info (dict): Configuration dictionary containing keys:
            - name (str): Dataset identifier
            - split (str): Target split to load
            - subset (str, optional): Configuration subset name
            - remove_columns (list, optional): Explicit list of columns to remove.
              Required for streaming datasets.
            - source_name (str, optional): Custom source tag. Defaults to name.
        streaming (bool): If True, loads the dataset in streaming mode.
        
    Returns:
        Dataset or IterableDataset: The adapted and filtered dataset.
    """
    name = dataset_info["name"]
    split = dataset_info.get("split")
    subset = dataset_info.get("subset")
    adapter = get_adapter(name)

    # Load from HF Hub
    if subset:
        ds = load_dataset(name, subset, split=split, streaming=streaming)
    else:
        ds = load_dataset(name, split=split, streaming=streaming)

    # Adapter mapping function
    def map_fn(example):
        text = adapter.adapt(example)
        return {
            "text": text,
            "source": dataset_info.get("source_name", name),
        }

    # Get the columns to remove
    raw_columns = dataset_info.get("remove_columns")
    if raw_columns is None and not streaming:
        raw_columns = ds.column_names

    # Map the dataset
    adapted_ds = ds.map(map_fn, remove_columns=raw_columns)

    # Remove empty samples
    adapted_ds = adapted_ds.filter(lambda x: isinstance(x["text"], str) and len(x["text"]) > 0)

    return adapted_ds