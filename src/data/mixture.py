from datasets import interleave_datasets
from src.data.loader import load_and_adapt_dataset
from src.data.preprocess import preprocess_dataset

def create_dataset_mixture(
    datasets_config: list,
    preprocessing_config: dict,
    streaming: bool = False,
    seed: int = 42,
    stopping_strategy: str = "all_exhausted",
):
    if not datasets_config:
        raise ValueError("datasets_config is empty; cannot construct mixture.")

    loaded_datasets = []
    weights = []

    for dataset_info in datasets_config:
        adapted_ds = load_and_adapt_dataset(dataset_info, streaming=streaming)
        preprocessed_ds = preprocess_dataset(adapted_ds, preprocessing_config)

        # Shuffle each dataset before mixing
        if not streaming:
            preprocessed_ds = preprocessed_ds.shuffle(seed=seed)

        loaded_datasets.append(preprocessed_ds)
        weights.append(float(dataset_info.get("weight", 1.0)))

    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError(f"Sum of dataset weights must be positive, got {total_weight}")

    probabilities = [w / total_weight for w in weights]

    return interleave_datasets(
        datasets=loaded_datasets,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )