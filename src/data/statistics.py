import os
import json
import yaml
from src.data.loader import load_and_adapt_dataset
from src.data.preprocess import preprocess_dataset

def compute_dataset_stats(dataset) -> dict:
    texts = dataset["text"]
    num_samples = len(texts)

    if num_samples == 0:
        return {
            "samples": 0,
            "avg_chars": 0.0,
            "avg_words": 0.0,
            "max_chars": 0,
            "min_chars": 0,
        }

    char_lengths = [len(t) for t in texts]
    word_counts = [len(t.split()) for t in texts]

    return {
        "samples": num_samples,
        "avg_chars": round(sum(char_lengths) / num_samples, 2),
        "avg_words": round(sum(word_counts) / num_samples, 2),
        "max_chars": int(max(char_lengths)),
        "min_chars": int(min(char_lengths)),
    }


def generate_and_save_all_stats(
    config_path: str,
    output_path: str,
    limit_samples: int = None,
) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    preprocessing_config = config.get("preprocessing", {})
    stats_report = {}

    def make_key(split_label, ds_info):
        name = ds_info["name"]
        subset = ds_info.get("subset")
        split = ds_info.get("split", "unknown")
        if subset:
            return f"{split_label}:{name}:{subset}:{split}"
        return f"{split_label}:{name}:{split}"

    def process_and_stats(dataset_list, split_label):
        for ds_info in dataset_list:
            key = make_key(split_label, ds_info)
            print(f"Loading and adapting {split_label} dataset: {key}...")

            ds = load_and_adapt_dataset(ds_info, streaming=False)

            max_s = ds_info.get("max_samples")
            if limit_samples is not None:
                max_s = min(max_s, limit_samples) if max_s else limit_samples

            if max_s and max_s < len(ds):
                ds = ds.select(range(max_s))

            ds = preprocess_dataset(ds, preprocessing_config)
            stats_report[key] = compute_dataset_stats(ds)

    datasets_section = config.get("datasets", {})
    process_and_stats(datasets_section.get("train", []), "train")
    process_and_stats(datasets_section.get("eval", []), "eval")

    # Compile training mixture composition
    mixture_datasets = []
    train_total_samples = 0
    for ds_info in datasets_section.get("train", []):
        key = make_key("train", ds_info)
        raw_name = ds_info["name"]
        short_name = raw_name.split("/")[-1]
        
        samples = stats_report[key]["samples"]
        weight = float(ds_info.get("weight", 1.0))
        
        mixture_datasets.append({
            "name": short_name,
            "samples": samples,
            "weight": weight
        })
        train_total_samples += samples

    # Normalize weights in mixture composition report so they sum to 1.0
    total_weight = sum(d["weight"] for d in mixture_datasets)
    if total_weight > 0:
        for d in mixture_datasets:
            d["weight"] = round(d["weight"] / total_weight, 4)

    mixture_composition = {
        "train_total_samples": train_total_samples,
        "datasets": mixture_datasets
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save stats report
    with open(output_path, "w") as f:
        json.dump(stats_report, f, indent=2)

    # Save mixture composition report
    mixture_composition_path = os.path.join(output_dir, "mixture_composition.json")
    with open(mixture_composition_path, "w") as f:
        json.dump(mixture_composition, f, indent=2)

    print(f"Successfully saved dataset statistics to: {output_path}")
    print(f"Successfully saved mixture composition to: {mixture_composition_path}")
    return stats_report