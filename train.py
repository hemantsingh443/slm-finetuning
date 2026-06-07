import os
import yaml
import argparse
import torch
from transformers import get_cosine_schedule_with_warmup

from src.data.mixture import create_dataset_mixture
from src.data.tokenizer import LMTokenizerWrapper
from src.data.loader import load_and_adapt_dataset
from src.data.preprocess import preprocess_dataset
from src.data.statistics import generate_and_save_all_stats
from src.models.loader import load_model_and_tokenizer
from src.models.lora import get_lora_model
from src.training.callbacks import WandBLogger
from src.training.trainer import CausalLMTrainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Causal LM with LoRA")
    parser.add_argument("--configs_dir", type=str, default="configs", help="Directory containing config files")
    parser.add_argument("--limit_samples", type=int, default=None, help="Limit number of train/eval samples for quick testing")
    parser.add_argument("--wandb_project", type=str, default="slm-finetuning", help="W&B Project Name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B Run Name")
    args = parser.parse_args()

    # 1. Load configs
    configs_dir = args.configs_dir
    with open(os.path.join(configs_dir, "model.yaml"), "r") as f:
        model_config = yaml.safe_load(f)
    with open(os.path.join(configs_dir, "training.yaml"), "r") as f:
        training_config = yaml.safe_load(f)
    with open(os.path.join(configs_dir, "lora.yaml"), "r") as f:
        lora_config = yaml.safe_load(f)
    with open(os.path.join(configs_dir, "datasets.yaml"), "r") as f:
        datasets_config = yaml.safe_load(f)
    with open(os.path.join(configs_dir, "tokenization.yaml"), "r") as f:
        tokenization_config = yaml.safe_load(f)
    with open(os.path.join(configs_dir, "evaluation.yaml"), "r") as f:
        evaluation_config = yaml.safe_load(f)

    # Combine configurations for W&B
    full_config = {
        **model_config,
        **training_config,
        **lora_config,
        **datasets_config,
        **tokenization_config,
        **evaluation_config
    }

    # 2. Generate and save dataset statistics report
    print("\n--- Generating Dataset Statistics ---")
    generate_and_save_all_stats(
        config_path=os.path.join(configs_dir, "datasets.yaml"),
        output_path="reports/dataset_stats.json",
        limit_samples=args.limit_samples
    )

    # 3. Load Model and Tokenizer
    print("\n--- Loading Model and Tokenizer ---")
    model, tokenizer = load_model_and_tokenizer(model_config, training_config)
    
    # 4. Apply LoRA Adaptor
    print("\n--- Applying LoRA adapter config ---")
    peft_model = get_lora_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # 5. Build datasets
    print("\n--- Loading and adapting training datasets mixture ---")
    train_datasets = datasets_config.get("datasets", {}).get("train", [])
    preprocessing_config = datasets_config.get("preprocessing", {})
    
    # Optional limit for quick run
    if args.limit_samples:
        for ds_info in train_datasets:
            ds_info["max_samples"] = min(ds_info.get("max_samples", float('inf')), args.limit_samples)

    train_mixture = create_dataset_mixture(
        datasets_config=train_datasets,
        preprocessing_config=preprocessing_config,
        streaming=False,
        seed=training_config.get("seed", 42)
    )

    # Tokenize and pack training dataset
    print("\n--- Tokenizing and Packing training dataset ---")
    tokenizer_wrapper = LMTokenizerWrapper(tokenizer)
    tokenized_train_ds = tokenizer_wrapper.tokenize_and_pack(train_mixture, tokenization_config)
    print(f"Packed training dataset contains {len(tokenized_train_ds)} samples.")

    # Load and preprocess evaluation datasets (evaluate them separately)
    eval_datasets = datasets_config.get("datasets", {}).get("eval", [])
    tokenized_eval_datasets = {}
    for eval_ds_info in eval_datasets:
        name = eval_ds_info["name"]
        short_name = name.split("/")[-1]
        print(f"Loading evaluation dataset: {short_name}...")
        eval_ds = load_and_adapt_dataset(eval_ds_info, streaming=False)
        
        if args.limit_samples:
            max_s = min(eval_ds_info.get("max_samples", float('inf')), args.limit_samples)
            if max_s < len(eval_ds):
                eval_ds = eval_ds.select(range(max_s))
                
        eval_ds = preprocess_dataset(eval_ds, preprocessing_config)
        
        # Tokenize (DO NOT PACK evaluation datasets to preserve length buckets analysis)
        def tokenize_eval_fn(examples):
            return tokenizer(examples["text"], truncation=True, max_length=tokenization_config.get("tokenization", {}).get("max_length", 512))
            
        remove_cols = [c for c in ["text", "source"] if c in eval_ds.column_names]
        tokenized_eval_ds = eval_ds.map(
            tokenize_eval_fn,
            batched=True,
            remove_columns=remove_cols,
            num_proc=tokenization_config.get("tokenization", {}).get("num_proc", 4)
        )
        
        # Set target labels to input_ids (and make sure padding tokens are masked to -100)
        def map_eval_labels(example):
            example["labels"] = example["input_ids"].copy()
            return example
            
        tokenized_eval_ds = tokenized_eval_ds.map(map_eval_labels)
        tokenized_eval_datasets[short_name] = tokenized_eval_ds

    # 6. Initialize W&B Logger
    print("\n--- Initializing W&B Logger ---")
    wandb_logger = WandBLogger(
        project=args.wandb_project,
        config=full_config,
        name=args.wandb_name
    )
    wandb_logger.initialize()

    # 7. Optimizer & Scheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.AdamW(
        peft_model.parameters(),
        lr=float(training_config.get("training", {}).get("learning_rate", 2e-4)),
        weight_decay=float(training_config.get("training", {}).get("weight_decay", 0.01))
    )

    # Cosine scheduler setup
    num_training_steps = len(tokenized_train_ds) * int(training_config.get("training", {}).get("epochs", 1))
    warmup_steps = int(num_training_steps * float(training_config.get("training", {}).get("warmup_ratio", 0.03)))
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    # 8. Trainer
    print("\n--- Initializing CausalLMTrainer ---")
    trainer = CausalLMTrainer(
        model=peft_model,
        train_dataset=tokenized_train_ds,
        optimizer=optimizer,
        device=device,
        training_config=training_config,
        wandb_logger=wandb_logger,
        scheduler=scheduler,
        tokenizer=tokenizer
    )

    # 9. Train and Evaluate
    print("\n--- Starting Model Training ---")
    _ = trainer.train()

    print("\n--- Starting Evaluation ---")
    eval_results = {}
    prompts = ["Once upon a time", "The future of AI", "Deep learning is"]
    
    for short_name, eval_ds in tokenized_eval_datasets.items():
        print(f"\nEvaluating dataset: {short_name}...")
        metrics = trainer.evaluate(eval_ds, name=short_name, prompts=prompts)
        eval_results[short_name] = metrics

    # Save final evaluation results report
    os.makedirs("reports", exist_ok=True)
    with open("reports/eval_results.json", "w") as f:
        import json
        json.dump(eval_results, f, indent=2)
    print("\nEvaluation results saved to reports/eval_results.json")

    # 10. Close Logger and Save Model
    wandb_logger.finish()
    
    # Save adapter checkpoints
    peft_model.save_pretrained("checkpoints/final_lora_adapter")
    tokenizer.save_pretrained("checkpoints/final_lora_adapter")
    print("Model checkpoints saved to checkpoints/final_lora_adapter")

if __name__ == "__main__":
    main()
