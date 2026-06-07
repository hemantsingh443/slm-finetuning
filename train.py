import argparse
import os
import yaml
import torch
from datasets import Dataset

from src.data.loader import load_and_adapt_dataset
from src.data.preprocess import preprocess_dataset
from src.data.mixture import create_dataset_mixture
from src.data.tokenizer import LMTokenizerWrapper
from src.models.loader import load_model_and_tokenizer
from src.models.lora import get_lora_model
from src.training.callbacks import WandBLogger
from src.training.trainer import CausalLMTrainer
from src.training.checkpoint_manager import CheckpointManager

def main():
    parser = argparse.ArgumentParser(description="SLM Fine-tuning pipeline")
    parser.add_argument("--steps", type=int, default=None, help="Force number of training steps")
    parser.add_argument("--epochs", type=int, default=None, help="Force number of epochs")
    parser.add_argument("--model_name", type=str, default=None, help="Override model name")
    parser.add_argument("--limit_samples", type=int, default=None, help="Limit number of dataset samples loaded")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--lora_rank", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "fp16", "bf16"], help="Override precision mode")
    args = parser.parse_args()

    # Load configs
    configs_dir = "configs"
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

    # Overrides
    if args.model_name:
        model_config["model"]["name"] = args.model_name
    if args.steps:
        training_config["training"]["max_steps"] = args.steps
        # Automatically make logging, save, and eval steps fit within the test steps
        training_config["training"]["logging_steps"] = max(1, args.steps // 10)
        training_config["training"]["save_steps"] = max(1, args.steps // 2)
        training_config["training"]["eval_steps"] = max(1, args.steps // 2)
    if args.epochs:
        training_config["training"]["epochs"] = args.epochs
    if args.lora_rank:
        lora_config["lora"]["rank"] = args.lora_rank
        lora_config["lora"]["alpha"] = args.lora_rank * 2
    if args.batch_size:
        training_config["training"]["batch_size"] = args.batch_size
    if args.precision:
        training_config["training"]["precision"] = args.precision

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # 1. Load Model & Tokenizer
    print("Loading model and tokenizer...", flush=True)
    model, tokenizer = load_model_and_tokenizer(model_config, training_config)

    # 2. Build datasets
    print("Loading training dataset mixture...", flush=True)
    train_mixture_config = datasets_config.get("datasets", {}).get("train", [])
    preprocessing_config = datasets_config.get("preprocessing", {})
    
    # Always use streaming to avoid downloading massive dataset files (like OpenWebText 24GB)
    use_streaming = True
    total_limit = args.limit_samples if args.limit_samples is not None else 300000

    train_mixture = create_dataset_mixture(
        datasets_config=train_mixture_config,
        preprocessing_config=preprocessing_config,
        streaming=use_streaming,
        seed=training_config.get("seed", 42),
    )

    print(f"Streaming mode enabled. Materializing first {total_limit} samples from stream...", flush=True)
    train_mixture = train_mixture.take(total_limit)
    train_mixture = Dataset.from_list(list(train_mixture))

    print(f"Loaded training dataset of size: {len(train_mixture)}", flush=True)

    # Pack training dataset
    print("Tokenizing and packing training dataset...", flush=True)
    tokenizer_wrapper = LMTokenizerWrapper(tokenizer)
    packed_train_dataset = tokenizer_wrapper.tokenize_and_pack(train_mixture, tokenization_config)
    print(f"Packed training dataset contains {len(packed_train_dataset)} sequences", flush=True)

    # Load and process evaluation datasets
    print("Loading and tokenizing evaluation datasets...", flush=True)
    eval_datasets = {}
    
    def tokenize_eval_fn(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=tokenization_config.get("tokenization", {}).get("max_length", 512),
        )
        outputs["labels"] = [ids.copy() for ids in outputs["input_ids"]]
        return outputs

    for eval_info in datasets_config.get("datasets", {}).get("eval", []):
        raw_name = eval_info["name"]
        short_name = raw_name.split("/")[-1]
        print(f"Processing eval dataset: {short_name}...", flush=True)
        
        eval_ds = load_and_adapt_dataset(eval_info, streaming=use_streaming)
        
        max_samples = eval_info.get("max_samples", None)
        if args.limit_samples:
            max_samples = min(max_samples, args.limit_samples) if max_samples else args.limit_samples
            
        if use_streaming:
            if max_samples:
                eval_ds = eval_ds.take(max_samples)
            eval_ds = Dataset.from_list(list(eval_ds))
        else:
            if max_samples and max_samples < len(eval_ds):
                eval_ds = eval_ds.select(range(max_samples))
            
        eval_ds = preprocess_dataset(eval_ds, preprocessing_config)
        
        # Tokenize without packing
        eval_tokenized = eval_ds.map(
            tokenize_eval_fn,
            batched=True,
            remove_columns=eval_ds.column_names,
        )
        eval_datasets[short_name] = eval_tokenized

    # 3. Apply LoRA PEFT Adapters
    print("Applying LoRA PEFT adapters...", flush=True)
    peft_model = get_lora_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # 4. Initialize W&B Logger
    print("Initializing W&B run...", flush=True)
    wandb_logger = WandBLogger(
        project="slm-finetuning",
        config={
            "model_name": model_config["model"]["name"],
            "lora_rank": lora_config["lora"]["rank"],
            "batch_size": training_config["training"].get("batch_size", 1),
            "learning_rate": training_config["training"].get("learning_rate", 2e-4),
        },
        name=f"run-{model_config['model']['name'].split('/')[-1]}-rank{lora_config['lora']['rank']}",
    )
    wandb_logger.initialize()

    # 5. Initialize CheckpointManager
    checkpoint_manager = CheckpointManager(base_dir=args.checkpoint_dir, wandb_logger=wandb_logger)

    # 6. Initialize Optimizer
    optimizer = torch.optim.AdamW(
        peft_model.parameters(),
        lr=float(training_config["training"].get("learning_rate", 2e-4)),
        weight_decay=float(training_config["training"].get("weight_decay", 0.01)),
    )
    
    # 7. Initialize Trainer and run training
    print("Initializing CausalLMTrainer and starting training...", flush=True)
    trainer = CausalLMTrainer(
        model=peft_model,
        train_dataset=packed_train_dataset,
        optimizer=optimizer,
        device=device,
        training_config=training_config,
        wandb_logger=wandb_logger,
        tokenizer=tokenizer,
        checkpoint_manager=checkpoint_manager,
        eval_dataset=eval_datasets,
    )
    
    trainer.train()
    
    # Finish W&B
    wandb_logger.finish()
    print("Training pipeline run completed successfully!", flush=True)

if __name__ == "__main__":
    main()
