import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

def load_model_and_tokenizer(model_config: dict, training_config: dict = None):
    model_settings = model_config.get("model", {})
    model_name = model_settings.get("name", "EleutherAI/pythia-410m")

    tokenizer_settings = model_config.get("tokenizer", {})
    use_fast = tokenizer_settings.get("use_fast", True)
    trust_remote_code = tokenizer_settings.get("trust_remote_code", False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = None
    gradient_checkpointing = False

    if training_config:
        train_settings = training_config.get("training", {})
        precision = train_settings.get("precision", "fp32")

        if precision == "fp16":
            torch_dtype = torch.float16
        elif precision == "bf16":
            torch_dtype = torch.bfloat16

        gradient_checkpointing = train_settings.get("gradient_checkpointing", False)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model, tokenizer


def load_peft_checkpoint_or_base(model_name: str, training_config: dict = None):
    """Loads either a base causal LM model or a wrapped PEFT checkpoint."""
    import os
    import json
    from peft import PeftModel
    
    adapter_config_path = os.path.join(model_name, "adapter_config.json")
    if os.path.isdir(model_name) and os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path")
        
        # Load base model & tokenizer
        model_config = {
            "model": {"name": base_model_name},
            "tokenizer": {"use_fast": True, "trust_remote_code": False}
        }
        base_model, tokenizer = load_model_and_tokenizer(model_config, training_config)
        
        print(f"Loading PEFT adapter from checkpoint: {model_name}...", flush=True)
        model = PeftModel.from_pretrained(base_model, model_name)
        return model, tokenizer
    else:
        model_config = {
            "model": {"name": model_name},
            "tokenizer": {"use_fast": True, "trust_remote_code": False}
        }
        return load_model_and_tokenizer(model_config, training_config)


