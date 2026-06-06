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

