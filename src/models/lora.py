from peft import LoraConfig, get_peft_model

def get_lora_model(model, lora_config: dict):
    lora_settings = lora_config.get("lora", {})

    peft_config = LoraConfig(
        r=lora_settings.get("rank", 16),
        lora_alpha=lora_settings.get("alpha", 32),
        target_modules=lora_settings.get("target_modules", ["query_key_value"]),
        lora_dropout=lora_settings.get("dropout", 0.05),
        bias=lora_settings.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )

    return get_peft_model(model, peft_config)