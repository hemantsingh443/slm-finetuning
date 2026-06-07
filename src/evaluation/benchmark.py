import time
import torch
import gc
import os
from src.models.loader import load_model_and_tokenizer
from src.models.lora import get_lora_model

def get_peak_memory_gb() -> float:
    """Returns the peak memory allocated in GB (CUDA peak, fallback to RSS on CPU)."""
    if torch.cuda.is_available():
        # Clear cache and get peak
        return torch.cuda.max_memory_allocated() / (1024**3)
    else:
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except ImportError:
            return 0.0

def benchmark_config(
    model_name: str = "EleutherAI/pythia-410m",
    lora_rank: int = 8,
    batch_size: int = 1,
    sequence_length: int = 512,
    precision_mode: str = "fp32",
    device: str = None,
) -> dict:
    """Benchmarks a single configuration for latency, throughput, and memory usage."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reset peak stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()

    model_config = {
        "model": {"name": model_name},
        "tokenizer": {"use_fast": True, "trust_remote_code": False}
    }
    training_config = {
        "training": {
            "precision": precision_mode,
            "gradient_checkpointing": False
        }
    }
    lora_config = {
        "lora": {
            "target_modules": ["query_key_value"],
            "rank": lora_rank,
            "alpha": lora_rank * 2,
            "dropout": 0.05,
            "bias": "none"
        }
    }

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config, training_config)
    
    # Apply LoRA Peft adapter
    peft_model = get_lora_model(model, lora_config)
    peft_model = peft_model.to(device)
    peft_model.eval()

    # Model parameters count
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in peft_model.parameters())

    # Generate mock inputs
    # Use random integer tokens within vocabulary size
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 50257
    input_ids = torch.randint(10, min(1000, vocab_size - 1), (batch_size, sequence_length), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = peft_model(input_ids)

    # Reset memory stats again after loading and warming up to isolate generation footprint
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 1. Forward Pass Latency
    start_fwd = time.perf_counter()
    with torch.no_grad():
        _ = peft_model(input_ids)
    fwd_latency = time.perf_counter() - start_fwd

    # 2. Generation Latency and Speed
    max_new_tokens = 50
    start_gen = time.perf_counter()
    with torch.no_grad():
        outputs = peft_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_latency = time.perf_counter() - start_gen

    # Calculate tokens/sec
    generated_tokens = outputs.shape[1] - input_ids.shape[1]
    total_tokens = generated_tokens * batch_size
    tokens_per_sec = total_tokens / gen_latency if gen_latency > 0 else 0.0

    # Calculate peak memory
    peak_memory = get_peak_memory_gb()

    # Clean up model
    del peft_model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "model_name": model_name,
        "lora_rank": lora_rank,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "precision": precision_mode,
        "trainable_params": trainable_params,
        "all_params": all_params,
        "forward_latency_s": round(fwd_latency, 4),
        "generation_latency_s": round(gen_latency, 4),
        "generated_tokens": generated_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "peak_memory_gb": round(peak_memory, 4),
    }
