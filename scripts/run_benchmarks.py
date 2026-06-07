import os
import sys
import argparse
import json
import csv

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.evaluation.benchmark import benchmark_config

def print_markdown_table(headers, keys, data_list):
    """Prints a beautiful markdown table from a list of dictionaries."""
    header_str = "| " + " | ".join(headers) + " |"
    sep_str = "| " + " | ".join(["---"] * len(headers)) + " |"
    print(header_str, flush=True)
    print(sep_str, flush=True)
    for data in data_list:
        # Format floating numbers nicely
        row_elements = []
        for k in keys:
            val = data.get(k, "")
            if isinstance(val, float):
                row_elements.append(f"{val:.4f}" if "latency" in k or "memory" in k else f"{val:.2f}")
            elif isinstance(val, int):
                row_elements.append(f"{val:,}" if "params" in k else str(val))
            else:
                row_elements.append(str(val))
        row_str = "| " + " | ".join(row_elements) + " |"
        print(row_str, flush=True)

def main():
    parser = argparse.ArgumentParser(description="Run SLM Benchmarks")
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-410m",
        help="Model identifier to benchmark"
    )
    args = parser.parse_args()
    
    model_name = args.model
    print(f"==================================================", flush=True)
    print(f"Starting Benchmark Suite for Model: {model_name}", flush=True)
    print(f"==================================================\n", flush=True)

    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Control precision for GPU/CPU: fp16 on GPU, fp32 on CPU
    device_precision = "fp16" if torch.cuda.is_available() else "fp32"
    
    # 1. LoRA Rank Comparison (r=8, r=16, r=32)
    # Control: batch_size=1, seq_len=512, precision=device_precision
    print("--- Running LoRA Rank Comparison (r=8, r=16, r=32) ---", flush=True)
    rank_results = []
    for rank in [8, 16, 32]:
        print(f"Benchmarking Rank {rank}...", flush=True)
        res = benchmark_config(
            model_name=model_name,
            lora_rank=rank,
            batch_size=1,
            sequence_length=512,
            precision_mode=device_precision,
        )
        rank_results.append(res)
        
        # Save individual JSON files
        json_path = os.path.join(results_dir, f"benchmark_rank{rank}.json")
        with open(json_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved: {json_path}", flush=True)

    # 2. Sequence Length Scaling (128, 256, 512, 1024)
    # Control: lora_rank=16, batch_size=1, precision=device_precision
    print("\n--- Running Sequence Length Scaling (128, 256, 512, 1024) ---", flush=True)
    seq_results = []
    for seq_len in [128, 256, 512, 1024]:
        print(f"Benchmarking Sequence Length {seq_len}...", flush=True)
        res = benchmark_config(
            model_name=model_name,
            lora_rank=16,
            batch_size=1,
            sequence_length=seq_len,
            precision_mode=device_precision,
        )
        seq_results.append(res)

    # 3. Batch Size Scaling (1, 2, 4, 8)
    # Control: lora_rank=16, seq_len=512, precision=device_precision
    print("\n--- Running Batch Size Scaling (1, 2, 4, 8) ---", flush=True)
    batch_results = []
    for bs in [1, 2, 4, 8]:
        print(f"Benchmarking Batch Size {bs}...", flush=True)
        res = benchmark_config(
            model_name=model_name,
            lora_rank=16,
            batch_size=bs,
            sequence_length=512,
            precision_mode=device_precision,
        )
        batch_results.append(res)

    # 4. Precision Comparison (fp32, fp16)
    # Control: lora_rank=16, batch_size=1, seq_len=512
    # Note: fp16 requires CUDA, so if not available, we skip/note it.
    print("\n--- Running Precision Comparison ---", flush=True)
    precision_modes = ["fp32"]
    if torch.cuda.is_available():
        precision_modes.append("fp16")
    
    prec_results = []
    for prec in precision_modes:
        print(f"Benchmarking Precision Mode {prec}...", flush=True)
        res = benchmark_config(
            model_name=model_name,
            lora_rank=16,
            batch_size=1,
            sequence_length=512,
            precision_mode=prec,
        )
        prec_results.append(res)

    # PRINT REPORT TABLES
    print("\n" + "="*50, flush=True)
    print("BENCHMARK REPORT SUMMARY", flush=True)
    print("="*50 + "\n", flush=True)

    # Table 1: LoRA Rank Comparison
    print("### LoRA Rank Comparison", flush=True)
    print_markdown_table(
        headers=["Rank", "Trainable Params", "Latency (s)", "Tokens/sec", "Peak Memory (GB)"],
        keys=["lora_rank", "trainable_params", "generation_latency_s", "tokens_per_sec", "peak_memory_gb"],
        data_list=rank_results
    )
    print("\n", flush=True)

    # Table 2: Sequence Length Scaling
    print("### Sequence Length Scaling Comparison", flush=True)
    print_markdown_table(
        headers=["Seq Len", "Latency (s)", "Tokens/sec", "Peak Memory (GB)"],
        keys=["sequence_length", "generation_latency_s", "tokens_per_sec", "peak_memory_gb"],
        data_list=seq_results
    )
    print("\n", flush=True)

    # Table 3: Batch Size Scaling
    print("### Batch Size Scaling Comparison", flush=True)
    print_markdown_table(
        headers=["Batch Size", "Latency (s)", "Tokens/sec", "Peak Memory (GB)"],
        keys=["batch_size", "generation_latency_s", "tokens_per_sec", "peak_memory_gb"],
        data_list=batch_results
    )
    print("\n", flush=True)

    # Table 4: Precision Comparison
    print("### Precision Mode Comparison", flush=True)
    print_markdown_table(
        headers=["Precision", "Latency (s)", "Tokens/sec", "Peak Memory (GB)"],
        keys=["precision", "generation_latency_s", "tokens_per_sec", "peak_memory_gb"],
        data_list=prec_results
    )
    print("\n", flush=True)

    # Export a CSV for rank comparison
    rank_csv_path = os.path.join(results_dir, "rank_comparison.csv")
    with open(rank_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lora_rank", "trainable_params", "generation_latency_s", "tokens_per_sec", "peak_memory_gb"])
        for res in rank_results:
            writer.writerow([
                res["lora_rank"],
                res["trainable_params"],
                res["generation_latency_s"],
                res["tokens_per_sec"],
                res["peak_memory_gb"]
            ])
    print(f"Exported Rank Comparison CSV to: {rank_csv_path}", flush=True)

if __name__ == "__main__":
    # Import torch here to avoid import time overhead
    import torch
    main()
