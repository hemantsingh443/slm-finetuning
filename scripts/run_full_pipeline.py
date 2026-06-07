import os
import sys
import argparse
import subprocess

def run_command(command_list, description):
    print(f"\n==================================================", flush=True)
    print(f"Executing: {description}", flush=True)
    print(f"Command: {' '.join(command_list)}", flush=True)
    print(f"==================================================\n", flush=True)
    
    # Run synchronously and stream output
    env = os.environ.copy()
    if "WANDB_MODE" not in env:
        env["WANDB_MODE"] = "offline"
        
    process = subprocess.Popen(
        command_list,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env
    )
    process.wait()
    
    if process.returncode != 0:
        print(f"\n[ERROR] {description} failed with return code {process.returncode}.", flush=True)
        sys.exit(process.returncode)
    else:
        print(f"\n[SUCCESS] {description} completed successfully.\n", flush=True)

def main():
    parser = argparse.ArgumentParser(description="SLM Automated Training & Benchmarking Pipeline")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-160m", help="Base model identifier")
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--limit_samples", type=int, default=None, help="Limit dataset samples loaded")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/automated_run", help="Where to save checkpoints")
    parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank for PEFT config")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "fp16", "bf16"], help="Model precision mode")
    
    args = parser.parse_args()
    
    # Locate Python interpreter
    python_exe = sys.executable
    if not python_exe:
        python_exe = "python"

    # Step 1: Run Training
    train_cmd = [
        python_exe,
        "train.py",
        "--model_name", args.model_name,
        "--checkpoint_dir", args.checkpoint_dir
    ]
    if args.steps:
        train_cmd.extend(["--steps", str(args.steps)])
    if args.epochs:
        train_cmd.extend(["--epochs", str(args.epochs)])
    if args.limit_samples:
        train_cmd.extend(["--limit_samples", str(args.limit_samples)])
    if args.lora_rank:
        train_cmd.extend(["--lora_rank", str(args.lora_rank)])
    if args.batch_size:
        train_cmd.extend(["--batch_size", str(args.batch_size)])
    if args.precision:
        train_cmd.extend(["--precision", args.precision])
        
    run_command(train_cmd, f"Fine-tuning model: {args.model_name}")

    # Step 2: Run Benchmarking
    final_checkpoint_dir = os.path.join(args.checkpoint_dir, "final")
    
    # Check if final checkpoint folder exists
    if not os.path.exists(final_checkpoint_dir):
        print(f"[ERROR] Final checkpoint not found at: {final_checkpoint_dir}", flush=True)
        sys.exit(1)
        
    benchmark_cmd = [
        python_exe,
        "scripts/run_benchmarks.py",
        "--model", final_checkpoint_dir
    ]
    
    run_command(benchmark_cmd, f"Running benchmark sweeps on checkpoint: {final_checkpoint_dir}")

    print("==================================================", flush=True)
    print("ALL AUTOMATED PIPELINE STEPS COMPLETED!", flush=True)
    print(f"Checkpoints directory: {args.checkpoint_dir}", flush=True)
    print("Benchmark reports are generated in the 'results/' directory.", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    main()
