# SLM Fine-Tuning and Performance Scaling (Pythia-410m LoRA)

This directory contains the codebase, benchmarking sweeps, and technical report for low-rank adaptation (LoRA) fine-tuning of the Pythia-410m autoregressive language model.

For full details on empirical evaluations, domain perplexity analysis, qualitative validation, and hardware sweeps, please see the technical report:

- **[Read the Technical Report PDF (report/report.pdf)](report/report.pdf)**

Interactive training logs, hyperparameter settings, and loss convergence charts are publicly viewable on Weights & Biases:

- **[Weights & Biases Project Dashboard](https://wandb.ai/hemantsingh11119-learn/slm-finetuning)**

---

## Project Structure

```
slm-finetuning/
├── README.md                      # This project guide (links to report/report.pdf)
├── train.py                       # Main training entry point
├── requirements.txt               # Python package dependencies
├── configs/                       # Configuration files for training & evaluation
│   ├── datasets.yaml              # Dataset mixture paths and weightings
│   ├── lora.yaml                  # LoRA adapter rank & scaling hyperparameters
│   ├── model.yaml                 # Base model selection (Pythia-410m)
│   └── tokenization.yaml          # Sequence packing and tokenizer parameters
├── src/                           # Source code modules
│   ├── data/                      # Dataset mixtures loading & sequence packing
│   ├── model/                     # LoRA injection and PEFT helper utils
│   ├── training/                  # Custom Hugging Face CausalLMTrainer wrapper
│   └── evaluation/                # Perplexity evaluation and benchmarking metrics
├── scripts/                       # Run scripts
│   ├── run_benchmarks.py          # Main hardware sweeps profiling script
│   └── run_full_pipeline.py       # Full training & validation runner
├── report/                        # Technical report assets and source files
│   ├── report.tex                 # LaTeX report source document (native TikZ)
│   ├── report.pdf                 # Compiled 9-page technical report (PDF)
│   └── plots/                     # Vector PDF figures used in the report
│       ├── training_loss_comparison.pdf
│       ├── validation_perplexity_comparison.pdf
│       ├── batch_size_scaling_profile.pdf
│       └── seq_len_memory_scaling.pdf
└── experiments_results/           # Saved checkpoints, metrics, and logs
```

---

## Quick Start

### 1. Installation

Create a virtual environment and install the required dependencies:

```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Fine-Tuning
Execute the fine-tuning training pipeline on Pythia-410m using default configurations:
```bash
python train.py
```
Alternatively, you can override hyperparameter configurations directly via command-line flags:
```bash
python train.py --lora_rank 16 --batch_size 4 --steps 2000 --precision fp16
```

### 3. Profiling & Benchmarking
Run the hardware scaling profiling sweeps (measuring throughput, latency, and VRAM footprint across batch sizes, sequence lengths, and precision modes) for the base model or a specific custom checkpoint:
```bash
# Benchmark the default model config
python scripts/run_benchmarks.py

# Benchmark a specific fine-tuned local checkpoint
python scripts/run_benchmarks.py --model checkpoints/final
```

### 4. Automated End-to-End Pipeline
You can run the entire pipeline (fine-tuning a model and running the benchmark sweep on the resulting final checkpoint sequentially) using the `run_full_pipeline.py` orchestrator script.

For example, to run a quick automated training and benchmarking smoke test:
```bash
python scripts/run_full_pipeline.py --model_name EleutherAI/pythia-160m --steps 10 --limit_samples 1000 --checkpoint_dir checkpoints/smoke_test
```
