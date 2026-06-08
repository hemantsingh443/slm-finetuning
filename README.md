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

Execute the fine-tuning training pipeline on Pythia-410m:

```bash
python train.py
```

### 3. Profiling & Benchmarking

Run the hardware scaling profiling sweep (measuring generation throughput, latency, and VRAM memory footprint across batch sizes, sequence lengths, and precision modes):

```bash
python scripts/run_benchmarks.py
```
