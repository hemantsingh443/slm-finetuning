import os
import sys
import unittest
from unittest.mock import patch
import torch
from datasets import Dataset

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.mixture import create_dataset_mixture
from src.data.tokenizer import LMTokenizerWrapper
from src.models.loader import load_model_and_tokenizer
from src.models.lora import get_lora_model
from src.training.callbacks import initialize_wandb, WandBLogger
from src.training.trainer import CausalLMTrainer

# Set W&B to offline mode to run unit tests without internet/credentials
os.environ["WANDB_MODE"] = "offline"

class TestTrainingSmoke(unittest.TestCase):

    def test_wandb_logger_and_verification(self):
        print("\n--- Verifying W&B Logger ---", flush=True)
        # Test direct initialization, logging, and finish
        run = initialize_wandb(project="smoke-test", name="verify-run")
        self.assertIsNotNone(run)
        
        import wandb
        wandb.log({"test_metric": 123})
        wandb.finish()
        print("W&B verified successfully.", flush=True)

    @patch("src.data.mixture.load_and_adapt_dataset")
    def test_training_pipeline_smoke(self, mock_load):
        print("\n--- Running Training Pipeline Smoke Test ---", flush=True)
        
        # 1. Mock Raw Dataset with 100 samples
        mock_text = "Once upon a time, there was a tiny little story that had some characters in it."
        raw_data = {
            "text": [mock_text] * 100,
            "source": ["TinyStories"] * 100
        }
        mock_ds = Dataset.from_dict(raw_data)
        mock_load.return_value = mock_ds

        # 2. Dataset Mixture
        datasets_config = [
            {"name": "roneneldan/TinyStories", "weight": 1.0}
        ]
        preprocessing_config = {
            "min_chars": 10,
            "max_chars": 5000,
            "remove_empty_samples": True,
            "lowercase": False,
            "normalize_whitespace": True,
            "strip_whitespace": True
        }
        mixture_ds = create_dataset_mixture(
            datasets_config=datasets_config,
            preprocessing_config=preprocessing_config,
            streaming=False,
            seed=42
        )
        self.assertEqual(len(mixture_ds), 100)

        # 3. Model & Tokenizer Loader
        model_config = {
            "model": {
                "name": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
            },
            "tokenizer": {
                "use_fast": True,
                "trust_remote_code": False
            }
        }
        training_config = {
            "training": {
                "precision": "fp32",
                "gradient_checkpointing": False,
                "batch_size": 1,
                "epochs": 1,
                "logging_steps": 1
            }
        }
        
        print("Loading tiny random model and tokenizer...", flush=True)
        model, tokenizer = load_model_and_tokenizer(model_config, training_config)
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)

        # 4. Tokenization & Packing
        tokenization_config = {
            "tokenization": {
                "max_length": 16,  # small block size to yield many blocks
                "num_proc": 1
            }
        }
        wrapper = LMTokenizerWrapper(tokenizer)
        packed_ds = wrapper.tokenize_and_pack(mixture_ds, tokenization_config)
        
        # Assert that we have enough samples (at least 10)
        self.assertGreaterEqual(len(packed_ds), 10, "Packed dataset is too small for 10 steps.")
        print(f"Packed dataset size: {len(packed_ds)} samples", flush=True)

        # 5. Apply LoRA Adapters
        lora_config = {
            "lora": {
                "target_modules": ["query_key_value"],
                "rank": 4,
                "alpha": 8,
                "dropout": 0.05,
                "bias": "none"
            }
        }
        peft_model = get_lora_model(model, lora_config)
        self.assertIsNotNone(peft_model)

        # 6. Optimizer
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-3)

        # 7. Initialize WandBLogger for training
        wandb_logger = WandBLogger(project="smoke-test", name="training-run")
        wandb_logger.initialize()

        # 8. Train single epoch (first 10 steps)
        # Select the first block and replicate it 10 times to ensure training is on identical inputs,
        # which guarantees the loss consistently decreases.
        tiny_train_dataset = packed_ds.select([0] * 10)
        
        trainer = CausalLMTrainer(
            model=peft_model,
            train_dataset=tiny_train_dataset,
            optimizer=optimizer,
            device="cpu",
            training_config=training_config,
            wandb_logger=wandb_logger,
            tokenizer=tokenizer
        )
        
        losses = trainer.train()
        
        # Evaluate model after training to verify validation pipeline and metrics logging
        eval_metrics = trainer.evaluate(tiny_train_dataset, name="smoke_val", prompts=["The future of AI"])
        self.assertIsNotNone(eval_metrics)
        self.assertIn("overall", eval_metrics)
        self.assertIn("token_loss", eval_metrics["overall"])
        
        wandb_logger.finish()

        self.assertEqual(len(losses), 10, "Expected exactly 10 training steps.")
        
        # Assert loss decreases
        first_loss = losses[0]
        last_loss = losses[-1]
        print(f"First loss: {first_loss:.4f}, Last loss: {last_loss:.4f}", flush=True)
        self.assertLess(last_loss, first_loss, "Expected loss to decrease after training steps.")
        print("Training smoke test completed successfully. Loss decreased.", flush=True)


if __name__ == "__main__":
    unittest.main()
