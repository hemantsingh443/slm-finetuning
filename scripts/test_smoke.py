import os
import sys
import yaml
import unittest

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.loader import load_model_and_tokenizer
from src.models.lora import get_lora_model

class TestModelSmoke(unittest.TestCase):
    def test_model_lora_forward_pass(self):
        print("\n--- Running SLM Model & LoRA Integration Smoke Test ---", flush=True)

        # 1. Load Configurations
        configs_dir = os.path.join(project_root, "configs")
        with open(os.path.join(configs_dir, "model.yaml"), "r") as f:
            model_config = yaml.safe_load(f)
        with open(os.path.join(configs_dir, "training.yaml"), "r") as f:
            training_config = yaml.safe_load(f)
        with open(os.path.join(configs_dir, "lora.yaml"), "r") as f:
            lora_config = yaml.safe_load(f)

        # 2. Load Model and Tokenizer
        print("Loading model and tokenizer...", flush=True)
        model, tokenizer = load_model_and_tokenizer(model_config, training_config)
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)

        # 3. Apply LoRA PEFT Adapters
        print("Applying LoRA PEFT adapters...", flush=True)
        peft_model = get_lora_model(model, lora_config)
        self.assertIsNotNone(peft_model)

        # 4. Count and Log Parameter Statistics
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in peft_model.parameters())
        non_trainable_params = all_params - trainable_params

        print(f"Trainable parameters: {trainable_params:,}", flush=True)
        print(f"Non-trainable parameters: {non_trainable_params:,}", flush=True)
        print(f"Total parameters: {all_params:,}", flush=True)
        peft_model.print_trainable_parameters()

        self.assertGreater(trainable_params, 0, "Expected at least one trainable LoRA parameter.")

        # 5. Execute Tiny Forward Pass on Mock Data
        print("Executing forward pass on mock input data...", flush=True)
        device = "cuda" if next(peft_model.parameters()).is_cuda else "cpu"
        
        mock_input = "SLM Fine-tuning pipeline smoke test."
        inputs = tokenizer(mock_input, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = peft_model(**inputs)
        logits = outputs.logits
        self.assertIsNotNone(logits)
        
        print(f"Forward pass completed successfully. Output logits shape: {list(logits.shape)}", flush=True)


if __name__ == "__main__":
    unittest.main()
