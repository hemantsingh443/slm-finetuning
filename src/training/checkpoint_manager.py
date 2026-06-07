import os
import torch

class CheckpointManager:
    """Manages saving and tracking of checkpoints for PEFT models."""

    def __init__(self, base_dir: str = "checkpoints", wandb_logger=None):
        self.base_dir = base_dir
        self.wandb_logger = wandb_logger
        os.makedirs(self.base_dir, exist_ok=True)

    def save_checkpoint(self, model, step: int, epoch: int, is_final: bool = False):
        """Saves a model checkpoint and logs stats to W&B."""
        if is_final:
            folder_name = "final"
        else:
            folder_name = f"step_{step}"

        checkpoint_dir = os.path.join(self.base_dir, folder_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save PEFT model weights or state dict
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(checkpoint_dir)
        else:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))

        print(f"Checkpoint saved successfully to: {checkpoint_dir}", flush=True)

        if self.wandb_logger:
            self.wandb_logger.log({
                "train/checkpoint": checkpoint_dir,
                "train/checkpoint_step": step,
                "train/checkpoint_epoch": epoch
            }, step=step)
