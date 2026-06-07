import time
import math
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator

class CausalLMTrainer:
    """Simple training loop for causal LM fine-tuning."""

    def __init__(
        self,
        model,
        train_dataset,
        optimizer,
        device,
        training_config: dict = None,
        wandb_logger=None,
        scheduler=None,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.wandb_logger = wandb_logger
        self.training_config = training_config

        self.config = training_config.get("training", {}) if training_config else {}
        self.batch_size = self.config.get("batch_size", 1)
        self.epochs = self.config.get("epochs", 1)
        self.logging_steps = self.config.get("logging_steps", 10)
        self.grad_clip_norm = self.config.get("max_grad_norm", 1.0)

    def train(self):
        self.model.train()

        # Log static metadata to W&B once
        if self.wandb_logger:
            import wandb
            if wandb.run:
                seed = 42
                if self.training_config and "seed" in self.training_config:
                    seed = self.training_config["seed"]

                param = next(self.model.parameters(), None)
                precision = "fp32"
                if param is not None:
                    if param.dtype == torch.float16:
                        precision = "fp16"
                    elif param.dtype == torch.bfloat16:
                        precision = "bf16"

                lora_rank = None
                if hasattr(self.model, "peft_config"):
                    for config in self.model.peft_config.values():
                        if hasattr(config, "r"):
                            lora_rank = config.r
                            break

                config_update = {
                    "precision": precision,
                    "seed": seed
                }
                if lora_rank is not None:
                    config_update["lora_rank"] = lora_rank

                wandb.config.update(config_update)

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
        )

        global_step = 0
        step_losses = []

        for epoch in range(self.epochs):
            for step, batch in enumerate(dataloader):
                step_start = time.perf_counter()

                self.optimizer.zero_grad(set_to_none=True)

                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm,
                )

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                step_time = time.perf_counter() - step_start

                loss_val = loss.item()
                step_losses.append(loss_val)

                if global_step % self.logging_steps == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    weight_decay = self.optimizer.param_groups[0].get("weight_decay", 0.0)
                    
                    ppl = math.exp(loss_val) if loss_val < 100 else float('inf')
                    bpt = loss_val / math.log(2)

                    tokens_in_batch = batch["input_ids"].numel()
                    tokens_per_sec = tokens_in_batch / step_time if step_time > 0 else 0.0

                    samples_in_batch = batch["input_ids"].shape[0]
                    samples_per_sec = samples_in_batch / step_time if step_time > 0 else 0.0

                    print(f"Epoch {epoch} | Step {global_step} | Loss: {loss_val:.4f} | PPL: {ppl:.4f} | BPT: {bpt:.4f}", flush=True)

                    if self.wandb_logger:
                        self.wandb_logger.log({
                            "train/token_loss": loss_val,
                            "train/loss": loss_val,
                            "train/ppl": ppl,
                            "train/bpt": bpt,
                            "train/lr": lr,
                            "train/weight_decay": weight_decay,
                            "train/grad_norm": float(grad_norm),
                            "train/step_time": step_time,
                            "train/throughput_tokens_per_sec": tokens_per_sec,
                            "train/samples_per_sec": samples_per_sec,
                            "step": global_step,
                            "epoch": epoch,
                        }, step=global_step)

                global_step += 1

        return step_losses

    def evaluate(self, eval_dataset, name: str = None, global_step: int = None):
        """Runs evaluation on eval_dataset and logs metrics to W&B."""
        from src.evaluation.metrics import evaluate_model
        
        metrics = evaluate_model(
            model=self.model,
            eval_dataset=eval_dataset,
            device=self.device,
            batch_size=self.batch_size,
        )
        
        overall = metrics["overall"]
        token_loss = overall["token_loss"]
        ppl = overall["perplexity"]
        bpt = overall["bits_per_token"]
        
        prefix = f"val/{name}/" if name else "val/"
        
        log_dict = {
            f"{prefix}token_loss": token_loss,
            f"{prefix}ppl": ppl,
            f"{prefix}bpt": bpt,
        }
        
        if name:
            log_dict["val/token_loss"] = token_loss
            log_dict["val/ppl"] = ppl
            log_dict["val/bpt"] = bpt

        name_str = f" [{name}]" if name else ""
        print(f"Validation{name_str} | Loss: {token_loss:.4f} | PPL: {ppl:.4f} | BPT: {bpt:.4f}", flush=True)

        if self.wandb_logger:
            log_payload = {**log_dict}
            if global_step is not None:
                log_payload["step"] = global_step
            self.wandb_logger.log(log_payload, step=global_step)

        return metrics