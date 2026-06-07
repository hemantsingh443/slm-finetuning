import torch
import math
from torch.utils.data import DataLoader
from transformers import default_data_collator

def evaluate_model(
    model,
    eval_dataset,
    device,
    batch_size: int = 4,
    length_buckets: list = None,
):
    """Evaluates the model on an evaluation dataset and returns detailed metrics.
    
    Metrics include overall token loss, perplexity, bits per token, sequence loss,
    and bucket-wise breakdowns if length_buckets is specified.
    """
    model.eval()
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    
    if length_buckets is None:
        length_buckets = ["0-128", "128-256", "256-512"]

    # Parse length buckets
    parsed_buckets = []
    for bucket in length_buckets:
        parts = bucket.split("-")
        if len(parts) == 2:
            parsed_buckets.append((int(parts[0]), int(parts[1])))
            
    # Accumulators
    total_sum_loss = 0.0
    total_num_tokens = 0
    all_sequence_losses = []
    
    # Bucket accumulators: key is (min_len, max_len) -> {sum_loss, num_tokens, seq_losses}
    bucket_data = {
        b: {"sum_loss": 0.0, "num_tokens": 0, "seq_losses": []}
        for b in parsed_buckets
    }
    # Fallback bucket for sequences longer than the highest limit
    max_limit = max(b[1] for b in parsed_buckets) if parsed_buckets else 0
    fallback_bucket = (max_limit, float('inf'))
    bucket_data[fallback_bucket] = {"sum_loss": 0.0, "num_tokens": 0, "seq_losses": []}

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_losses = token_losses.view(shift_labels.size(0), shift_labels.size(1))
            
            for i in range(shift_labels.size(0)):
                seq_mask = (shift_labels[i] != -100)
                seq_token_count = seq_mask.sum().item()
                if seq_token_count > 0:
                    seq_loss_sum = token_losses[i][seq_mask].sum().item()
                    seq_loss_avg = seq_loss_sum / seq_token_count
                    
                    # Accumulate overall
                    total_sum_loss += seq_loss_sum
                    total_num_tokens += seq_token_count
                    all_sequence_losses.append(seq_loss_avg)
                    
                    # Accumulate bucket-wise
                    assigned = False
                    for b in parsed_buckets:
                        if b[0] <= seq_token_count < b[1]:
                            bucket_data[b]["sum_loss"] += seq_loss_sum
                            bucket_data[b]["num_tokens"] += seq_token_count
                            bucket_data[b]["seq_losses"].append(seq_loss_avg)
                            assigned = True
                            break
                    if not assigned:
                        bucket_data[fallback_bucket]["sum_loss"] += seq_loss_sum
                        bucket_data[fallback_bucket]["num_tokens"] += seq_token_count
                        bucket_data[fallback_bucket]["seq_losses"].append(seq_loss_avg)

    # Compute overall metrics
    overall_token_loss = total_sum_loss / total_num_tokens if total_num_tokens > 0 else 0.0
    overall_ppl = math.exp(overall_token_loss) if overall_token_loss < 100 and total_num_tokens > 0 else float('inf')
    overall_bpt = overall_token_loss / math.log(2) if total_num_tokens > 0 else 0.0
    overall_seq_loss = sum(all_sequence_losses) / len(all_sequence_losses) if all_sequence_losses else 0.0

    metrics = {
        "overall": {
            "token_loss": overall_token_loss,
            "perplexity": overall_ppl,
            "bits_per_token": overall_bpt,
            "sequence_loss": overall_seq_loss,
        },
        "buckets": {}
    }

    # Compute bucket-wise metrics
    for b, data in bucket_data.items():
        b_sum_loss = data["sum_loss"]
        b_num_tokens = data["num_tokens"]
        b_seq_losses = data["seq_losses"]
        
        b_token_loss = b_sum_loss / b_num_tokens if b_num_tokens > 0 else 0.0
        b_ppl = math.exp(b_token_loss) if b_token_loss < 100 and b_num_tokens > 0 else float('inf')
        b_bpt = b_token_loss / math.log(2) if b_num_tokens > 0 else 0.0
        b_seq_loss = sum(b_seq_losses) / len(b_seq_losses) if b_seq_losses else 0.0
        
        bucket_key = f"{b[0]}-{b[1]}"
        metrics["buckets"][bucket_key] = {
            "token_loss": b_token_loss,
            "perplexity": b_ppl,
            "bits_per_token": b_bpt,
            "sequence_loss": b_seq_loss,
            "samples": len(b_seq_losses)
        }

    return metrics
