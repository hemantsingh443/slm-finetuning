import torch

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
    do_sample: bool = True,
    device=None,
):
    """Generates text from a prompt using the model and tokenizer."""
    if device is None:
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
