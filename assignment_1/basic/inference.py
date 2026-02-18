import torch 
from torch import Tensor
from jaxtyping import Float, Int
from . import model
from .train import TransformerLM
from .Tokenizer import Tokenizer
import os


def load_model(config, save_checkpoint_path, device):
    if config.get("d_ff") is None:
        config["d_ff"] = 4*config.get("hidden_dimension")

    LM = TransformerLM(
        vocab_size=config["vocab_size"],
        d_model=config["hidden_dimension"],
        device=device,
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        context_length=config["context_length"],
        rope_theta=config["rope_theta"],
        bias=config["use_bias"],
        remove_rope=config["remove_rope"],
        remove_rmsnorm=config["remove_rmsnorm"],
        use_post_norm=config["use_post_norm"],
    )

    ckpt = torch.load(save_checkpoint_path, map_location=device)
    
    LM.load_state_dict(ckpt["model_state_dict"])
    LM.eval()
    return LM

def sample_next_tokens(logits: Float[Tensor, "..."], softmax: model.Softmax, temperature: float = 1, top_p: float | None = None, generator: torch.Generator | None = None) -> Int[Tensor, "..."]:
    if temperature == 0:
        return torch.argmax(logits, dim= -1)
    
    scaled_logits = logits / temperature
    
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True) # from top to bottom
        sorted_probs = softmax(sorted_logits)
        cum_probs = torch.cumsum(sorted_probs, dim= -1)

        sorted_mask = cum_probs > top_p # marks the first token that crosses p as true meaning it is excluded 
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone() # include the first token to pass p and don't drop it  
        sorted_mask[..., 0] = False # False for keeping and True for dropping
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf")) 

        filtered_logits = torch.full_like(logits, -1e4)
        filtered_logits.scatter_(dim=-1, index= sorted_indices, src= sorted_logits)
        
        scaled_logits = filtered_logits

    probs: Float[torch.Tensor, "V"] = softmax(scaled_logits)
    # return the id of the token linked to the highest probability
    next_token_id: Float[torch.Tensor, "1"] = torch.multinomial(probs, num_samples=1, generator=generator)
    return next_token_id.squeeze(-1)

def generate_token_id(LM:TransformerLM, prompt:str, tokenizer: Tokenizer, temperature, stop_token, max_tokens,top_p:float| None = None)->Int[Tensor, "..."]:
    seq_id = tokenizer.encode(prompt)
    prompt_len = len(seq_id)
    softmax = model.Softmax(dim=-1)
    device = next(LM.parameters()).device
    context_len = int(LM.context_length)
    stop_token_id = tokenizer.encode(stop_token)[0]

    with torch.no_grad():
        for _ in range (max_tokens):
            model_input_ids = seq_id[-context_len:] # sliding window context
            x: Float[Tensor, "T"] = torch.tensor(model_input_ids, dtype=torch.long, device=device)
            logits:Float[Tensor, "V"] = LM(x)
            next_token_id = int(sample_next_tokens(logits = logits[-1, :], temperature=temperature, softmax= softmax, top_p=top_p)) # last position
            seq_id.append(next_token_id)
            if next_token_id == stop_token_id: break
    return seq_id, prompt_len

def generate_text(LM : TransformerLM, 
                  prompt:str, 
                  merge_path: os.PathLike | str, 
                  vocab_path: os.PathLike, 
                  special_tokens: list[str], 
                  temperature: float, 
                  stop_token: str, 
                  max_tokens:int,
                  top_p: float | None = None):
    tokenizer = Tokenizer.from_files(merges_filepath=merge_path,vocab_filepath=vocab_path, special_tokens=special_tokens)
    seq_id, prompt_len =  generate_token_id(LM=LM, prompt=prompt, tokenizer=tokenizer, temperature=temperature, stop_token=stop_token, max_tokens=max_tokens, top_p= top_p)
    return {
        "text":tokenizer.decode(seq_id),
        "prompt_text": tokenizer.decode(seq_id[:prompt_len]),
        "completion_text": tokenizer.decode(seq_id[:prompt_len:]),
        "token_ids": seq_id,
        "completion_tokens_ids": seq_id[prompt_len:],
        "prompt_token_count": prompt_len,
        "completion_token_count": len(seq_id[prompt_len:]),
        "total_token_count": len(seq_id)
        }
